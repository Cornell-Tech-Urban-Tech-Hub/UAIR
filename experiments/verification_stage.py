import os
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import ray
import pandas as pd
from packaging.version import Version

from experiments.verification import (
    init_verification,
    verify_batch_pandas,
    parse_thresholds_string,
)

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verification stage: compute per-chunk/doc verification and augment docs")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--taxonomy_json", type=str, required=True)
    p.add_argument("--chunks_dir", type=str, default=None, help="Override path to taxonomy chunk outputs")
    p.add_argument("--docs_dir", type=str, default=None, help="Override path to docs outputs to augment")
    # Verify params
    p.add_argument("--verify_method", choices=["off", "embed", "nli", "combo", "combo_judge"], default="combo")
    p.add_argument("--verify_top_k", type=int, default=3)
    p.add_argument("--verify_thresholds", type=str, default="sim=0.55,ent=0.85,contra=0.05")
    p.add_argument("--verify_device", type=str, choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--verify_output", type=str, default="verification")
    # Execution
    p.add_argument("--num_cpus", type=int, default=6)
    p.add_argument("--num_gpus", type=int, default=0)
    p.add_argument("--ray_address", type=str, default=None)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_limit", type=int, default=100)
    # W&B
    p.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb_project", type=str, default="sensing-ai-risks")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_prefix", type=str, default="")
    p.add_argument("--wandb_suffix", type=str, default="")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    return p


ARGS = _build_arg_parser().parse_args()


def main() -> None:
    # Resolve run paths
    run_id = ARGS.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(ARGS.output_dir, run_id)
    chunks_dir = ARGS.chunks_dir or os.path.join(run_dir, "chunks")
    docs_dir = ARGS.docs_dir or os.path.join(run_dir, "docs")
    ver_base = os.path.join(run_dir, str(ARGS.verify_output))
    ver_chunks_dir = os.path.join(ver_base, "chunks")
    ver_docs_dir = os.path.join(ver_base, "docs")
    for d in (ver_chunks_dir, ver_docs_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Init Ray
    if not ray.is_initialized():
        if ARGS.ray_address:
            ray.init(address=ARGS.ray_address, log_to_driver=True)
        else:
            ray.init(log_to_driver=True, num_cpus=ARGS.num_cpus, num_gpus=ARGS.num_gpus)
    try:
        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = True
        ctx.execution_options.resource_limits.cpu = int(ARGS.num_cpus)
        ctx.execution_options.resource_limits.gpu = float(ARGS.num_gpus)
    except Exception:
        pass

    # W&B
    if ARGS.wandb_mode != "disabled" and wandb is not None:
        wandb.init(
            project=ARGS.wandb_project,
            entity=ARGS.wandb_entity,
            group=ARGS.wandb_group,
            mode=ARGS.wandb_mode,
            name=f"{ARGS.wandb_prefix}{run_id}{ARGS.wandb_suffix}",
            tags=ARGS.wandb_tags,
            config={
                "stage": "verification",
                "verify_method": ARGS.verify_method,
                "verify_top_k": int(ARGS.verify_top_k),
                "verify_thresholds": ARGS.verify_thresholds,
                "verify_device": ARGS.verify_device,
            },
        )

    # Read taxonomy chunks
    ds_chunks = ray.data.read_parquet(chunks_dir)
    if ARGS.debug:
        try:
            ds_chunks = ds_chunks.limit(int(ARGS.debug_limit))
        except Exception:
            pass

    # Load taxonomy and init verification actor per worker
    taxonomy = json.load(open(ARGS.taxonomy_json))["taxonomy"]
    sim_thr, ent_thr, con_thr = parse_thresholds_string(ARGS.verify_thresholds)
    thresholds = {"sim": sim_thr, "ent": ent_thr, "contra": con_thr}

    class _VerifierActor:
        def __init__(self):
            self._initialized = False
            try:
                init_verification(
                    taxonomy=taxonomy,
                    method=ARGS.verify_method,
                    top_k=int(ARGS.verify_top_k),
                    thresholds=thresholds,
                    device=ARGS.verify_device,
                )
                self._initialized = True
            except Exception as e:
                logging.warning(f"Failed to initialize verification: {e}")

        def __call__(self, _df: pd.DataFrame) -> pd.DataFrame:
            if not self._initialized:
                _df["ver_verified_chunk"] = False
                _df["ver_sim_max"] = None
                _df["ver_nli_ent_max"] = None
                _df["ver_nli_evidence"] = None
                return _df
            try:
                return verify_batch_pandas(_df)
            except Exception as e:
                logging.warning(f"Verification failed: {e}")
                _df["ver_verified_chunk"] = False
                _df["ver_sim_max"] = None
                _df["ver_nli_ent_max"] = None
                _df["ver_nli_evidence"] = None
                return _df

    ds_verified = ds_chunks.map_batches(_VerifierActor, batch_format="pandas")

    # Persist chunk-level verification
    ds_verified.write_parquet(ver_chunks_dir)

    # Aggregate to doc level
    def _reduce_verify(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame([])
        verified_chunk = bool(df.get("ver_verified_chunk", pd.Series([], dtype=bool)).any())
        best_ent = None
        best_evi = None
        max_sim = None
        max_sim_evi = None
        if "ver_nli_ent_max" in df.columns:
            try:
                idx = int(df["ver_nli_ent_max"].astype(float).fillna(-1).idxmax())
                best_ent = float(df.loc[idx, "ver_nli_ent_max"]) if pd.notna(df.loc[idx, "ver_nli_ent_max"]) else None
                best_evi = df.loc[idx, "ver_nli_evidence"] if "ver_nli_evidence" in df.columns else None
            except Exception:
                pass
        if "ver_sim_max" in df.columns:
            try:
                idx2 = int(df["ver_sim_max"].astype(float).fillna(-1).idxmax())
                max_sim = float(df.loc[idx2, "ver_sim_max"]) if pd.notna(df.loc[idx2, "ver_sim_max"]) else None
                max_sim_evi = df.loc[idx2, "ver_top_sent"] if "ver_top_sent" in df.columns else None
            except Exception:
                pass
        out = {
            "article_id": df.get("article_id", pd.Series([None])).iloc[0],
            "verified_doc": bool(verified_chunk),
            "ver_doc_best_ent": best_ent,
            "ver_doc_best_evidence": best_evi,
            "ver_doc_max_sim": max_sim,
            "ver_doc_max_sim_evidence": max_sim_evi,
        }
        return pd.DataFrame([out])

    ver_docs_ds = ds_verified.groupby("article_id").map_groups(_reduce_verify, batch_format="pandas")
    ver_docs_ds.write_parquet(ver_docs_dir)

    # Join with docs parquet and overwrite docs outputs with augmented columns
    try:
        docs_ds = ray.data.read_parquet(docs_dir)
        final_ds = docs_ds.join(ver_docs_ds, on="article_id", how="left")
        # Overwrite docs dir safely
        try:
            shutil.rmtree(docs_dir, ignore_errors=True)
        except Exception:
            pass
        Path(docs_dir).mkdir(parents=True, exist_ok=True)
        final_ds.write_parquet(docs_dir)
    except Exception as e:
        logging.warning(f"Failed to augment docs with verification results: {e}")

    # W&B quick metric
    try:
        if ARGS.wandb_mode != "disabled" and wandb is not None:
            df_docs = ver_docs_ds.to_pandas()
            ver_frac = float(df_docs["verified_doc"].fillna(False).mean()) if len(df_docs) else 0.0
            wandb.log({"verify/docs_verified_fraction": ver_frac})
    except Exception:
        pass


if __name__ == "__main__":
    main()


