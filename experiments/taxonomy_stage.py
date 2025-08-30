import os
import re
import json
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray
import pandas as pd
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from experiments.utils.common import load_structured, start_wandb_heartbeat
from experiments.utils.vllm_config import build_engine_kwargs

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Taxonomy stage: Ray Data + vLLM")
    # IO
    p.add_argument("--input_parquet", type=str, required=True)
    p.add_argument("--taxonomy_json", type=str, required=True)
    p.add_argument("--prompts_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--relevance_dir", type=str, required=True, help="Path to relevance chunks parquet dir")
    # Model
    p.add_argument("--model_source", type=str, required=True)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max_model_len_tokens", type=int, default=8192)
    p.add_argument("--max_output_tokens", type=int, default=8)
    p.add_argument("--safety_margin_tokens", type=int, default=2048)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    p.add_argument("--kv_cache_dtype", type=str, choices=["auto", "fp8_e5m2", "fp8", "fp16", "fp32"], default="auto")
    p.add_argument("--max_num_batched_tokens", type=int, default=3072)
    p.add_argument("--max_num_seqs", type=int, default=4)
    # Execution
    p.add_argument("--ray_address", type=str, default=None)
    p.add_argument("--num_cpus", type=int, default=6)
    p.add_argument("--num_gpus", type=int, default=2)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug_limit", type=int, default=100)
    # Verification
    p.add_argument("--verify_method", choices=["off", "embed", "nli", "combo", "combo_judge"], default="combo")
    p.add_argument("--verify_top_k", type=int, default=3)
    p.add_argument("--verify_thresholds", type=str, default="sim=0.55,ent=0.85,contra=0.05")
    p.add_argument("--verify_device", type=str, choices=["cpu", "cuda"], default="cpu")
    # W&B
    p.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb_project", type=str, default="sensing-ai-risks")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_prefix", type=str, default="")
    p.add_argument("--wandb_suffix", type=str, default="")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb_sample_rows", type=int, default=100)
    p.add_argument("--upload_artifacts", action="store_true")
    return p


ARGS = _build_arg_parser().parse_args()


def _get_tokenizer(model_source: str):
    from transformers import AutoTokenizer  # type: ignore
    return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)


def _create_vllm_config(model_source: str, enable_chunked: bool = True) -> vLLMEngineProcessorConfig:
    ekw = build_engine_kwargs(
        max_model_len=ARGS.max_model_len_tokens,
        gpu_memory_utilization=ARGS.gpu_memory_utilization,
        tensor_parallel_size=ARGS.tensor_parallel_size,
        max_num_batched_tokens=ARGS.max_num_batched_tokens,
        max_num_seqs=ARGS.max_num_seqs,
        kv_cache_dtype=(None if ARGS.kv_cache_dtype == "auto" else ARGS.kv_cache_dtype),
        enable_chunked_prefill=enable_chunked,
        enforce_eager=True,
        tokenizer_pool_size=None,
    )
    return vLLMEngineProcessorConfig(
        model_source=model_source,
        engine_kwargs=ekw,
        concurrency=ARGS.concurrency,
        batch_size=ARGS.batch_size,
    )


def _parse_predicted_label(answer_text: str, num_categories: int) -> str:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return "None"
    if re.search(r"\bnone\b", answer_text, flags=re.IGNORECASE):
        return "None"
    m = re.search(r"\b(\d{1,6})\b", answer_text)
    if m:
        try:
            v = int(m.group(1))
            if 1 <= v <= num_categories:
                return str(v)
        except Exception:
            pass
    return "None"


def main() -> None:
    # Init Ray
    if not ray.is_initialized():
        if ARGS.ray_address:
            ray.init(address=ARGS.ray_address, log_to_driver=True)
        else:
            ray.init(log_to_driver=True, num_cpus=ARGS.num_cpus, num_gpus=ARGS.num_gpus)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_progress_bars = True
    try:
        ctx.execution_options.resource_limits.cpu = int(ARGS.num_cpus)
        ctx.execution_options.resource_limits.gpu = float(ARGS.num_gpus)
    except Exception:
        pass

    # Resolve run output
    run_id = ARGS.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(ARGS.output_dir, run_id)
    chunks_out_dir = os.path.join(run_dir, "chunks")
    docs_out_dir = os.path.join(run_dir, "docs")
    summaries_dir = os.path.join(run_dir, "summaries")
    figures_dir = os.path.join(run_dir, "figures")
    for d in (chunks_out_dir, docs_out_dir, summaries_dir, figures_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

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
                "stage": "taxonomy",
                "input_parquet": ARGS.input_parquet,
                "model_source": ARGS.model_source,
                "tensor_parallel_size": ARGS.tensor_parallel_size,
                "batch_size": ARGS.batch_size,
                "max_model_len_tokens": ARGS.max_model_len_tokens,
            },
        )

    # Load dataset filtered to relevance YES
    rel_ds = ray.data.read_parquet(ARGS.relevance_dir, ray_remote_args={"num_cpus": 0.25})
    def _parse_rel(row: Dict[str, Any]) -> Dict[str, Any]:
        text = str(row.get("relevance_answer", "")).strip().upper()
        row["relevance_label"] = "YES" if re.search(r"\bYES\b", text) else ("NO" if re.search(r"\bNO\b", text) else "NO")
        return row
    rel_ds = rel_ds.map(_parse_rel)
    ds = rel_ds.filter(lambda r: r.get("relevance_label") == "YES")
    try:
        ds = ds.materialize()
    except Exception:
        pass

    # Prompts/taxonomy
    taxonomy = load_structured(ARGS.taxonomy_json)["taxonomy"]
    try:
        prompts = load_structured(ARGS.prompts_json)
    except Exception:
        prompts = {}
    taxonomy_str = "\n".join([f"{\n.join(subcats)}" for _, subcats in taxonomy.items()])
    taxonomy_str = "\n".join([f"{i+1}. {line}" for i, line in enumerate(taxonomy_str.split("\n"))])
    system_prompt = prompts.get("taxonomy_v1", {}).get("system_template") or (
        "You are a news analyst that classifies news articles into one of the following categories of risks from deployed artificial intelligence systems: "
        f"{taxonomy_str}\n\n"
        "You will be given a news article (possibly a chunk of a longer article). "
        "Classify it into one of the categories, or return 'None' if it does not fit into any of the categories, or if it is not about artificial intelligence. "
        "Return only the category number or 'None'."
    )

    cfg = _create_vllm_config(ARGS.model_source, enable_chunked=True)
    vllm_processor = build_llm_processor(
        cfg,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"[article_id={row['article_id']} chunk={row.get('chunk_id', 0)}/{row.get('num_chunks', 1)}]\n{row.get('chunk_text','')}"
                )},
            ],
            sampling_params=dict(
                seed=ARGS.seed,
                temperature=0,
                top_p=1,
                top_k=-1,
                min_p=0,
                presence_penalty=0,
                max_tokens=ARGS.max_output_tokens,
            ),
            article_id=row.get("article_id"),
            article_path=row.get("article_path"),
            chunk_id=row.get("chunk_id"),
            num_chunks=row.get("num_chunks"),
            chunk_text=row.get("chunk_text"),
        ),
        postprocess=lambda row: dict(answer=row["generated_text"], **row),
    )

    hb2_stop, _ = start_wandb_heartbeat(wandb, ray, "taxonomy_stage", dir_to_count=chunks_out_dir)
    t_tax0 = time.perf_counter()
    ds = vllm_processor(ds)
    t_tax1 = time.perf_counter()
    try:
        if hb2_stop is not None:
            hb2_stop.set()
    except Exception:
        pass

    # Parse labels, persist chunks
    num_categories = sum(len(v) for _, v in taxonomy.items())
    ds = ds.map(lambda r: {**r, "chunk_label": _parse_predicted_label(r.get("answer", ""), num_categories)})
    ds.write_parquet(chunks_out_dir)
    final_ds = ds.groupby("article_id").map_groups(_reduce_group, batch_format="pandas")
    final_ds.write_parquet(docs_out_dir)
    final_df = final_ds.to_pandas()

    # Basic summaries/plots
    try:
        taxonomy_list = pd.DataFrame(list(taxonomy.items()), columns=["category", "subcategories"]).explode("subcategories")
        taxonomy_list["subcategory"] = taxonomy_list["subcategories"]
        taxonomy_list = taxonomy_list.drop(columns=["subcategories"]) if "subcategories" in taxonomy_list.columns else taxonomy_list
        taxonomy_list["category_number"] = range(1, len(taxonomy_list) + 1)
        counts = final_df["predicted_category_number"].value_counts(dropna=False)
        counts = counts.rename_axis("predicted_category_number").reset_index(name="estimated_relevant_articles")
        try:
            counts["predicted_category_number"] = counts["predicted_category_number"].astype(int)
        except Exception:
            counts = counts[counts["predicted_category_number"] != "None"]
            counts["predicted_category_number"] = counts["predicted_category_number"].astype(int)
        article_counts = taxonomy_list.merge(
            counts,
            left_on="category_number",
            right_on="predicted_category_number",
            how="left",
        )
        if "predicted_category_number" in article_counts.columns:
            article_counts = article_counts.drop(columns=["predicted_category_number"])
        article_counts["estimated_relevant_articles"] = article_counts["estimated_relevant_articles"].fillna(0)
        article_counts.to_csv(os.path.join(summaries_dir, "article_counts.csv"), index=False)
    except Exception:
        pass

    # Timings
    try:
        run_summary = {
            "run_id": run_id,
            "input_parquet": ARGS.input_parquet,
            "taxonomy_json": ARGS.taxonomy_json,
            "model_source": ARGS.model_source,
            "num_docs": int(len(final_df)),
            "timings": {"taxonomy_s": (t_tax1 - t_tax0)},
        }
        Path(summaries_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(summaries_dir, "taxonomy_summary.json"), "w") as f:
            json.dump(run_summary, f, indent=2)
    except Exception:
        pass


def _reduce_group(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([])
    labels = [str(x) for x in df["chunk_label"].tolist()]
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    if counts:
        def sort_key(item):
            label, count = item
            is_none = 1 if str(label).lower() == "none" else 0
            numeric_val = int(label) if str(label).isdigit() else -1
            return (count, -is_none, numeric_val)
        final_label = max(counts.items(), key=sort_key)[0]
    else:
        final_label = "None"
    non_empty_ids = [x for x in df.get("article_id", pd.Series([], dtype=str)).tolist() if isinstance(x, str) and x]
    non_empty_paths = [x for x in df.get("article_path", pd.Series([], dtype=str)).tolist() if isinstance(x, str) and x]
    return pd.DataFrame([{
        "article_id": (non_empty_ids[0] if non_empty_ids else None),
        "article_path": (non_empty_paths[0] if non_empty_paths else None),
        "predicted_category_number": final_label,
        "num_chunks": int(df["num_chunks"].max()) if "num_chunks" in df else len(labels),
        "chunk_labels": ",".join(labels),
    }])


if __name__ == "__main__":
    main()


