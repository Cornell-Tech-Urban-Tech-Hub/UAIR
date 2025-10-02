# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use Ray Data for data parallel batch inference.

Ray Data is a data processing framework that can process very large datasets
with first-class support for vLLM.

Ray Data provides functionality for:
* Reading and writing to most popular file formats and cloud object storage.
* Streaming execution, so you can run inference on datasets that far exceed
  the aggregate RAM of the cluster.
* Scale up the workload without code changes.
* Automatic sharding, load-balancing, and autoscaling across a Ray cluster,
  with built-in fault-tolerance and retry semantics.
* Continuous batching that keeps vLLM replicas saturated and maximizes GPU
  utilization.
* Compatible with tensor/pipeline parallel inference.

Learn more about Ray Data's LLM integration:
https://docs.ray.io/en/latest/data/working-with-llms.html
"""

import ray
import shutil
import json
import re
import hashlib
from typing import List, Dict, Any, Optional
import threading
import os
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import hashlib as _hashlib
import sys

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover - optional dependency at this stage
    wandb = None

import pandas as pd
import matplotlib.pyplot as plt
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
from processing.verification import (
    init_verification,
    verify_batch_pandas,
    parse_thresholds_string,
)

assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)

# Uncomment to reduce clutter in stdout
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ray Data + vLLM batch inference with optional W&B logging")
    # IO
    parser.add_argument("--input_parquet", type=str, default="/share/pierson/matt/UAIR/data/global_subset/relevant_articles.parquet")
    parser.add_argument("--taxonomy_json", type=str, default="/share/pierson/matt/UAIR/resources/taxonomy_weitz.json")
    parser.add_argument("--keywords_json", type=str, default="/share/pierson/matt/UAIR/resources/keywords.json")
    parser.add_argument("--prompts_json", type=str, default="/share/pierson/matt/UAIR/resources/prompts.json")
    parser.add_argument("--output_dir", type=str, default="/share/pierson/matt/UAIR/data/inferences/")
    parser.add_argument("--stage", choices=["full", "relevance", "taxonomy"], default="full", help="Which stage(s) to run")
    parser.add_argument("--relevance_dir", type=str, default=None, help="Path to previously saved relevance outputs (for stage=taxonomy)")
    # Model / engine
    parser.add_argument("--model_source", type=str, default="/share/pierson/matt/zoo/models/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max_model_len_tokens", type=int, default=8192)
    parser.add_argument("--max_output_tokens", type=int, default=8)
    parser.add_argument("--safety_margin_tokens", type=int, default=2048)
    parser.add_argument("--chunk_overlap_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--conservative_vllm", action="store_true", help="Use conservative vLLM settings to avoid OOM errors")
    parser.add_argument("--kv_cache_dtype", type=str, choices=["auto", "fp8_e5m2", "fp8", "fp16", "fp32"], default="auto", help="KV cache dtype to reduce memory if needed")
    parser.add_argument("--max_num_batched_tokens", type=int, default=None, help="Override vLLM max_num_batched_tokens")
    parser.add_argument("--max_num_seqs", type=int, default=None, help="Override vLLM max_num_seqs")
    # Chunking / flags
    parser.add_argument("--disable_chunking", action="store_true", help="Disable chunking and treat entire article as single chunk")
    parser.add_argument("--disable_keyword_prefilter", action="store_true", help="Skip keyword prefilter stage even if article_text is present")
    parser.add_argument("--prefilter_mode", choices=["pre_gating", "post_gating", "off"], default="pre_gating")
    parser.add_argument("--disable_relevance_pass", action="store_true", help="Skip the binary LLM relevance pass and run taxonomy directly")
    # Execution
    parser.add_argument("--ray_address", type=str, default=None, help="Ray cluster address (e.g., 'auto'). If provided, connect to an existing cluster and do not limit resources in ray.init().")
    parser.add_argument("--num_cpus", type=int, default=6)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_limit", type=int, default=100)
    # W&B
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", type=str, default="sensing-ai-risks")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_prefix", type=str, default="")
    parser.add_argument("--wandb_suffix", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_sample_rows", type=int, default=100)
    parser.add_argument("--upload_artifacts", action="store_true")
    parser.add_argument("--log_article_paths", action="store_true")
    parser.add_argument("--anonymize_paths", action="store_true")
    parser.add_argument("--log_rationales", action="store_true")
    parser.add_argument("--relevance_prompt_id", type=str, default="relevance_v1")
    parser.add_argument("--taxonomy_prompt_id", type=str, default="taxonomy_v1")
    parser.add_argument("--run_id", type=str, default=None)
    # Verification
    parser.add_argument("--verify_method", choices=["off", "embed", "nli", "combo", "combo_judge"], default=os.getenv("VERIFY_METHOD", "combo"))
    parser.add_argument("--verify_top_k", type=int, default=int(os.getenv("VERIFY_TOP_K", "3")))
    parser.add_argument("--verify_thresholds", type=str, default=os.getenv("VERIFY_THRESHOLDS", "sim=0.55,ent=0.85,contra=0.05"))
    parser.add_argument("--verify_output", type=str, default=os.getenv("VERIFY_OUTPUT", "verification"))
    parser.add_argument("--verify_device", type=str, choices=["cpu", "cuda"], default=os.getenv("VERIFY_DEVICE", "cpu"))
    return parser


ARGS = _build_arg_parser().parse_args()


def _sha256_file(path: str, max_bytes: int | None = None) -> str | None:
    try:
        h = _hashlib.sha256()
        with open(path, "rb") as f:
            if max_bytes is None:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            else:
                remaining = max_bytes
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    # Package versions
    try:
        import transformers  # type: ignore
        info["transformers_version"] = getattr(transformers, "__version__", None)
    except Exception:
        info["transformers_version"] = None
    try:
        import vllm  # type: ignore
        info["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        info["vllm_version"] = None
    info["ray_version"] = ray.__version__
    # CUDA / GPU
    try:
        import torch  # type: ignore
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["num_gpus"] = torch.cuda.device_count()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            names = []
            mems = []
            for i in range(torch.cuda.device_count()):
                names.append(torch.cuda.get_device_name(i))
                mems.append(torch.cuda.get_device_properties(i).total_memory)
            info["gpu_names"] = names
            info["gpu_total_memory_bytes"] = mems
        else:
            info["gpu_names"] = []
    except Exception:
        info["cuda_available"] = None
        info["num_gpus"] = None
        info["gpu_names"] = None
    return info


# Lightweight periodic W&B heartbeat to expose progress during long Ray stages
def _start_wandb_heartbeat(stage_name: str, interval_s: float = 15.0, dir_to_count: Optional[str] = None):
    if not (wandb is not None and ARGS.wandb_mode != "disabled" and getattr(wandb, "run", None) is not None):
        return None, None
    stop_event = threading.Event()
    start_ts = time.time()

    def _poll():
        while not stop_event.is_set():
            payload: Dict[str, Any] = {
                "heartbeat/stage": stage_name,
                "heartbeat/elapsed_s": time.time() - start_ts,
            }
            # Cluster resource snapshot
            try:
                avail = ray.available_resources()
                total = ray.cluster_resources()
                payload["resources/available/CPU"] = float(avail.get("CPU", 0.0))
                payload["resources/available/GPU"] = float(avail.get("GPU", 0.0))
                payload["resources/total/CPU"] = float(total.get("CPU", 0.0))
                payload["resources/total/GPU"] = float(total.get("GPU", 0.0))
            except Exception:
                pass
            # Optional directory progress (e.g., number of parquet parts written)
            if dir_to_count:
                try:
                    files = list(Path(dir_to_count).glob("**/*.parquet"))
                    payload["progress/dir"] = dir_to_count
                    payload["progress/parquet_files"] = len(files)
                except Exception:
                    pass
            try:
                wandb.log(payload, commit=True)
            except Exception:
                pass
            stop_event.wait(interval_s)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    return stop_event, t

# Initialize Ray
if not ray.is_initialized():
    if ARGS.ray_address:
        # Connect to existing cluster, do not cap resources in ray.init
        try:
            ray.init(address=ARGS.ray_address, log_to_driver=True)
        except Exception as e:
            logging.error(f"Failed to connect to Ray cluster at {ARGS.ray_address}: {e}")
            raise
    else:
        # Local or head-only mode with explicit limits
        try:
            ray.init(log_to_driver=True, num_cpus=ARGS.num_cpus, num_gpus=ARGS.num_gpus)
        except Exception as e:
            logging.error(f"Failed to initialize Ray locally: {e}")
            raise
else:
    logging.info("Ray already initialized, skipping initialization")

ray.data.DataContext.get_current().enable_progress_bars = True


# Resolve run ID and output directory
RUN_ID = ARGS.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = ARGS.output_dir
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, RUN_ID)
Path(RUN_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Optionally initialize W&B
if ARGS.wandb_mode != "disabled" and wandb is not None:
    wandb.init(
        project=ARGS.wandb_project,
        entity=ARGS.wandb_entity,
        group=ARGS.wandb_group,
        mode=ARGS.wandb_mode,
        name=f"{ARGS.wandb_prefix}{RUN_ID}{ARGS.wandb_suffix}",
        tags=ARGS.wandb_tags,
        config={
            "input_parquet": ARGS.input_parquet,
            "taxonomy_json": ARGS.taxonomy_json,
            "model_source": ARGS.model_source,
            "tensor_parallel_size": ARGS.tensor_parallel_size,
            "batch_size": ARGS.batch_size,
            "max_model_len_tokens": ARGS.max_model_len_tokens,
            "max_output_tokens": ARGS.max_output_tokens,
            "safety_margin_tokens": ARGS.safety_margin_tokens,
            "chunk_overlap_tokens": ARGS.chunk_overlap_tokens,
            "disable_chunking": ARGS.disable_chunking,
            "seed": ARGS.seed,
            "relevance_prompt_id": ARGS.relevance_prompt_id,
            "taxonomy_prompt_id": ARGS.taxonomy_prompt_id,
            "prefilter_mode": ARGS.prefilter_mode,
            "verify_method": ARGS.verify_method,
            "verify_top_k": ARGS.verify_top_k,
            "verify_thresholds": ARGS.verify_thresholds,
        },
    )
    # Log environment info and file hashes as config updates
    env_info = _get_env_info()
    input_hash = _sha256_file(ARGS.input_parquet, max_bytes=64 * 1024 * 1024)
    taxonomy_hash = _sha256_file(ARGS.taxonomy_json)
    wandb.config.update({
        **{f"env/{k}": v for k, v in env_info.items()},
        "hash/input_parquet_sha256_64mb": input_hash,
        "hash/taxonomy_sha256": taxonomy_hash,
    }, allow_val_change=True)


_t0 = time.perf_counter()
try:
    ds = ray.data.read_parquet(ARGS.input_parquet)
    _t_read = time.perf_counter()
    logging.info(f"Successfully loaded parquet from {ARGS.input_parquet}")
except Exception as e:
    logging.error(f"Failed to load parquet file {ARGS.input_parquet}: {e}")
    raise

# only keep the first N rows for debugging if DEBUG
DEBUG = bool(ARGS.debug)
DEBUG_LIMIT = int(ARGS.debug_limit)
if DEBUG:
    # Use limit() to retain a Dataset (take() returns a Python list)
    ds = ds.limit(DEBUG_LIMIT)
    # Show a small sample and schema for the limited dataset
    print(ds.schema())
    ds.show(3)
    
taxonomy = json.load(open(ARGS.taxonomy_json))["taxonomy"]

taxonomy_str = "\n".join([f"{'\n'.join(subcategories)}" for category, subcategories in taxonomy.items()])
taxonomy_str = "\n".join([f"{i+1}. {line}" for i, line in enumerate(taxonomy_str.split("\n"))])

# Load prompts (for relevance and taxonomy)
try:
    PROMPTS = json.load(open(ARGS.prompts_json))
except Exception:
    PROMPTS = {}

# Model source used by both tokenizer and vLLM engine
MODEL_SOURCE = ARGS.model_source

# Number of leaf categories used to validate numeric predictions
NUM_CATEGORIES = sum(len(subcategories) for _, subcategories in taxonomy.items())

# vLLM context window and tokenization-aware chunking
# Keep conservative to avoid KV cache OOM during engine init
MAX_MODEL_LEN_TOKENS = ARGS.max_model_len_tokens
MAX_OUTPUT_TOKENS = ARGS.max_output_tokens
SAFETY_MARGIN_TOKENS = ARGS.safety_margin_tokens
CHUNK_OVERLAP_TOKENS = ARGS.chunk_overlap_tokens

_SYSTEM_PROMPT_TEXT = (
    PROMPTS.get(ARGS.taxonomy_prompt_id, {}).get("system_template")
    or (
        "You are a news analyst that classifies news articles into one of the following categories of risks from deployed artificial intelligence systems: "
        f"{taxonomy_str}\n\n"
        "You will be given a news article (possibly a chunk of a longer article). "
        "Classify it into one of the categories, or return 'None' if it does not fit into any of the categories, or if it is not about artificial intelligence. "
        "Return only the category number or 'None'."
    )
)
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=True, use_fast=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {MODEL_SOURCE}: {e}")
    return _TOKENIZER


def _get_max_user_input_tokens() -> int:
    tok = _get_tokenizer()
    system_tokens = len(tok.encode(_SYSTEM_PROMPT_TEXT, add_special_tokens=False))
    return max(1024, MAX_MODEL_LEN_TOKENS - MAX_OUTPUT_TOKENS - system_tokens - SAFETY_MARGIN_TOKENS)


def _to_blocks(raw_value: Any) -> List[str]:
    """Normalize relevant_blocks (JSON string or list) to a list of strings."""
    if isinstance(raw_value, list):
        return [x for x in raw_value if isinstance(x, str)]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str)]
        except Exception:
            # treat as a single block string if not JSON
            return [raw_value]
    return []


def chunk_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return [text or ""]
    tok = _get_tokenizer()
    input_ids = tok.encode(text, add_special_tokens=False)
    max_user_tokens = _get_max_user_input_tokens()
    if len(input_ids) <= max_user_tokens:
        return [text]
    chunks: List[str] = []
    step = max(1, max_user_tokens - CHUNK_OVERLAP_TOKENS)
    for start in range(0, len(input_ids), step):
        end = min(start + max_user_tokens, len(input_ids))
        piece_ids = input_ids[start:end]
        piece_text = tok.decode(piece_ids, skip_special_tokens=True)
        if piece_text:
            chunks.append(piece_text)
        if end == len(input_ids):
            break
    return chunks


def ensure_article_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure we have a stable article_id and preserve article_path.

    - If article_id missing/empty, derive as sha1(article_path) when available,
      else sha1 of the concatenated relevant_blocks text.
    - We deliberately do not create a separate doc_id; article_id is the key.
    """
    article_id_val = row.get("article_id")
    if not (isinstance(article_id_val, str) and article_id_val):
        article_path_val = row.get("article_path", "")
        if isinstance(article_path_val, str) and article_path_val:
            article_id_val = hashlib.sha1(article_path_val.encode("utf-8")).hexdigest()
        else:
            blocks = _to_blocks(row.get("relevant_blocks", []))
            full_text = "\n\n".join(blocks)
            article_id_val = hashlib.sha1(full_text.encode("utf-8")).hexdigest()
    out_article_path = row.get("article_path", None)
    return {**row, "article_id": article_id_val, "article_path": out_article_path}


def explode_into_chunks(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = _to_blocks(row.get("relevant_blocks", []))
    if blocks:
        text = "\n\n".join(blocks)
    else:
        text = str(row.get("article_text", ""))
    chunks = chunk_text(text)
    exploded: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        exploded.append({
            **row,
            "chunk_id": idx,
            "num_chunks": len(chunks),
            "chunk_text": chunk,
        })
    return exploded


def parse_predicted_label(answer_text: str) -> str:
    if not isinstance(answer_text, str) or not answer_text.strip():
        return "None"
    if re.search(r"\bnone\b", answer_text, flags=re.IGNORECASE):
        return "None"
    match = re.search(r"\b(\d{1,6})\b", answer_text)
    if match:
        try:
            value = int(match.group(1))
            if 1 <= value <= NUM_CATEGORIES:
                return str(value)
        except ValueError:
            pass
    return "None"


def trim_chunk_for_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the chunk text stays within the token budget even after chat templating.

    We already estimated system overhead, but we conservatively re-trim by tokens here.
    """
    text = row.get("chunk_text", "")
    if not isinstance(text, str):
        return row
    tok = _get_tokenizer()
    max_user_tokens = _get_max_user_input_tokens()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) > max_user_tokens:
        ids = ids[:max_user_tokens]
        row["chunk_text"] = tok.decode(ids, skip_special_tokens=True)
    return row


def add_chunk_token_count(row: Dict[str, Any]) -> Dict[str, Any]:
    text = row.get("chunk_text", "")
    if isinstance(text, str) and text:
        tok = _get_tokenizer()
        row["chunk_token_count"] = len(tok.encode(text, add_special_tokens=False))
    else:
        row["chunk_token_count"] = 0
    return row


# ---------- Helper utilities for outputs, summaries, and plots ----------
def make_output_dirs(base_dir: str) -> Dict[str, str]:
    chunks_dir = os.path.join(base_dir, "chunks")
    docs_dir = os.path.join(base_dir, "docs")
    summaries_dir = os.path.join(base_dir, "summaries")
    figures_dir = os.path.join(base_dir, "figures")
    for d in [chunks_dir, docs_dir, summaries_dir, figures_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return {"chunks": chunks_dir, "docs": docs_dir, "summaries": summaries_dir, "figures": figures_dir}


def generate_category_counts(final_df: pd.DataFrame, taxonomy: Dict[str, list]) -> pd.DataFrame:
    taxonomy_list = pd.DataFrame(list(taxonomy.items()), columns=["category", "subcategories"]).explode("subcategories")
    taxonomy_list["subcategory"] = taxonomy_list["subcategories"]
    if "subcategories" in taxonomy_list.columns:
        taxonomy_list = taxonomy_list.drop(columns=["subcategories"])
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
    return article_counts


def plot_and_save_category_counts(article_counts: pd.DataFrame, figures_dir: str) -> str:
    plot_df = article_counts.sort_values("estimated_relevant_articles", ascending=True)
    num_bars = len(plot_df)
    fig_height = min(max(num_bars * 0.35, 6), 30)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.barh(plot_df["subcategory"], plot_df["estimated_relevant_articles"], color="#4C78A8")
    ax.set_xlabel("Number of Estimated Relevant Articles", fontsize=16)
    ax.set_ylabel("Subcategory", fontsize=16)
    ax.set_title("Number of Estimated Relevant Articles per Subcategory", fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.invert_yaxis()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='x', labelsize=14)
    plt.subplots_adjust(left=0.4, right=0.98)
    fig_path = os.path.join(figures_dir, "predicted_articles_per_subcategory.png")
    plt.savefig(fig_path, pad_inches=0.25, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def plot_and_save_tokens_hist(chunks_sample_df: pd.DataFrame | None, figures_dir: str) -> str | None:
    try:
        if chunks_sample_df is not None and "chunk_token_count" in chunks_sample_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(chunks_sample_df["chunk_token_count"], bins=50, color="#72B7B2")
            ax.set_title("Token Count per Chunk (sampled)")
            ax.set_xlabel("Tokens per chunk")
            ax.set_ylabel("Frequency")
            out = os.path.join(figures_dir, "hist_tokens_per_chunk.png")
            plt.savefig(out, pad_inches=0.25, bbox_inches='tight')
            plt.close(fig)
            return out
    except Exception:
        return None
    return None


def plot_and_save_chunks_hist(final_df: pd.DataFrame, figures_dir: str) -> str | None:
    try:
        if "num_chunks" in final_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(final_df["num_chunks"], bins=50, color="#E45756")
            ax.set_title("Chunks per Article")
            ax.set_xlabel("Chunks")
            ax.set_ylabel("Number of articles")
            out = os.path.join(figures_dir, "hist_chunks_per_article.png")
            plt.savefig(out, pad_inches=0.25, bbox_inches='tight')
            plt.close(fig)
            return out
    except Exception:
        return None
    return None


def plot_and_save_per_country_counts(final_df: pd.DataFrame, figures_dir: str) -> str | None:
    try:
        per_country_total = final_df.groupby("country_code").size().reset_index(name="num_articles")
        per_country_total = per_country_total.sort_values("num_articles", ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.barh(per_country_total["country_code"], per_country_total["num_articles"], color="#4C78A8")
        ax.set_xlabel("Articles")
        ax.set_ylabel("Country")
        ax.set_title("Articles per Country (Top 20)")
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        out = os.path.join(figures_dir, "articles_per_country_top20.png")
        plt.savefig(out, pad_inches=0.25, bbox_inches='tight')
        plt.close(fig)
        return out
    except Exception:
        return None


def plot_and_save_per_country_stacked(final_df: pd.DataFrame, article_counts: pd.DataFrame, figures_dir: str) -> str | None:
    try:
        cat_counts_global = article_counts.sort_values("estimated_relevant_articles", ascending=False)
        top_categories = cat_counts_global["category_number"].head(10).tolist() if "category_number" in cat_counts_global.columns else []
        tmp = final_df.copy()
        tmp = tmp[tmp["predicted_category_number"] != "None"]
        tmp["predicted_category_number"] = tmp["predicted_category_number"].astype(int)
        if top_categories:
            tmp = tmp[tmp["predicted_category_number"].isin(top_categories)]
        top_countries = tmp["country_code"].value_counts().head(20).index.tolist()
        tmp = tmp[tmp["country_code"].isin(top_countries)]
        pivot = tmp.pivot_table(index="country_code", columns="predicted_category_number", values="article_id", aggfunc="count", fill_value=0)
        pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False)
        fig, ax = plt.subplots(figsize=(18, 10))
        bottom = None
        for col in sorted(pivot.columns):
            vals = pivot[col].values
            if bottom is None:
                ax.bar(pivot.index, vals, label=str(col))
                bottom = vals
            else:
                ax.bar(pivot.index, vals, bottom=bottom, label=str(col))
                bottom = bottom + vals
        ax.set_title("Per-Country Taxonomy Distribution (Top 20 countries x Top 10 categories)")
        ax.set_xlabel("Country")
        ax.set_ylabel("Articles")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        out = os.path.join(figures_dir, "per_country_taxonomy_stacked.png")
        plt.savefig(out, pad_inches=0.25, bbox_inches='tight')
        plt.close(fig)
        return out
    except Exception:
        return None


def compute_per_country_relevance_rates(input_parquet: str, final_df: pd.DataFrame, summaries_dir: str) -> str | None:
    try:
        per_country_total_in = ray.data.read_parquet(input_parquet).to_pandas()["article_path"].apply(
            lambda p: _derive_country(p)
        ).value_counts().rename_axis("country_code").reset_index(name="total_articles")
        per_country_yes = final_df.groupby("country_code").size().rename("yes_articles").reset_index()
        per_country_rel = per_country_total_in.merge(per_country_yes, how="left", on="country_code")
        per_country_rel["yes_articles"] = per_country_rel["yes_articles"].fillna(0).astype(int)
        per_country_rel["relevance_rate"] = per_country_rel["yes_articles"] / per_country_rel["total_articles"].replace({0: 1})
        per_country_rel = per_country_rel.sort_values("total_articles", ascending=False)
        out_csv = os.path.join(summaries_dir, "per_country_relevance_rates.csv")
        per_country_rel.to_csv(out_csv, index=False)
        return out_csv
    except Exception:
        return None

if DEBUG:
    size = ds.count()
    print(f"Size of dataset: {size} rows")

# Configure vLLM engine with robust settings
def create_vllm_config(gpu_memory_util: float = None, enable_chunked: bool = True) -> vLLMEngineProcessorConfig:
    def _create(
        gpu_memory_util_local: float | None,
        enable_chunked_local: bool,
        max_batched_tokens_local: int | None,
        max_seqs_local: int | None,
        force_eager_local: bool,
    ) -> vLLMEngineProcessorConfig:
        if gpu_memory_util_local is None:
            gpu_memory_util_local = min(0.85, ARGS.gpu_memory_utilization)

        engine_kwargs: Dict[str, Any] = {
            "max_model_len": MAX_MODEL_LEN_TOKENS,
            "gpu_memory_utilization": gpu_memory_util_local,
            "tensor_parallel_size": ARGS.tensor_parallel_size,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }

        # KV cache dtype override (to save memory)
        if getattr(ARGS, "kv_cache_dtype", "auto") and ARGS.kv_cache_dtype != "auto":
            engine_kwargs["kv_cache_dtype"] = ARGS.kv_cache_dtype

        # Chunked prefill knobs
        if enable_chunked_local and gpu_memory_util_local <= 0.8:
            engine_kwargs["enable_chunked_prefill"] = True
        if max_batched_tokens_local is not None:
            engine_kwargs["max_num_batched_tokens"] = int(max_batched_tokens_local)
        if max_seqs_local is not None:
            engine_kwargs["max_num_seqs"] = int(max_seqs_local)

        # Eager mode reduces graph capture memory overhead
        if force_eager_local or DEBUG:
            engine_kwargs["enforce_eager"] = True

        return vLLMEngineProcessorConfig(
            model_source=MODEL_SOURCE,
            engine_kwargs=engine_kwargs,
            concurrency=ARGS.concurrency,
            batch_size=ARGS.batch_size,
        )

    # Wrapper for backwards compatibility with existing calls
    return _create(
        gpu_memory_util,
        enable_chunked,
        ARGS.max_num_batched_tokens if ARGS.max_num_batched_tokens is not None else 4096,
        ARGS.max_num_seqs if ARGS.max_num_seqs is not None else (8 if enable_chunked else 4),
        False,
    )

# Try progressive fallback configurations
config = None
if ARGS.conservative_vllm:
    # Skip to very conservative settings if explicitly requested
    config_attempts = [
        (0.50, False, 2048, 2, True, "conservative configuration (requested)"),
    ]
    logging.info("Using conservative vLLM settings as requested")
else:
    config_attempts = [
        # Attempt 1: Primary configuration (may enable chunked prefill)
        (min(ARGS.gpu_memory_utilization, 0.85), True, 4096, 8, False, "primary configuration"),
        # Attempt 2: Reduced memory footprint, disable chunked prefill, reduce concurrency
        (0.65, False, 3072, 4, True, "reduced memory configuration"),
        # Attempt 3: Conservative settings
        (0.50, False, 2048, 2, True, "conservative configuration"),
        # Attempt 4: Very conservative for crowded GPUs
        (0.30, False, 1536, 1, True, "very conservative configuration"),
    ]

for gpu_mem, chunked, max_tok, max_seqs, force_eager, desc in config_attempts:
    try:
        logging.info(f"Attempting vLLM configuration: {desc} (gpu_mem={gpu_mem}, chunked={chunked})")
        # Allow per-attempt overrides for batch/seq settings and eager mode
        def _mk_cfg():
            return vLLMEngineProcessorConfig(
                model_source=MODEL_SOURCE,
                engine_kwargs={
                    **create_vllm_config(gpu_mem, chunked).engine_kwargs,  # type: ignore[attr-defined]
                    "max_num_batched_tokens": max_tok,
                    "max_num_seqs": max_seqs,
                    **({"enforce_eager": True} if force_eager else {}),
                },
                concurrency=ARGS.concurrency,
                batch_size=ARGS.batch_size,
            )
        config = _mk_cfg()
        # Test that the configuration is valid by creating the processor
        # Note: This doesn't actually initialize engines until data processing
        vllm_processor = build_llm_processor(
            config,
            preprocess=lambda row: dict(
                messages=(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT_TEXT},
                        {"role": "user", "content": (
                            f"[article_id={row['article_id']} chunk={row.get('chunk_id', 0)}/{row.get('num_chunks', 1)}]\n{row['chunk_text']}"
                        )},
                    ]
                ),
                sampling_params=dict(
                    seed=ARGS.seed,
                    temperature=0,
                    top_p=1,
                    top_k=-1,
                    min_p=0,
                    presence_penalty=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                ),
                # Explicitly pass through fields we need after inference
                article_id=row.get("article_id"),
                article_path=row.get("article_path"),
                chunk_id=row.get("chunk_id"),
                num_chunks=row.get("num_chunks"),
                chunk_text=row.get("chunk_text"),
            ),
            postprocess=lambda row: dict(
                answer=row["generated_text"],
                **row,  # This will return all the original columns in the dataset.
            ),
        )
        logging.info(f"Successfully created vLLM processor with {desc}")
        break
    except Exception as e:
        logging.warning(f"Failed to create vLLM config with {desc}: {e}")
        if desc == "conservative configuration":
            # Last attempt failed
            raise RuntimeError(f"All vLLM configuration attempts failed. Last error: {e}")
        continue

if config is None:
    raise RuntimeError("Failed to create any valid vLLM configuration")

# vllm_processor was already created in the configuration attempts above

# Optional relevance pass processor
relevance_prompt = PROMPTS.get(ARGS.relevance_prompt_id, {})
_RELEVANCE_SYSTEM = relevance_prompt.get("system", "You are a careful news analyst trained to decide if a news article is related to artificial intelligence (AI) in any substantive way.")
_RELEVANCE_USER_TMPL = relevance_prompt.get(
    "user_template",
    "Read the following article chunk and answer strictly YES or NO to the question: Is this article related to artificial intelligence?\n\n[article_id={article_id} chunk={chunk_id}/{num_chunks}]\n{chunk_text}",
)

# Create relevance processor using the same robust configuration
try:
    relevance_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": _RELEVANCE_SYSTEM},
                {"role": "user", "content": _RELEVANCE_USER_TMPL.format(
                    article_id=row.get("article_id"),
                    chunk_id=row.get("chunk_id", 0),
                    num_chunks=row.get("num_chunks", 1),
                    chunk_text=row.get("chunk_text", ""),
                )},
            ],
            sampling_params=dict(
                seed=ARGS.seed,
                temperature=0,
                top_p=1,
                top_k=-1,
                min_p=0,
                presence_penalty=0,
                max_tokens=(32 if ARGS.log_rationales else 8),
            ),
            article_id=row.get("article_id"),
            article_path=row.get("article_path"),
            chunk_id=row.get("chunk_id"),
            num_chunks=row.get("num_chunks"),
            chunk_text=row.get("chunk_text"),
        ),
        postprocess=lambda row: dict(relevance_answer=row["generated_text"], **row),
    )
    logging.info("Successfully created relevance processor")
except Exception as e:
    logging.error(f"Failed to create relevance processor: {e}")
    raise RuntimeError(f"Could not create relevance processor with current vLLM configuration: {e}")

# Toggle chunking via CLI or environment variable
DISABLE_CHUNKING = bool(ARGS.disable_chunking) or (str(os.getenv("DISABLE_CHUNKING", "0")).lower() in {"1", "true", "yes", "on"})

# Prepare dataset: attach doc ids and conditionally explode to chunks, then run inference
ds = ds.map(ensure_article_keys)

# Derive country code early from article_path for downstream stats
def _derive_country_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    path = row.get("article_path")
    try:
        m = re.search(r"/global_subset/([^/]+)/", str(path))
        row["country_code"] = m.group(1) if m else "unknown"
    except Exception:
        row["country_code"] = "unknown"
    return row

ds = ds.map(_derive_country_from_row)

# Keyword prefilter: only if not disabled and article_text exists
try:
    input_fields = [f.name for f in ds.schema().fields]
except Exception:
    input_fields = []

if (not ARGS.disable_keyword_prefilter) and ("article_text" in input_fields):
    try:
        with open(ARGS.keywords_json, "r") as f:
            kw = json.load(f)
        method_terms = [str(t).lower() for t in kw.get("methods", [])]
        domain_terms = [str(t).lower() for t in kw.get("domains", [])]
    except Exception:
        method_terms, domain_terms = [], []

    if method_terms or domain_terms:
        def _keyword_map(row: Dict[str, Any]) -> Dict[str, Any]:
            text = str(row.get("article_text", "")).lower()
            method_hits = any(term in text for term in method_terms)
            domain_hits = any(term in text for term in domain_terms)
            row["relevant_keyword"] = bool(method_hits and domain_hits)
            return row

        ds = ds.map(_keyword_map)
        # Intermediate W&B logging for keyword prefilter (pre-gating)
        if ARGS.wandb_mode != "disabled" and wandb is not None and getattr(wandb, "run", None) is not None:
            try:
                _sample_n = 2000
                sample_df = ds.random_shuffle().limit(_sample_n).to_pandas()
                if "relevant_keyword" in sample_df.columns and len(sample_df) > 0:
                    yes_frac = float(sample_df["relevant_keyword"].mean())
                    yes_cnt = int(sample_df["relevant_keyword"].sum())
                    wandb.log({
                        "keyword/sample_n": int(len(sample_df)),
                        "keyword/sample_yes_count": yes_cnt,
                        "keyword/sample_yes_fraction": yes_frac,
                    })
                    prev_cols = [c for c in ["article_id", "country_code", "relevant_keyword"] if c in sample_df.columns]
                    if prev_cols:
                        wandb.log({"tables/keyword_sample": wandb.Table(dataframe=sample_df[prev_cols].head(200))})
            except Exception:
                pass
        if ARGS.prefilter_mode == "pre_gating":
            ds = ds.filter(lambda r: bool(r.get("relevant_keyword", True)))
            # Post-gating keyword logging (sample-based)
            if ARGS.wandb_mode != "disabled" and wandb is not None and getattr(wandb, "run", None) is not None:
                try:
                    _post_sample_df = ds.random_shuffle().limit(2000).to_pandas()
                    wandb.log({
                        "keyword/post_gate_sample_n": int(len(_post_sample_df)),
                    })
                    prev_cols2 = [c for c in ["article_id", "country_code", "relevant_keyword"] if c in _post_sample_df.columns]
                    if prev_cols2:
                        wandb.log({"tables/keyword_post_gate_sample": wandb.Table(dataframe=_post_sample_df[prev_cols2].head(200))})
                except Exception:
                    pass
_t_keyword = time.perf_counter()

if DISABLE_CHUNKING:
    # Treat the entire article as a single chunk
    def _mk_single_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
        blocks = _to_blocks(row.get("relevant_blocks", []))
        text_val = ("\n\n".join(blocks)) if blocks else str(row.get("article_text", ""))
        return {**row, "chunk_id": 0, "num_chunks": 1, "chunk_text": text_val}

    ds = ds.map(_mk_single_chunk)
else:
    ds = ds.flat_map(explode_into_chunks)

ds = ds.map(trim_chunk_for_prompt)
ds = ds.map(add_chunk_token_count)
_t_chunk = time.perf_counter()

# Optional relevance pass (or load from prior run)
if ARGS.stage == "taxonomy" and ARGS.relevance_dir:
    # Load precomputed relevance outputs and gate to YES to minimize GPU pressure before taxonomy
    try:
        ds_rel = ray.data.read_parquet(ARGS.relevance_dir)
    except Exception as e:
        logging.error(f"Failed to read relevance_dir={ARGS.relevance_dir}: {e}")
        raise
    def _parse_rel(row: Dict[str, Any]) -> Dict[str, Any]:
        text = str(row.get("relevance_answer", "")).strip().upper()
        label = "YES" if re.search(r"\bYES\b", text) else ("NO" if re.search(r"\bNO\b", text) else "NO")
        if ARGS.log_rationales:
            m = re.search(r"Rationale\s*:\s*(.*)$", str(row.get("relevance_answer", "")), flags=re.IGNORECASE | re.DOTALL)
            if m:
                row["relevance_rationale"] = m.group(1).strip()
        row["relevance_label"] = label
        return row
    ds_rel = ds_rel.map(_parse_rel)
    ds = ds_rel.filter(lambda r: r.get("relevance_label") == "YES")
    try:
        ds = ds.materialize()
    except Exception:
        pass
elif not ARGS.disable_relevance_pass:
    hb_stop, hb_thread = _start_wandb_heartbeat("relevance_stage")
    ds_rel = relevance_processor(ds)
    # Robust barrier: persist and reload to ensure relevance engine actors are torn down
    _relevance_out_dir = os.path.join(RUN_OUTPUT_DIR, "relevance", "chunks")
    try:
        Path(_relevance_out_dir).mkdir(parents=True, exist_ok=True)
        # Use write with explicit error handling
        try:
            ds_rel.write_parquet(_relevance_out_dir)
            ds_rel = ray.data.read_parquet(_relevance_out_dir)
        except Exception as e:
            logging.warning(f"Failed to persist relevance results to {_relevance_out_dir}: {e}")
            # Force materialization to complete the stage
            ds_rel = ds_rel.materialize()
    except Exception as e:
        logging.warning(f"Failed to create relevance tmp directory: {e}")
        # As a fallback, materialize to force completion of the stage
        try:
            ds_rel = ds_rel.materialize()
        except Exception as mat_e:
            logging.error(f"Failed to materialize relevance dataset: {mat_e}")
            raise
    def _parse_rel(row: Dict[str, Any]) -> Dict[str, Any]:
        text = str(row.get("relevance_answer", "")).strip().upper()
        label = "YES" if re.search(r"\bYES\b", text) else ("NO" if re.search(r"\bNO\b", text) else "NO")
        if ARGS.log_rationales:
            # Try to capture trailing rationale in a simple way
            m = re.search(r"Rationale\s*:\s*(.*)$", str(row.get("relevance_answer", "")), flags=re.IGNORECASE | re.DOTALL)
            if m:
                row["relevance_rationale"] = m.group(1).strip()
        row["relevance_label"] = label
        return row
    ds_rel = ds_rel.map(_parse_rel)
    # Intermediate W&B logging for relevance stage (label distribution and sample rows)
    if ARGS.wandb_mode != "disabled" and wandb is not None and getattr(wandb, "run", None) is not None:
        try:
            rel_counts_df = ds_rel.groupby("relevance_label").count().to_pandas()
            total_chunks = int(rel_counts_df["count"].sum()) if "count" in rel_counts_df.columns else None
            yes_cnt = int(rel_counts_df.loc[rel_counts_df.get("relevance_label") == "YES", "count"].sum()) if total_chunks is not None else None
            no_cnt = int(rel_counts_df.loc[rel_counts_df.get("relevance_label") == "NO", "count"].sum()) if total_chunks is not None else None
            payload = {}
            if total_chunks is not None:
                payload["relevance/total_chunks"] = total_chunks
            if yes_cnt is not None:
                payload["relevance/yes_chunks"] = yes_cnt
            if no_cnt is not None:
                payload["relevance/no_chunks"] = no_cnt
            if total_chunks and yes_cnt is not None:
                payload["relevance/yes_fraction"] = (yes_cnt / max(total_chunks, 1))
            if payload:
                wandb.log(payload)
            # Log a small sample table of relevance outputs
            try:
                sample_n = min(200, total_chunks or 0)
                if sample_n > 0:
                    sample_df = ds_rel.limit(sample_n).to_pandas()
                    cols = [c for c in ["article_id", "chunk_id", "country_code", "relevance_label", ("relevance_answer" if ARGS.log_rationales else None)] if c and c in sample_df.columns]
                    if cols:
                        wandb.log({"tables/relevance_sample": wandb.Table(dataframe=sample_df[cols])})
            except Exception:
                pass
        except Exception:
            pass
    # Gate to YES only (post_gating means evaluate on all; filter after relevance)
    ds = ds_rel.filter(lambda r: r.get("relevance_label") == "YES")
    # Second barrier to free any lingering relevance compute resources
    try:
        ds = ds.materialize()
    except Exception as e:
        logging.warning(f"Failed to materialize filtered relevance dataset: {e}")
        # Continue without materialization
    # Stop relevance heartbeat
    try:
        if hb_stop is not None:
            hb_stop.set()
    except Exception:
        pass
    # If only running relevance stage, exit early to ensure vLLM actors are freed
    if ARGS.stage == "relevance":
        logging.info("Relevance stage completed; exiting before taxonomy stage as requested.")
        raise SystemExit(0)
_t_rel = time.perf_counter()

# Taxonomy pass with enhanced error handling
hb2_stop, hb2_thread = _start_wandb_heartbeat("taxonomy_stage")
try:
    logging.info("Starting taxonomy inference with vLLM...")
    ds = vllm_processor(ds)
    logging.info("Successfully completed taxonomy inference")
except Exception as e:
    _t_tax = time.perf_counter()
    try:
        if hb2_stop is not None:
            hb2_stop.set()
    except Exception:
        pass
    
    error_msg = str(e)
    if "Engine core initialization failed" in error_msg:
        logging.error(f"""
vLLM Engine initialization failed. This is likely due to:

1. **GPU Memory Issue**: The model is too large for available GPU memory.
   - Current GPU memory utilization: {ARGS.gpu_memory_utilization}
   - Model: {MODEL_SOURCE}
   - Tensor parallel size: {ARGS.tensor_parallel_size}
   - Max model length: {MAX_MODEL_LEN_TOKENS}

2. **Model Loading Issue**: The model path may be incorrect or inaccessible.
   - Check that {MODEL_SOURCE} exists and is accessible

3. **GPU Resource Conflict**: Another process may be using the GPU.
   - Check `nvidia-smi` for GPU usage

**Suggestions:**
- Try reducing --gpu_memory_utilization (e.g., 0.7 or 0.6)
- Try reducing --max_model_len_tokens
- Try --tensor_parallel_size 1 if you have memory constraints
- Use --debug mode to test with small data first

Original error: {error_msg}
""")
    else:
        logging.error(f"Taxonomy inference failed with error: {error_msg}")
    
    raise RuntimeError(f"vLLM taxonomy inference failed: {error_msg}")

_t_tax = time.perf_counter()
try:
    if hb2_stop is not None:
        hb2_stop.set()
except Exception:
    pass

# Parse per-chunk labels
ds = ds.map(lambda row: {**row, "chunk_label": parse_predicted_label(row.get("answer", ""))})

# Persist chunk-level outputs
chunks_out_dir = os.path.join(RUN_OUTPUT_DIR, "chunks")
docs_out_dir = os.path.join(RUN_OUTPUT_DIR, "docs")
summaries_dir = os.path.join(RUN_OUTPUT_DIR, "summaries")
figures_dir = os.path.join(RUN_OUTPUT_DIR, "figures")
Path(chunks_out_dir).mkdir(parents=True, exist_ok=True)
Path(docs_out_dir).mkdir(parents=True, exist_ok=True)
Path(summaries_dir).mkdir(parents=True, exist_ok=True)
Path(figures_dir).mkdir(parents=True, exist_ok=True)

hb3_stop, hb3_thread = _start_wandb_heartbeat("write_chunks", dir_to_count=chunks_out_dir)
try:
    ds.write_parquet(chunks_out_dir)
except Exception:
    pass
try:
    if hb3_stop is not None:
        hb3_stop.set()
except Exception:
    pass

# Aggregate per-document via majority vote
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
    # Choose first non-empty values for metadata across chunks
    if "article_id" in df.columns:
        non_empty_ids = [x for x in df["article_id"].tolist() if isinstance(x, str) and x]
    else:
        non_empty_ids = []
    if "article_path" in df.columns:
        non_empty_paths = [x for x in df["article_path"].tolist() if isinstance(x, str) and x]
    else:
        logging.warning("No article_path column found in dataset.")
        non_empty_paths = []
    article_id_val = (non_empty_ids[0] if non_empty_ids else None)
    article_path_val = (non_empty_paths[0] if non_empty_paths else None)
    out = {
        "article_id": article_id_val,
        "article_path": article_path_val,
        "predicted_category_number": final_label,
        "num_chunks": int(df["num_chunks"].max()) if "num_chunks" in df else len(labels),
        "chunk_labels": ",".join(labels),
    }
    return pd.DataFrame([out])

hb4_stop, hb4_thread = _start_wandb_heartbeat("aggregate_docs")
final_ds = ds.groupby("article_id").map_groups(_reduce_group, batch_format="pandas")
_t_agg = time.perf_counter()
try:
    if hb4_stop is not None:
        hb4_stop.set()
except Exception:
    pass

# ---------------- Verification stage (chunk-level then doc-level) ----------------
if ARGS.verify_method != "off":
    try:
        # Initialize models on each worker lazily via a small warmup map
        _sim, _ent, _con = parse_thresholds_string(ARGS.verify_thresholds)
        thresholds = {"sim": _sim, "ent": _ent, "contra": _con}
        # Work from persisted chunk outputs to avoid recomputing upstream lineage
        ds_verify = None
        try:
            ds_verify = ray.data.read_parquet(chunks_out_dir)
        except Exception:
            ds_verify = ds  # fallback, may recompute lineage
        # Stateful actor so models are loaded once per worker
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
                    pass
            def __call__(self, _df: pd.DataFrame) -> pd.DataFrame:
                if not self._initialized:
                    # Return dataframe with empty verification columns
                    _df["ver_verified_chunk"] = False
                    _df["ver_sim_max"] = None
                    _df["ver_nli_ent_max"] = None
                    _df["ver_nli_evidence"] = None
                    return _df
                try:
                    return verify_batch_pandas(_df)
                except Exception as e:
                    logging.warning(f"Verification failed: {e}")
                    # Return original dataframe with verification defaults
                    _df["ver_verified_chunk"] = False
                    _df["ver_sim_max"] = None
                    _df["ver_nli_ent_max"] = None  
                    _df["ver_nli_evidence"] = None
                    return _df
        # Compute verification features per chunk on persisted chunk outputs
        ds_verify = ds_verify.map_batches(_VerifierActor, batch_format="pandas")
    except Exception:
        pass

    # Aggregate verification to document-level
    def _reduce_verify(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame([])
        verified_chunk = bool(df.get("ver_verified_chunk", pd.Series([], dtype=bool)).any())
        best_ent = None
        best_evi = None
        max_sim = None
        max_sim_evi = None
        if "ver_nli_ent_max" in df.columns:
            # pick max entailment evidence
            try:
                idx = int(df["ver_nli_ent_max"].astype(float).fillna(-1).idxmax())
                best_ent = float(df.loc[idx, "ver_nli_ent_max"]) if not pd.isna(df.loc[idx, "ver_nli_ent_max"]) else None
                best_evi = df.loc[idx, "ver_nli_evidence"] if "ver_nli_evidence" in df.columns else None
            except Exception:
                pass
        if "ver_sim_max" in df.columns:
            try:
                idx2 = int(df["ver_sim_max"].astype(float).fillna(-1).idxmax())
                max_sim = float(df.loc[idx2, "ver_sim_max"]) if not pd.isna(df.loc[idx2, "ver_sim_max"]) else None
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

    try:
        ver_docs_ds = (ds_verify if 'ds_verify' in locals() and ds_verify is not None else ds).groupby("article_id").map_groups(_reduce_verify, batch_format="pandas")
    except Exception as e:
        logging.warning(f"Failed to aggregate verification data: {e}")
        # Create empty verification dataset as fallback
        ver_docs_ds = ray.data.from_items([{"article_id": None, "verified_doc": False}]).limit(0)

    # Optionally persist verification outputs
    try:
        _ver_out_base = os.path.join(RUN_OUTPUT_DIR, str(ARGS.verify_output))
        _ver_chunks_dir = os.path.join(_ver_out_base, "chunks")
        _ver_docs_dir = os.path.join(_ver_out_base, "docs")
        Path(_ver_chunks_dir).mkdir(parents=True, exist_ok=True)
        Path(_ver_docs_dir).mkdir(parents=True, exist_ok=True)
        hbv1_stop, hbv1_thread = _start_wandb_heartbeat("write_verify_chunks", dir_to_count=_ver_chunks_dir)
        try:
            (ds_verify if 'ds_verify' in locals() and ds_verify is not None else ds).write_parquet(_ver_chunks_dir)
        except Exception:
            pass
        try:
            if hbv1_stop is not None:
                hbv1_stop.set()
        except Exception:
            pass
        hbv2_stop, hbv2_thread = _start_wandb_heartbeat("write_verify_docs", dir_to_count=_ver_docs_dir)
        try:
            ver_docs_ds.write_parquet(_ver_docs_dir)
        except Exception:
            pass
        try:
            if hbv2_stop is not None:
                hbv2_stop.set()
        except Exception:
            pass
    except Exception:
        pass
    # Join verification doc flags back to final_ds on article_id
    try:
        final_ds = final_ds.join(ver_docs_ds, on="article_id", how="left")
    except Exception:
        try:
            # Fallback: convert to pandas and merge if join unsupported
            final_df_tmp = final_ds.to_pandas()
            ver_df_tmp = ver_docs_ds.to_pandas()
            final_df = final_df_tmp.merge(ver_df_tmp, on="article_id", how="left")
            final_ds = ray.data.from_pandas(final_df)
        except Exception:
            pass

# Peek first 10 results.
# NOTE: This is for local testing and debugging. For production use case,
# one should write full result out as shown below.
#outputs = final_ds.take(limit=1)

#for output in outputs:
#    print(f"Doc: {output.get('doc_id')}")
#    print(f"Predicted category number: {output.get('predicted_category_number')}")

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
#
hb5_stop, hb5_thread = _start_wandb_heartbeat("write_docs", dir_to_count=docs_out_dir)
final_ds.write_parquet(docs_out_dir)
_t_write = time.perf_counter()
try:
    if hb5_stop is not None:
        hb5_stop.set()
except Exception:
    pass

# Build pandas DataFrame for summaries and plots
final_df = final_ds.to_pandas()

# Sample chunk-level DataFrame for histograms
try:
    chunks_sample_df = ds.limit(20000).to_pandas()
except Exception:
    chunks_sample_df = None

# Derive country_code from article_path (segment after 'global_subset/')
def _derive_country(path: str) -> str:
    try:
        m = re.search(r"/global_subset/([^/]+)/", str(path))
        return m.group(1) if m else "unknown"
    except Exception:
        return "unknown"

if "article_path" in final_df.columns:
    final_df["country_code"] = final_df["article_path"].apply(_derive_country)
else:
    final_df["country_code"] = "unknown"

# Category mapping DataFrame (taxonomy order -> category_number)
taxonomy_list = pd.DataFrame(list(taxonomy.items()), columns=["category", "subcategories"]).explode("subcategories")
taxonomy_list["subcategory"] = taxonomy_list["subcategories"]
taxonomy_list = taxonomy_list.drop(columns=["subcategories"]) if "subcategories" in taxonomy_list.columns else taxonomy_list
taxonomy_list["category_number"] = range(1, len(taxonomy_list) + 1)

# Merge counts
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

# Per-country metrics
per_country = (
    final_df.groupby(["country_code", "predicted_category_number"]).size().reset_index(name="num_articles")
)

# Save summaries
article_counts_path = os.path.join(summaries_dir, "article_counts.csv")
per_country_path = os.path.join(summaries_dir, "per_country_metrics.csv")
article_counts.to_csv(article_counts_path, index=False)
per_country.to_csv(per_country_path, index=False)

# Per-country relevance rates (if relevance pass enabled)
per_country_rel_csv = None
try:
    if not ARGS.disable_relevance_pass:
        per_country_rel_csv = compute_per_country_relevance_rates(ARGS.input_parquet, final_df, summaries_dir)
except Exception:
    per_country_rel_csv = None

# Plot category bar chart
plot_df = article_counts.sort_values("estimated_relevant_articles", ascending=True)
num_bars = len(plot_df)
fig_height = min(max(num_bars * 0.35, 6), 30)
fig, ax = plt.subplots(figsize=(20, fig_height))
ax.barh(plot_df["subcategory"], plot_df["estimated_relevant_articles"], color="#4C78A8")
ax.set_xlabel("Number of Estimated Relevant Articles", fontsize=16)
ax.set_ylabel("Subcategory", fontsize=16)
ax.set_title("Number of Estimated Relevant Articles per Subcategory", fontsize=16)
ax.grid(axis='x', linestyle='--', alpha=0.3)
ax.invert_yaxis()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis='x', labelsize=14)
plt.subplots_adjust(left=0.4, right=0.98)
fig_path = os.path.join(figures_dir, "predicted_articles_per_subcategory.png")
plt.savefig(fig_path, pad_inches=0.25, bbox_inches='tight')
plt.close(fig)

# Plot histogram: tokens per chunk (sampled)
try:
    if chunks_sample_df is not None and "chunk_token_count" in chunks_sample_df.columns:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.hist(chunks_sample_df["chunk_token_count"], bins=50, color="#72B7B2")
        ax2.set_title("Token Count per Chunk (sampled)")
        ax2.set_xlabel("Tokens per chunk")
        ax2.set_ylabel("Frequency")
        fig2_path = os.path.join(figures_dir, "hist_tokens_per_chunk.png")
        plt.savefig(fig2_path, pad_inches=0.25, bbox_inches='tight')
        plt.close(fig2)
    else:
        fig2_path = None
except Exception:
    fig2_path = None

# Plot histogram: chunks per article
try:
    if "num_chunks" in final_df.columns:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.hist(final_df["num_chunks"], bins=50, color="#E45756")
        ax3.set_title("Chunks per Article")
        ax3.set_xlabel("Chunks")
        ax3.set_ylabel("Number of articles")
        fig3_path = os.path.join(figures_dir, "hist_chunks_per_article.png")
        plt.savefig(fig3_path, pad_inches=0.25, bbox_inches='tight')
        plt.close(fig3)
    else:
        fig3_path = None
except Exception:
    fig3_path = None

# Plot per-country article counts
try:
    per_country_total = final_df.groupby("country_code").size().reset_index(name="num_articles")
    per_country_total = per_country_total.sort_values("num_articles", ascending=True).tail(20)
    fig4, ax4 = plt.subplots(figsize=(16, 10))
    ax4.barh(per_country_total["country_code"], per_country_total["num_articles"], color="#4C78A8")
    ax4.set_xlabel("Articles")
    ax4.set_ylabel("Country")
    ax4.set_title("Articles per Country (Top 20)")
    ax4.grid(axis='x', linestyle='--', alpha=0.3)
    fig4_path = os.path.join(figures_dir, "articles_per_country_top20.png")
    plt.savefig(fig4_path, pad_inches=0.25, bbox_inches='tight')
    plt.close(fig4)
except Exception:
    fig4_path = None

# Per-country taxonomy distribution (stacked, top 20 countries x top 10 categories)
try:
    cat_counts_global = article_counts.sort_values("estimated_relevant_articles", ascending=False)
    top_categories = cat_counts_global["category_number"].head(10).tolist() if "category_number" in cat_counts_global.columns else []
    tmp = final_df.copy()
    tmp = tmp[tmp["predicted_category_number"] != "None"]
    tmp["predicted_category_number"] = tmp["predicted_category_number"].astype(int)
    if top_categories:
        tmp = tmp[tmp["predicted_category_number"].isin(top_categories)]
    top_countries = tmp["country_code"].value_counts().head(20).index.tolist()
    tmp = tmp[tmp["country_code"].isin(top_countries)]
    pivot = tmp.pivot_table(index="country_code", columns="predicted_category_number", values="article_id", aggfunc="count", fill_value=0)
    pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False)
    fig5, ax5 = plt.subplots(figsize=(18, 10))
    bottom = None
    for col in sorted(pivot.columns):
        vals = pivot[col].values
        if bottom is None:
            bars = ax5.bar(pivot.index, vals, label=str(col))
            bottom = vals
        else:
            ax5.bar(pivot.index, vals, bottom=bottom, label=str(col))
            bottom = bottom + vals
    ax5.set_title("Per-Country Taxonomy Distribution (Top 20 countries x Top 10 categories)")
    ax5.set_xlabel("Country")
    ax5.set_ylabel("Articles")
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig5.tight_layout()
    fig5_path = os.path.join(figures_dir, "per_country_taxonomy_stacked.png")
    plt.savefig(fig5_path, pad_inches=0.25, bbox_inches='tight')
    plt.close(fig5)
except Exception:
    fig5_path = None

# W&B logging (optional)
if ARGS.wandb_mode != "disabled" and wandb is not None and wandb.run is not None:
    # Scalars
    total_docs = int(len(final_df))
    none_frac = float((final_df["predicted_category_number"] == "None").mean()) if len(final_df) else 0.0
    # Counts at gates
    try:
        total_input_articles = int(ray.data.read_parquet(ARGS.input_parquet).count())
    except Exception:
        total_input_articles = None
    try:
        keyword_after = int(ds.count())  # after relevance gate this is not keyword stage; skip if unreliable
    except Exception:
        keyword_after = None
    wandb.log({
        "docs/total": total_docs,
        "docs/none_fraction": none_frac,
        **({"gate/input_articles": total_input_articles} if total_input_articles is not None else {}),
    })
    # Image
    try:
        wandb.log({"plots/predicted_articles_per_subcategory": wandb.Image(fig_path)})
    except Exception:
        pass
    if fig2_path:
        try:
            wandb.log({"plots/hist_tokens_per_chunk": wandb.Image(fig2_path)})
        except Exception:
            pass
    if fig3_path:
        try:
            wandb.log({"plots/hist_chunks_per_article": wandb.Image(fig3_path)})
        except Exception:
            pass
    if fig4_path:
        try:
            wandb.log({"plots/articles_per_country_top20": wandb.Image(fig4_path)})
        except Exception:
            pass
    if fig5_path:
        try:
            wandb.log({"plots/per_country_taxonomy_stacked": wandb.Image(fig5_path)})
        except Exception:
            pass
    # Per-country relevance rates plot (if available)
    try:
        if per_country_rel_csv:
            _df_rel = pd.read_csv(per_country_rel_csv)
            _df_rel_plot = _df_rel.sort_values("total_articles", ascending=True).tail(20)
            fig6, ax6 = plt.subplots(figsize=(16, 10))
            ax6.barh(_df_rel_plot["country_code"], _df_rel_plot["relevance_rate"], color="#72B7B2")
            ax6.set_xlabel("LLM Relevance Rate (YES / total)")
            ax6.set_ylabel("Country")
            ax6.set_title("Per-Country LLM Relevance Rate (Top 20 by volume)")
            fig6_path = os.path.join(figures_dir, "per_country_relevance_rate_top20.png")
            plt.savefig(fig6_path, pad_inches=0.25, bbox_inches='tight')
            plt.close(fig6)
            wandb.log({"plots/per_country_relevance_rate_top20": wandb.Image(fig6_path)})
    except Exception:
        pass

    # Verification metrics
    try:
        if "verified_doc" in final_df.columns:
            ver_frac = float(final_df["verified_doc"].fillna(False).mean()) if len(final_df) else 0.0
            wandb.log({
                "verify/docs_verified_fraction": ver_frac,
                "verify/method": ARGS.verify_method,
                "verify/top_k": int(ARGS.verify_top_k),
            })
            # Per-category verified fraction
            if "predicted_category_number" in final_df.columns:
                tmp = final_df.copy()
                vc = (
                    tmp.groupby("predicted_category_number")["verified_doc"].mean().reset_index(name="verified_fraction")
                )
                # Log table sample
                wandb.log({"tables/verify_per_category": wandb.Table(dataframe=vc)})
                # Plot: verified fraction per category (bar)
                try:
                    _plot_df = vc.copy()
                    _plot_df["predicted_category_number"] = _plot_df["predicted_category_number"].astype(str)
                    figv, axv = plt.subplots(figsize=(16, 6))
                    axv.bar(_plot_df["predicted_category_number"], _plot_df["verified_fraction"], color="#59A14F")
                    axv.set_xlabel("Category")
                    axv.set_ylabel("Verified Fraction")
                    axv.set_title("Verification Rate per Category")
                    axv.tick_params(axis='x', rotation=45)
                    _vf_path = os.path.join(figures_dir, "verify_fraction_per_category.png")
                    plt.savefig(_vf_path, pad_inches=0.25, bbox_inches='tight')
                    plt.close(figv)
                    wandb.log({"plots/verify_fraction_per_category": wandb.Image(_vf_path)})
                except Exception:
                    pass
            # Samples table
            try:
                cols = [c for c in [
                    "article_id", "article_path", "predicted_category_number", "verified_doc",
                    "ver_doc_best_ent", "ver_doc_best_evidence", "ver_doc_max_sim", "ver_doc_max_sim_evidence"
                ] if c in final_df.columns]
                if cols:
                    sample_df = final_df[cols].sample(n=min(200, len(final_df)), random_state=ARGS.seed) if len(final_df) > 0 else final_df[cols]
                    wandb.log({"tables/verify_docs_sample": wandb.Table(dataframe=sample_df)})
            except Exception:
                pass
            # Distribution plots for chunk-level metrics (sample)
            try:
                # Read a sample of chunks from written parquet if accessible
                chunks_df_small = None
                try:
                    chunks_df_small = ray.data.read_parquet(chunks_out_dir).limit(50000).to_pandas()
                except Exception:
                    pass
                if chunks_df_small is not None:
                    if "ver_sim_max" in chunks_df_small.columns:
                        fig_s, ax_s = plt.subplots(figsize=(12, 6))
                        ax_s.hist(chunks_df_small["ver_sim_max"].dropna().astype(float), bins=50, color="#4E79A7")
                        ax_s.set_title("Chunk-level Max Similarity (sample)")
                        ax_s.set_xlabel("cosine sim")
                        ax_s.set_ylabel("frequency")
                        _sim_hist_path = os.path.join(figures_dir, "hist_chunk_ver_sim_max.png")
                        plt.savefig(_sim_hist_path, pad_inches=0.25, bbox_inches='tight')
                        plt.close(fig_s)
                        wandb.log({"plots/hist_chunk_ver_sim_max": wandb.Image(_sim_hist_path)})
                    if "ver_nli_ent_max" in chunks_df_small.columns:
                        fig_n, ax_n = plt.subplots(figsize=(12, 6))
                        ax_n.hist(chunks_df_small["ver_nli_ent_max"].dropna().astype(float), bins=50, color="#F28E2B")
                        ax_n.set_title("Chunk-level NLI Entailment (sample)")
                        ax_n.set_xlabel("p(entailment)")
                        ax_n.set_ylabel("frequency")
                        _ent_hist_path = os.path.join(figures_dir, "hist_chunk_ver_nli_ent.png")
                        plt.savefig(_ent_hist_path, pad_inches=0.25, bbox_inches='tight')
                        plt.close(fig_n)
                        wandb.log({"plots/hist_chunk_ver_nli_ent": wandb.Image(_ent_hist_path)})
            except Exception:
                pass
    except Exception:
        pass

    # Stage timings & throughput
    try:
        timings = {
            "time/read_s": _t_read - _t0,
            "time/keyword_s": _t_keyword - _t_read,
            "time/chunk_s": _t_chunk - _t_keyword,
            "time/relevance_s": _t_rel - _t_chunk,
            "time/taxonomy_s": _t_tax - _t_rel,
            "time/aggregate_s": _t_agg - _t_tax,
            "time/write_s": _t_write - _t_agg,
            "time/total_s": _t_write - _t0,
        }
        # Throughput metrics
        throughputs = {}
        try:
            chunks_after_chunking = int(ray.data.read_parquet(chunks_out_dir).count())
            if timings["time/chunk_s"] > 0:
                throughputs["throughput/chunking_chunks_per_s"] = chunks_after_chunking / max(timings["time/chunk_s"], 1e-9)
        except Exception:
            pass
        try:
            if timings["time/taxonomy_s"] > 0:
                throughputs["throughput/taxonomy_chunks_per_s"] = chunks_after_chunking / max(timings["time/taxonomy_s"], 1e-9)
        except Exception:
            pass
        if len(final_df) > 0 and (timings["time/aggregate_s"] > 0):
            throughputs["throughput/aggregate_docs_per_s"] = len(final_df) / max(timings["time/aggregate_s"], 1e-9)
        wandb.log({**timings, **throughputs})
    except Exception:
        pass
    # Tables (sampled)
    try:
        sample_n = max(0, int(ARGS.wandb_sample_rows))
        if sample_n > 0 and total_docs > 0:
            cols = ["article_id", "article_path", "country_code", "predicted_category_number", "num_chunks", "chunk_labels"]
            cols = [c for c in cols if c in final_df.columns]
            sample_df = final_df[cols].sample(n=min(sample_n, total_docs), random_state=ARGS.seed)
            wandb.log({"tables/docs_sample": wandb.Table(dataframe=sample_df)})
    except Exception:
        pass
    # Artifacts (optional upload)
    if ARGS.upload_artifacts:
        try:
            art = wandb.Artifact(name=f"inferences_{RUN_ID}", type="dataset")
            art.add_dir(RUN_OUTPUT_DIR)
            wandb.log_artifact(art)
        except Exception:
            pass

# Cleanup temporary relevance directory if it exists
try:
    if '_relevance_tmp_dir' in globals() and os.path.isdir(_relevance_tmp_dir):
        shutil.rmtree(_relevance_tmp_dir, ignore_errors=True)
except Exception:
    pass

# Write run_summary.json with metrics and config
try:
    run_summary = {
        "run_id": RUN_ID,
        "input_parquet": ARGS.input_parquet,
        "taxonomy_json": ARGS.taxonomy_json,
        "model_source": ARGS.model_source,
        "num_docs": int(len(final_df)),
        "timings": {
            "read_s": _t_read - _t0,
            "keyword_s": _t_keyword - _t_read,
            "chunk_s": _t_chunk - _t_keyword,
            "relevance_s": _t_rel - _t_chunk,
            "taxonomy_s": _t_tax - _t_rel,
            "aggregate_s": _t_agg - _t_tax,
            "write_s": _t_write - _t_agg,
            "total_s": _t_write - _t0,
        },
        "config": {
            "tensor_parallel_size": ARGS.tensor_parallel_size,
            "batch_size": ARGS.batch_size,
            "max_model_len_tokens": ARGS.max_model_len_tokens,
            "max_output_tokens": ARGS.max_output_tokens,
            "safety_margin_tokens": ARGS.safety_margin_tokens,
            "chunk_overlap_tokens": ARGS.chunk_overlap_tokens,
            "disable_chunking": ARGS.disable_chunking,
            "seed": ARGS.seed,
            "prefilter_mode": ARGS.prefilter_mode,
            "disable_relevance_pass": ARGS.disable_relevance_pass,
            "relevance_prompt_id": ARGS.relevance_prompt_id,
            "taxonomy_prompt_id": ARGS.taxonomy_prompt_id,
            "verify_method": ARGS.verify_method,
            "verify_top_k": ARGS.verify_top_k,
            "verify_thresholds": ARGS.verify_thresholds,
        },
    }
    with open(os.path.join(RUN_OUTPUT_DIR, "summaries", "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)
except Exception:
    pass
