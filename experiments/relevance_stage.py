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
from experiments.utils.vllm_config import build_engine_kwargs
from experiments.utils.common import load_structured, start_wandb_heartbeat

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Relevance stage: Ray Data + vLLM")
    # IO
    p.add_argument("--input_parquet", type=str, required=True)
    p.add_argument("--taxonomy_json", type=str, required=True)
    p.add_argument("--keywords_json", type=str, required=True)
    p.add_argument("--prompts_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_id", type=str, default=None)
    # Model
    p.add_argument("--model_source", type=str, required=True)
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max_model_len_tokens", type=int, default=8192)
    p.add_argument("--max_output_tokens", type=int, default=8)
    p.add_argument("--safety_margin_tokens", type=int, default=2048)
    p.add_argument("--chunk_overlap_tokens", type=int, default=512)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    p.add_argument("--conservative_vllm", action="store_true")
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
    # Prefilter & chunk
    p.add_argument("--disable_keyword_prefilter", action="store_true")
    p.add_argument("--prefilter_mode", choices=["pre_gating", "post_gating", "off"], default="pre_gating")
    p.add_argument("--disable_chunking", action="store_true")
    p.add_argument("--log_rationales", action="store_true")
    # vLLM tokenizer pool
    p.add_argument("--tokenizer_pool_size", type=int, default=None)
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


def _start_wandb_heartbeat(stage_name: str, interval_s: float = 15.0, dir_to_count: Optional[str] = None):
    if not (wandb is not None and ARGS.wandb_mode != "disabled" and getattr(wandb, "run", None) is not None):
        return None, None
    import threading, time
    stop_event = threading.Event()
    start_ts = time.time()

    def _poll():
        while not stop_event.is_set():
            payload: Dict[str, Any] = {
                "heartbeat/stage": stage_name,
                "heartbeat/elapsed_s": time.time() - start_ts,
            }
            try:
                avail = ray.available_resources()
                total = ray.cluster_resources()
                payload["resources/available/CPU"] = float(avail.get("CPU", 0.0))
                payload["resources/available/GPU"] = float(avail.get("GPU", 0.0))
                payload["resources/total/CPU"] = float(total.get("CPU", 0.0))
                payload["resources/total/GPU"] = float(total.get("GPU", 0.0))
            except Exception:
                pass
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


def _get_tokenizer(model_source: str):
    from transformers import AutoTokenizer  # type: ignore
    return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)


def _get_max_user_input_tokens(tokenizer, system_prompt: str) -> int:
    system_tokens = len(tokenizer.encode(system_prompt, add_special_tokens=False))
    return max(1024, ARGS.max_model_len_tokens - ARGS.max_output_tokens - system_tokens - ARGS.safety_margin_tokens)


def _load_structured(path: str):
    return load_structured(path)


def _to_blocks(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [x for x in raw_value if isinstance(x, str)]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, str)]
        except Exception:
            return [raw_value]
    return []


def _ensure_article_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    import hashlib as _hash
    article_id_val = row.get("article_id")
    if not (isinstance(article_id_val, str) and article_id_val):
        article_path_val = row.get("article_path", "")
        if isinstance(article_path_val, str) and article_path_val:
            article_id_val = _hash.sha1(article_path_val.encode("utf-8")).hexdigest()
        else:
            blocks = _to_blocks(row.get("relevant_blocks", []))
            full_text = "\n\n".join(blocks)
            article_id_val = _hash.sha1(full_text.encode("utf-8")).hexdigest()
    out_article_path = row.get("article_path", None)
    return {**row, "article_id": article_id_val, "article_path": out_article_path}


def _derive_country_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    path = row.get("article_path")
    try:
        m = re.search(r"/global_subset/([^/]+)/", str(path))
        row["country_code"] = m.group(1) if m else "unknown"
    except Exception:
        row["country_code"] = "unknown"
    return row


def _chunk_text(text: str, tokenizer, system_prompt: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return [text or ""]
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    max_user_tokens = _get_max_user_input_tokens(tokenizer, system_prompt)
    if len(input_ids) <= max_user_tokens:
        return [text]
    chunks: List[str] = []
    step = max(1, max_user_tokens - ARGS.chunk_overlap_tokens)
    for start in range(0, len(input_ids), step):
        end = min(start + max_user_tokens, len(input_ids))
        piece_ids = input_ids[start:end]
        piece_text = tokenizer.decode(piece_ids, skip_special_tokens=True)
        if piece_text:
            chunks.append(piece_text)
        if end == len(input_ids):
            break
    return chunks


def _explode_into_chunks(row: Dict[str, Any], tokenizer, system_prompt: str) -> List[Dict[str, Any]]:
    blocks = _to_blocks(row.get("relevant_blocks", []))
    text = ("\n\n".join(blocks)) if blocks else str(row.get("article_text", ""))
    chunks = _chunk_text(text, tokenizer, system_prompt)
    out: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        out.append({**row, "chunk_id": idx, "num_chunks": len(chunks), "chunk_text": chunk})
    return out


def _trim_chunk_for_prompt(row: Dict[str, Any], tokenizer, system_prompt: str) -> Dict[str, Any]:
    text = row.get("chunk_text", "")
    if not isinstance(text, str):
        return row
    max_user_tokens = _get_max_user_input_tokens(tokenizer, system_prompt)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_user_tokens:
        ids = ids[:max_user_tokens]
        row["chunk_text"] = tokenizer.decode(ids, skip_special_tokens=True)
    return row


def _create_vllm_config(model_source: str, enable_chunked: bool = True) -> vLLMEngineProcessorConfig:
    def _mk(gpu_mem: float, chunked: bool, max_tok: int, max_seqs: int, eager: bool) -> vLLMEngineProcessorConfig:
        ekw = build_engine_kwargs(
            max_model_len=ARGS.max_model_len_tokens,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=ARGS.tensor_parallel_size,
            max_num_batched_tokens=max_tok,
            max_num_seqs=max_seqs,
            kv_cache_dtype=(None if ARGS.kv_cache_dtype == "auto" else ARGS.kv_cache_dtype),
            enable_chunked_prefill=chunked,
            enforce_eager=eager,
            tokenizer_pool_size=ARGS.tokenizer_pool_size,
        )
        return vLLMEngineProcessorConfig(
            model_source=model_source,
            engine_kwargs=ekw,
            concurrency=ARGS.concurrency,
            batch_size=ARGS.batch_size,
        )

    attempts = (
        [(0.5, False, 2048, 2, True)] if ARGS.conservative_vllm else [
            (min(ARGS.gpu_memory_utilization, 0.85), True, ARGS.max_num_batched_tokens, ARGS.max_num_seqs, False),
            (0.65, False, max(1024, int(ARGS.max_num_batched_tokens * 0.75)), max(1, int(ARGS.max_num_seqs * 0.5)), True),
            (0.50, False, max(1024, int(ARGS.max_num_batched_tokens * 0.66)), max(1, int(ARGS.max_num_seqs * 0.5)), True),
            (0.30, False, 1536, 1, True),
        ]
    )
    last_err: Optional[Exception] = None
    for gpu_mem, chunked, mtok, mseq, eager in attempts:
        try:
            cfg = _mk(gpu_mem, chunked, int(mtok), int(mseq), eager)
            # Create a processor to validate args (won't init engine yet)
            _ = build_llm_processor(cfg, preprocess=lambda row: {}, postprocess=lambda row: {})
            return cfg
        except Exception as e:
            last_err = e
            logging.warning(f"vLLM cfg attempt failed (gpu_mem={gpu_mem}, chunked={chunked}): {e}")
            continue
    raise RuntimeError(f"Failed to create vLLM config: {last_err}")


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
    rel_dir = os.path.join(run_dir, "relevance", "chunks")
    Path(rel_dir).mkdir(parents=True, exist_ok=True)

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
                "stage": "relevance",
                "input_parquet": ARGS.input_parquet,
                "model_source": ARGS.model_source,
                "tensor_parallel_size": ARGS.tensor_parallel_size,
                "batch_size": ARGS.batch_size,
                "max_model_len_tokens": ARGS.max_model_len_tokens,
            },
        )

    t0 = time.perf_counter()
    ds = ray.data.read_parquet(ARGS.input_parquet, ray_remote_args={"num_cpus": 0.25})
    if ARGS.debug:
        ds = ds.limit(int(ARGS.debug_limit))

    # Taxonomy, prompts, tokenizer & system text
    taxonomy = _load_structured(ARGS.taxonomy_json)["taxonomy"]
    try:
        prompts = _load_structured(ARGS.prompts_json)
    except Exception:
        prompts = {}
    taxonomy_str = "\n".join([f"{'\n'.join(subcats)}" for _, subcats in taxonomy.items()])
    taxonomy_str = "\n".join([f"{i+1}. {line}" for i, line in enumerate(taxonomy_str.split("\n"))])
    system_prompt = prompts.get("relevance_v1", {}).get("system", "You are a careful news analyst trained to decide if a news article is related to artificial intelligence (AI) in any substantive way.")
    user_tmpl = prompts.get("relevance_v1", {}).get(
        "user_template",
        "Read the following article chunk and answer strictly YES or NO to the question: Is this article related to artificial intelligence?\n\n[article_id={article_id} chunk={chunk_id}/{num_chunks}]\n{chunk_text}",
    )

    tok = _get_tokenizer(ARGS.model_source)

    # Map basic fields and chunking
    ds = ds.map(_ensure_article_keys)
    ds = ds.map(_derive_country_from_row)

    # Keyword prefilter
    try:
        input_fields = [f.name for f in ds.schema().fields]
    except Exception:
        input_fields = []
    if (not ARGS.disable_keyword_prefilter) and ("article_text" in input_fields):
        try:
            kw = _load_structured(ARGS.keywords_json)
            method_terms = [str(t).lower() for t in kw.get("methods", [])]
            domain_terms = [str(t).lower() for t in kw.get("domains", [])]
        except Exception:
            method_terms, domain_terms = [], []
        if method_terms or domain_terms:
            def _keyword_map(row: Dict[str, Any]) -> Dict[str, Any]:
                text = str(row.get("article_text", "")).lower()
                row["relevant_keyword"] = bool(any(t in text for t in method_terms) and any(t in text for t in domain_terms))
                return row
            ds = ds.map(_keyword_map)
            if ARGS.prefilter_mode == "pre_gating":
                ds = ds.filter(lambda r: bool(r.get("relevant_keyword", True)))

    # Chunking
    if ARGS.disable_chunking:
        def _mk_single_chunk(row: Dict[str, Any]) -> Dict[str, Any]:
            blocks = _to_blocks(row.get("relevant_blocks", []))
            text_val = ("\n\n".join(blocks)) if blocks else str(row.get("article_text", ""))
            return {**row, "chunk_id": 0, "num_chunks": 1, "chunk_text": text_val}
        ds = ds.map(_mk_single_chunk)
    else:
        ds = ds.flat_map(lambda r: _explode_into_chunks(r, tok, system_prompt))
    ds = ds.map(lambda r: _trim_chunk_for_prompt(r, tok, system_prompt))

    # Build vLLM config and processor
    cfg = _create_vllm_config(ARGS.model_source, enable_chunked=True)
    relevance_processor = build_llm_processor(
        cfg,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_tmpl.format(
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

    hb_stop, _ = start_wandb_heartbeat(wandb, ray, "relevance_stage", dir_to_count=rel_dir)
    ds_rel = relevance_processor(ds)
    # Persist and reload to force actor teardown
    try:
        ds_rel.write_parquet(rel_dir)
        _ = ray.data.read_parquet(rel_dir).count()
    except Exception:
        _ = ds_rel.materialize()
    finally:
        try:
            if hb_stop is not None:
                hb_stop.set()
        except Exception:
            pass

    t1 = time.perf_counter()
    # Simple summary
    try:
        with open(os.path.join(run_dir, "summaries"), "a"):
            pass
        summary_dir = os.path.join(run_dir, "summaries")
        Path(summary_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(summary_dir, "relevance_summary.json"), "w") as f:
            json.dump({
                "run_id": run_id,
                "input_parquet": ARGS.input_parquet,
                "relevance_chunks_dir": rel_dir,
                "timings": {"total_s": t1 - t0},
            }, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()


