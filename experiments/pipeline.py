import os
import sys
import time
import shlex
import json
import subprocess
from datetime import datetime
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from experiments.schema import validate_and_resolve


def _bool_flag(flag: str, val: bool) -> list[str]:
    return [flag] if bool(val) else []


def _kv_arg(name: str, val: Any) -> list[str]:
    if val is None:
        return []
    return [name, str(val)]


def _wandb_args(cfg: DictConfig) -> list[str]:
    return [
        "--wandb_mode", str(cfg.wandb.mode),
        "--wandb_project", str(cfg.wandb.project),
        "--wandb_prefix", str(cfg.wandb.prefix),
        "--wandb_suffix", str(cfg.wandb.suffix),
    ]


def _resource_args(cfg: DictConfig, use_gpus: bool = True) -> list[str]:
    out = ["--num_cpus", str(cfg.resources.num_cpus)]
    if use_gpus:
        out += ["--num_gpus", str(cfg.resources.num_gpus)]
    else:
        out += ["--num_gpus", "0"]
    return out


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Validate and type the config against structured schema
    cfg = validate_and_resolve(cfg)
    # Resolve paths and run id using Hydra's output dir
    run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = str(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Materialize Hydra configs (prompts, taxonomy, keywords) to YAML for stage scripts
    config_dir = os.path.join(output_dir, run_id, "configs")
    os.makedirs(config_dir, exist_ok=True)
    prompts_path = os.path.join(config_dir, "prompts.yaml")
    taxonomy_path = os.path.join(config_dir, "taxonomy.yaml")
    keywords_path = os.path.join(config_dir, "keywords.yaml")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore
    if yaml is not None:
        with open(prompts_path, "w") as f:
            yaml.safe_dump(OmegaConf.to_container(cfg.prompts, resolve=True), f, sort_keys=False)
        with open(taxonomy_path, "w") as f:
            yaml.safe_dump(OmegaConf.to_container(cfg.taxonomy, resolve=True), f, sort_keys=False)
        with open(keywords_path, "w") as f:
            yaml.safe_dump(OmegaConf.to_container(cfg.keywords, resolve=True), f, sort_keys=False)
    else:
        # Fallback to JSON if PyYAML is not available
        prompts_path = os.path.join(config_dir, "prompts.json")
        taxonomy_path = os.path.join(config_dir, "taxonomy.json")
        keywords_path = os.path.join(config_dir, "keywords.json")
        with open(prompts_path, "w") as f:
            json.dump(OmegaConf.to_container(cfg.prompts, resolve=True), f, indent=2)
        with open(taxonomy_path, "w") as f:
            json.dump(OmegaConf.to_container(cfg.taxonomy, resolve=True), f, indent=2)
        with open(keywords_path, "w") as f:
            json.dump(OmegaConf.to_container(cfg.keywords, resolve=True), f, indent=2)

    # Stage 1: relevance
    rel_args: list[str] = [
        sys.executable, "-m", "experiments.relevance_stage",
        "--run_id", run_id,
        "--input_parquet", cfg.data.input_parquet,
        "--output_dir", output_dir,
        "--taxonomy_json", taxonomy_path,
        "--keywords_json", keywords_path,
        "--prompts_json", prompts_path,
        "--model_source", cfg.relevance.model_source,
        "--tensor_parallel_size", str(cfg.relevance.tensor_parallel_size),
        "--batch_size", str(cfg.relevance.batch_size),
        "--concurrency", str(cfg.relevance.concurrency),
        "--max_model_len_tokens", str(cfg.relevance.max_model_len_tokens),
        "--max_output_tokens", str(cfg.relevance.max_output_tokens),
        "--safety_margin_tokens", str(cfg.relevance.safety_margin_tokens),
        "--chunk_overlap_tokens", str(cfg.relevance.chunk_overlap_tokens),
        "--gpu_memory_utilization", str(cfg.relevance.gpu_memory_utilization),
        "--kv_cache_dtype", str(cfg.relevance.kv_cache_dtype),
        "--max_num_batched_tokens", str(cfg.relevance.max_num_batched_tokens),
        "--max_num_seqs", str(cfg.relevance.max_num_seqs),
        "--tokenizer_pool_size", str(cfg.relevance.tokenizer_pool_size) if cfg.relevance.get("tokenizer_pool_size") is not None else "",
        "--seed", str(cfg.seed),
        "--prefilter_mode", str(cfg.relevance.prefilter_mode),
    ]
    # remove possible empty args caused by optional fields
    rel_args = [a for a in rel_args if a != ""]
    rel_args += _resource_args(cfg, use_gpus=True)
    rel_args += _wandb_args(cfg)
    rel_args += _bool_flag("--conservative_vllm", cfg.relevance.conservative_vllm)
    rel_args += _bool_flag("--disable_keyword_prefilter", cfg.relevance.disable_keyword_prefilter)
    rel_args += _bool_flag("--disable_chunking", cfg.relevance.disable_chunking)
    rel_args += _bool_flag("--log_rationales", cfg.relevance.log_rationales)
    rel_args += _bool_flag("--debug", cfg.debug.enable)
    rel_args += _kv_arg("--debug_limit", cfg.debug.limit)

    if bool(cfg.get("pipeline", {}).get("in_process", False)):
        # In-process execution
        import importlib
        mod = importlib.import_module("experiments.relevance_stage")
        mod.ARGS = mod._build_arg_parser().parse_args(rel_args[3:])
        if bool(cfg.get("pipeline", {}).get("dry_run", False)):
            print("[pipeline] DRY RUN relevance args:", " ".join(map(shlex.quote, rel_args)))
        else:
            mod.main()
    else:
        print(f"[pipeline] Launching relevance stage: {' '.join(shlex.quote(a) for a in rel_args)}")
        subprocess.run(rel_args, check=True)

    # Stage 2: taxonomy
    rel_dir = os.path.join(output_dir, run_id, "relevance", "chunks")
    tax_args: list[str] = [
        sys.executable, "-m", "experiments.taxonomy_stage",
        "--run_id", run_id,
        "--relevance_dir", rel_dir,
        "--input_parquet", cfg.data.input_parquet,
        "--output_dir", output_dir,
        "--taxonomy_json", taxonomy_path,
        "--prompts_json", prompts_path,
        "--model_source", cfg.taxonomy.model_source,
        "--tensor_parallel_size", str(cfg.taxonomy.tensor_parallel_size),
        "--batch_size", str(cfg.taxonomy.batch_size),
        "--concurrency", str(cfg.taxonomy.concurrency),
        "--max_model_len_tokens", str(cfg.taxonomy.max_model_len_tokens),
        "--max_output_tokens", str(cfg.taxonomy.max_output_tokens),
        "--gpu_memory_utilization", str(cfg.taxonomy.gpu_memory_utilization),
        "--kv_cache_dtype", str(cfg.taxonomy.kv_cache_dtype),
        "--max_num_batched_tokens", str(cfg.taxonomy.max_num_batched_tokens),
        "--max_num_seqs", str(cfg.taxonomy.max_num_seqs),
        "--tokenizer_pool_size", str(cfg.taxonomy.tokenizer_pool_size) if cfg.taxonomy.get("tokenizer_pool_size") is not None else "",
        "--seed", str(cfg.seed),
        "--verify_method", str(cfg.verify.method),
        "--verify_top_k", str(cfg.verify.top_k),
        "--verify_thresholds", str(cfg.verify.thresholds),
        "--verify_device", str(cfg.verify.device),
    ]
    tax_args += _resource_args(cfg, use_gpus=True)
    tax_args += _wandb_args(cfg)
    tax_args += _bool_flag("--debug", cfg.debug.enable)
    tax_args += _kv_arg("--debug_limit", cfg.debug.limit)

    if bool(cfg.get("pipeline", {}).get("in_process", False)):
        import importlib
        mod = importlib.import_module("experiments.taxonomy_stage")
        mod.ARGS = mod._build_arg_parser().parse_args(tax_args[3:])
        if bool(cfg.get("pipeline", {}).get("dry_run", False)):
            print("[pipeline] DRY RUN taxonomy args:", " ".join(map(shlex.quote, tax_args)))
        else:
            mod.main()
    else:
        print(f"[pipeline] Launching taxonomy stage: {' '.join(shlex.quote(a) for a in tax_args)}")
        subprocess.run(tax_args, check=True)

    # Stage 3: verification (optional on/off)
    if str(cfg.verify.method).lower() != "off":
        ver_args: list[str] = [
            sys.executable, "-m", "experiments.verification_stage",
            "--run_id", run_id,
            "--output_dir", output_dir,
            "--taxonomy_json", taxonomy_path,
            "--chunks_dir", os.path.join(output_dir, run_id, "chunks"),
            "--docs_dir", os.path.join(output_dir, run_id, "docs"),
            "--verify_method", str(cfg.verify.method),
            "--verify_top_k", str(cfg.verify.top_k),
            "--verify_thresholds", str(cfg.verify.thresholds),
            "--verify_device", str(cfg.verify.device),
            "--verify_output", str(cfg.verify.output),
            "--seed", str(cfg.seed),
        ]
        ver_args += _resource_args(cfg, use_gpus=False)
        ver_args += _wandb_args(cfg)
        ver_args += _bool_flag("--debug", cfg.debug.enable)
        ver_args += _kv_arg("--debug_limit", cfg.debug.limit)
        if bool(cfg.get("pipeline", {}).get("in_process", False)):
            import importlib
            mod = importlib.import_module("experiments.verification_stage")
            mod.ARGS = mod._build_arg_parser().parse_args(ver_args[3:])
            if bool(cfg.get("pipeline", {}).get("dry_run", False)):
                print("[pipeline] DRY RUN verification args:", " ".join(map(shlex.quote, ver_args)))
            else:
                mod.main()
        else:
            print(f"[pipeline] Launching verification stage: {' '.join(shlex.quote(a) for a in ver_args)}")
            subprocess.run(ver_args, check=True)


if __name__ == "__main__":
    main()


