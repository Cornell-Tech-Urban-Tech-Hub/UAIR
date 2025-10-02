#!/usr/bin/env python3
"""Test script for compute metadata collection.

This script tests the compute metadata collection functionality without
requiring a full pipeline run or W&B authentication.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import pipelines module
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.uair.wandb_logger import collect_compute_metadata


def test_basic_collection():
    """Test basic metadata collection without config."""
    print("=" * 80)
    print("Testing basic compute metadata collection (no config)")
    print("=" * 80)
    
    metadata = collect_compute_metadata(cfg=None)
    
    print("\nCollected metadata:")
    print(json.dumps(metadata, indent=2, default=str))
    
    # Verify expected fields
    expected_fields = [
        "compute.cpu_count",
        "compute.cpu_architecture",
        "compute.gpu_count",
    ]
    
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    for field in expected_fields:
        if field in metadata:
            print(f"✓ {field}: {metadata[field]}")
        else:
            print(f"✗ {field}: MISSING")
    
    # GPU-specific checks
    if metadata.get("compute.gpu_count", 0) > 0:
        if "compute.gpu_type" in metadata:
            print(f"✓ compute.gpu_type: {metadata['compute.gpu_type']}")
        if "compute.gpus" in metadata:
            print(f"✓ compute.gpus: {len(metadata['compute.gpus'])} GPU(s) with details")
            for i, gpu in enumerate(metadata["compute.gpus"]):
                print(f"  - GPU {i}: {gpu.get('name')} ({gpu.get('total_memory_gb', 'N/A')} GB)")
    
    # Memory check
    if "compute.memory_gb" in metadata:
        print(f"✓ compute.memory_gb: {metadata['compute.memory_gb']} GB")
    
    # SLURM check
    if "slurm" in metadata:
        print(f"✓ slurm: {len(metadata['slurm'])} SLURM variables detected")
        for key, val in metadata["slurm"].items():
            print(f"  - {key}: {val}")
    else:
        print("ℹ slurm: Not running in SLURM environment")
    
    return metadata


def test_with_mock_config():
    """Test metadata collection with a mock Hydra config."""
    print("\n" + "=" * 80)
    print("Testing compute metadata collection (with mock config)")
    print("=" * 80)
    
    # Create a mock config object
    class MockEngineKwargs:
        max_model_len = 4096
        max_num_seqs = 4
        gpu_memory_utilization = 0.7
        tensor_parallel_size = 2
        enable_prefix_caching = True
    
    class MockModel:
        model_source = "/path/to/model"
        batch_size = 4
        concurrency = 1
        engine_kwargs = MockEngineKwargs()
    
    class MockRuntime:
        debug = True
        sample_n = 1000
        job_memory_gb = 64
        streaming_io = False
        use_llm_classify = True
    
    class MockConfig:
        model = MockModel()
        runtime = MockRuntime()
    
    metadata = collect_compute_metadata(cfg=MockConfig())
    
    print("\nCollected metadata (with config):")
    print(json.dumps(metadata, indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("Verification (config-specific fields)")
    print("=" * 80)
    
    # Check model config
    if "model" in metadata:
        print(f"✓ model.model_source: {metadata['model'].get('model_source')}")
        print(f"✓ model.batch_size: {metadata['model'].get('batch_size')}")
        if "engine_kwargs" in metadata["model"]:
            print(f"✓ model.engine_kwargs: {len(metadata['model']['engine_kwargs'])} parameters")
    else:
        print("✗ model: MISSING")
    
    # Check runtime config
    if "runtime" in metadata:
        print(f"✓ runtime.debug: {metadata['runtime'].get('debug')}")
        print(f"✓ runtime.sample_n: {metadata['runtime'].get('sample_n')}")
        print(f"✓ runtime: {len(metadata['runtime'])} parameters total")
    else:
        print("✗ runtime: MISSING")
    
    return metadata


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COMPUTE METADATA COLLECTION TEST")
    print("=" * 80)
    
    try:
        # Test 1: Basic collection
        metadata1 = test_basic_collection()
        
        # Test 2: With mock config
        metadata2 = test_with_mock_config()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"✓ Basic metadata collection: {len(metadata1)} top-level fields")
        print(f"✓ Config-enhanced collection: {len(metadata2)} top-level fields")
        print("\nAll tests completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


