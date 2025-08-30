import os
import shutil
import subprocess


def test_pipeline_dry_run(tmp_path):
    env = os.environ.copy()
    out_dir = tmp_path / "outputs"
    env["OUTPUT_DIR"] = str(out_dir)
    # Dry-run with in_process to avoid launching subprocesses
    cmd = [
        "python",
        "-m",
        "experiments.pipeline",
        "pipeline.in_process=true",
        "pipeline.dry_run=true",
        "debug=on",
    ]
    # Use a small input if DATA_ROOT is not set; rely on default interpolation
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True, env=env)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


