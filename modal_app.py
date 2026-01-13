"""
Modal launcher for GPU kernel benchmarks.

Usage (local):
  pip install modal
  python -m modal setup
  modal run modal_app.py
"""
import os
import subprocess
import modal

APP_NAME = "ttt-kernel-bench"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "torch==2.3.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    )
    .pip_install("triton==2.3.0", "numpy")
)

repo_mount = modal.Mount.from_local_dir(
    os.path.dirname(__file__),
    remote_path="/root/TestTimeIdeas",
)


def _run(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return out.stdout + out.stderr


@app.function(
    gpu="H100",
    image=image,
    mounts=[repo_mount],
    timeout=60 * 30,
)
def run_bench():
    os.environ["PYTHONPATH"] = "/root/TestTimeIdeas"
    os.chdir("/root/TestTimeIdeas")
    out = []
    out.append(_run("python tests/bench.py --b 2 --h 4 --t 2048 --d 128 --dtype fp16 --iters 50"))
    out.append(_run("python tests/bench_gradgrad.py --b 2 --h 2 --t 128 --d 64 --dtype fp32"))
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(run_bench.remote())
