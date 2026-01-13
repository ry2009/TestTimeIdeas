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
    .apt_install("build-essential")
    .pip_install(
        "torch==2.3.1",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("numpy")
    .add_local_dir(
        os.path.dirname(__file__),
        remote_path="/root/TestTimeIdeas",
        ignore=[".git", "__pycache__", ".mypy_cache", ".pytest_cache"],
    )
)


def _run(cmd: str) -> str:
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return out.stdout + out.stderr


@app.function(
    gpu="H100",
    image=image,
    timeout=60 * 30,
)
def run_bench():
    os.environ["PYTHONPATH"] = "/root/TestTimeIdeas"
    os.chdir("/root/TestTimeIdeas")
    out = []
    out.append(_run("python tests/bench.py --b 2 --h 4 --t 2048 --d 128 --dtype fp16 --iters 50"))
    # grad-grad sweeps (small -> larger)
    out.append(_run("python tests/bench_gradgrad.py --b 2 --h 2 --t 128 --d 64 --dtype fp16 --bwd_mode recompute"))
    out.append(_run("python tests/bench_gradgrad.py --b 2 --h 2 --t 128 --d 64 --dtype fp16 --bwd_mode recompute_manual"))
    out.append(_run("python tests/bench_gradgrad.py --b 2 --h 2 --t 128 --d 64 --dtype fp16 --bwd_mode recompute_sdp"))
    out.append(_run("python tests/bench_gradgrad.py --b 2 --h 2 --t 128 --d 64 --dtype fp16 --bwd_mode save_p"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 512 --d 128 --dtype fp16 --bwd_mode recompute"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 512 --d 128 --dtype fp16 --bwd_mode recompute_manual"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 512 --d 128 --dtype fp16 --bwd_mode recompute_sdp"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 512 --d 128 --dtype fp16 --bwd_mode save_p"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 1024 --d 128 --dtype fp16 --bwd_mode recompute"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 1024 --d 128 --dtype fp16 --bwd_mode recompute_manual"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 1024 --d 128 --dtype fp16 --bwd_mode recompute_sdp"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 4 --t 1024 --d 128 --dtype fp16 --bwd_mode save_p"))
    # long-context sanity (reduce iters to keep runtime reasonable)
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 1 --t 8192 --d 64 --dtype fp16 --iters 5 --warmup 2 --bwd_mode recompute"))
    out.append(_run("python tests/bench_gradgrad.py --b 1 --h 1 --t 8192 --d 64 --dtype fp16 --iters 5 --warmup 2 --bwd_mode save_p"))
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(run_bench.remote())
