"""SGLang + small model on Modal (EU), OpenAI-compatible, scale-to-zero.

Deploy:  modal deploy inference/serve.py
Smoke:   modal run inference/serve.py        (needs MODAL_PROXY_TOKEN)
"""
import os
import subprocess
import time

import modal

APP_NAME = os.getenv("DAVID_MODAL_APP", "david-sglang")
SERVER_NAME = "SGLang"  # class name below; the API resolves the URL with this
MODEL = os.getenv("DAVID_MODEL", "Qwen/Qwen3-8B-AWQ")
MODEL_REVISION = os.getenv("DAVID_MODEL_REVISION")  # pin a commit hash for prod
GPU = os.getenv("DAVID_GPU", "L40S")  # Ada: native FP8 KV cache
PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.12.post1-cu130")
    .entrypoint([])
    .env({"HF_HUB_CACHE": "/root/.cache/huggingface", "HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.server(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache},
    compute_region="eu",           # EEA, 1.15x. "eu-west" = tighter, 1.75x
    routing_region="eu-west",      # Dublin proxy; fixed at first deploy
    min_containers=0,              # scale to zero
    scaledown_window=5 * MINUTES,
    startup_timeout=20 * MINUTES,  # first boot downloads weights
    target_concurrency=32,
    exit_grace_period=30,
    # Auth: Modal proxy token required (default). No unauthenticated=True.
)
class SGLang:
    @modal.enter()
    def start(self):
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", MODEL,
            "--served-model-name", MODEL,
            "--host", "0.0.0.0", "--port", str(PORT),
            "--context-length", "8192",
            "--kv-cache-dtype", "fp8_e4m3",
            "--tool-call-parser", "qwen25",
            "--reasoning-parser", "qwen3",
            "--mem-fraction-static", "0.85",
            "--cuda-graph-max-bs", "64",
            "--enable-metrics",
        ]
        if MODEL_REVISION:
            cmd += ["--revision", MODEL_REVISION]
        self.proc = subprocess.Popen(cmd)
        self._wait_ready()

    def _wait_ready(self, timeout=18 * MINUTES):
        import urllib.request

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"SGLang exited with code {self.proc.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                hf_cache.commit()
                return
            except Exception:
                time.sleep(3)
        raise TimeoutError("SGLang did not become healthy in time")

    @modal.exit()
    def stop(self):
        self.proc.terminate()


@app.local_entrypoint()
def smoke():
    from openai import OpenAI

    url = modal.Server.from_name(APP_NAME, SERVER_NAME).get_url()
    client = OpenAI(base_url=f"{url}/v1", api_key=os.environ["MODAL_PROXY_TOKEN"], timeout=600)
    print(f"→ {url}")
    t0 = time.time()
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Dis bonjour en une phrase."}],
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(r.choices[0].message.content)
    print(f"{time.time() - t0:.1f}s · {r.usage.prompt_tokens} in / {r.usage.completion_tokens} out")
