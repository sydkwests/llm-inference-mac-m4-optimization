import subprocess
import time
from typing import Dict

class InferenceCLIWrapper:
    def __init__(self, model_id: str):
        # e.g. "mlx-community/Phi-3-mini-4k-instruct"
        self.model_id = model_id

    def generate_with_metrics(self, prompt: str, max_tokens: int = 100) -> Dict:
        """Run mlx_lm.generate via subprocess and measure latency/throughput."""

        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", self.model_id,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", "0.7",
        ]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end = time.time()
        latency = end - start

        if result.returncode != 0:
            # Return stderr so we can debug
            return {
                "ok": False,
                "latency_seconds": latency,
                "tokens_generated": 0,
                "tokens_per_sec": 0.0,
                "output": "",
                "stderr": result.stderr.strip(),
                "stdout": result.stdout.strip(),
                "cmd": " ".join(cmd),
            }

        output = result.stdout.strip()
        tokens_generated = len(output.split())
        tps = tokens_generated / latency if latency > 0 else 0.0

        return {
            "ok": True,
            "latency_seconds": latency,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tps,
            "output": output[:400],
            "stderr": result.stderr.strip(),
            "cmd": " ".join(cmd),
        }
