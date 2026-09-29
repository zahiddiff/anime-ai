"""Minimal ComfyUI API client used by every AnimeAI script.

Usage from other scripts:
    from comfy_api import Comfy
    c = Comfy()
    outputs = c.run(workflow_dict)          # queues, waits, returns list of output file paths
"""
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "project" / "_comfy_output"
INPUT_DIR = ROOT / "comfyui" / "ComfyUI" / "input"


class Comfy:
    def __init__(self, host="127.0.0.1:8188"):
        self.base = f"http://{host}"
        self.client_id = uuid.uuid4().hex

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return json.loads(r.read())

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def alive(self):
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def stats(self):
        s = self._get("/system_stats")
        d = s["devices"][0]
        return {
            "vram_free_gb": round(d["vram_free"] / 1e9, 2),
            "vram_total_gb": round(d["vram_total"] / 1e9, 2),
            "ram_free_gb": round(s["system"]["ram_free"] / 1e9, 2),
        }

    def free_memory(self):
        """Ask ComfyUI to unload models (do this before handing the GPU to Ollama etc.)."""
        self._post("/free", {"unload_models": True, "free_memory": True})

    def queue(self, workflow):
        r = self._post("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if r.get("node_errors"):
            raise RuntimeError(f"node errors: {json.dumps(r['node_errors'], indent=1)}")
        return r["prompt_id"]

    def wait(self, prompt_id, poll=1.0, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self._get(f"/history/{prompt_id}")
            if prompt_id in h:
                entry = h[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise RuntimeError(f"execution error: {msgs}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll)
        raise TimeoutError(prompt_id)

    def run(self, workflow, timeout=3600):
        """Queue a workflow, wait for it, return (output paths, seconds)."""
        t0 = time.time()
        pid = self.queue(workflow)
        entry = self.wait(pid, timeout=timeout)
        paths = []
        for node_out in entry.get("outputs", {}).values():
            for key in ("images", "gifs", "videos"):
                for f in node_out.get(key, []):
                    if f.get("type") == "output":
                        paths.append(OUTPUT_DIR / f.get("subfolder", "") / f["filename"])
        return paths, round(time.time() - t0, 1)


def load_workflow(name):
    return json.loads((ROOT / "workflows" / f"{name}.api.json").read_text(encoding="utf-8"))
