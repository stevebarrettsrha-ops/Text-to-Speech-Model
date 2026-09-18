"""A stand-in for huggingface.co: the tree API and resolve downloads, with
Range support, so download/resume/skip logic runs for real.

/mock/* switches let the harness cut a transfer off mid-file, ignore a Range
header, or gate a repo behind a token.
"""
import os, random, sys
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

REPOS = {
    "Qwen/Qwen3-TTS-Tokenizer-12Hz": [
        ("config.json", 900),
        ("model.safetensors", 400_000),
        ("README.md", 4_000),
        (".gitattributes", 1_500),
    ],
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": [
        ("config.json", 1_200),
        ("model.safetensors", 900_000),
        ("pytorch_model.bin", 900_000),     # duplicate of the safetensors
        ("tokenizer.json", 20_000),
        ("README.md", 6_000),
    ],
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": [
        ("config.json", 1_100),
        ("model.safetensors", 850_000),
    ],
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": [("config.json", 1_100),
                                      ("model.safetensors", 1_500_000)],
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": [("config.json", 1_100),
                                             ("model.safetensors", 1_500_000)],
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": [("config.json", 1_100),
                                             ("model.safetensors", 1_600_000)],
}

MODE = {"cut_after": 0, "ignore_range": False, "gated": "", "slow": 0.0}
LOG = []


@app.post("/mock/mode")
def set_mode():
    MODE.update(request.get_json(silent=True) or {})
    return jsonify(MODE)


@app.get("/mock/log")
def get_log():
    return jsonify(LOG)


@app.get("/api/models/<path:repo>/tree/<rev>")
def tree(repo, rev):
    LOG.append(f"tree {repo}")
    if repo == MODE.get("gated") and not request.headers.get("Authorization"):
        return jsonify({"error": "gated"}), 401
    if repo not in REPOS:
        return jsonify({"error": "Repo not found"}), 404
    return jsonify([{"type": "file", "path": p, "size": s,
                     "oid": "0" * 40} for p, s in REPOS[repo]])


@app.get("/api/datasets/<path:repo>/tree/<rev>")
def tree_ds(repo, rev):
    return jsonify({"error": "Repo not found"}), 404


def body_for(path, size):
    seed = sum(path.encode()) or 1
    rnd = random.Random(seed)
    return bytes(rnd.randrange(256) for _ in range(size))


@app.get("/<path:repo>/resolve/<rev>/<path:fname>")
def resolve(repo, rev, fname):
    if repo not in REPOS:
        return "no repo", 404
    entry = next((e for e in REPOS[repo] if e[0] == fname), None)
    if not entry:
        return "no file", 404
    data = body_for(fname, entry[1])
    rng = request.headers.get("Range", "")
    start = 0
    status = 200
    headers = {"Accept-Ranges": "bytes", "Content-Type": "application/octet-stream"}
    if rng.startswith("bytes=") and not MODE["ignore_range"]:
        start = int(rng.split("=", 1)[1].split("-")[0])
        if start >= len(data):
            LOG.append(f"416 {fname}")
            return Response(status=416)
        status = 206
        headers["Content-Range"] = f"bytes {start}-{len(data)-1}/{len(data)}"
    chunk = data[start:]
    cut = MODE["cut_after"]
    truncated = False
    if cut and len(chunk) > cut:
        chunk = chunk[:cut]
        truncated = True
    if MODE.get("slow"):
        import time as _t; _t.sleep(float(MODE["slow"]))
    headers["Content-Length"] = str(len(chunk))
    LOG.append(f"get {fname} start={start} status={status} "
               f"sent={len(chunk)}{' CUT' if truncated else ''}")
    return Response(chunk, status=status, headers=headers)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("MOCK_HF_PORT", "8199")),
            threaded=True)
