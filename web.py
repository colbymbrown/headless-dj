#!/usr/bin/env python3
"""Vote on mixes; votes score the gene pool (see genes.py).

  .venv/bin/python web.py            # http://localhost:8085

One 👍/👎 per mix (in-memory, resets on restart). A vote credits the gene that
generated the mix; every 5 votes the pool evolves and the next mix kicks off.
"""
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import genes

PORT = 8085  # 8080 is taken by llama-server on this machine
MIXES = Path("mixes")
VOTED = set()  # mix filenames already voted on this server session

# Vote -> evolve -> next mix: each fresh vote kicks off a new mix in a
# background process (one GPU job at a time). Replaces the old daily cron.
ROOT = Path(__file__).resolve().parent
MIX_MINUTES = 30
_gen_lock = threading.Lock()


def kickoff_generation():
    """Spawn `dj.py --minutes MIX_MINUTES` in a background thread + process.
    Non-blocking: if a generation is already in flight, skip (one at a time)."""
    def run():
        if not _gen_lock.acquire(blocking=False):
            print("[vote] generation skipped, one already in flight", flush=True)
            return
        try:
            log = (ROOT / "mixes" / "vote-gen.log").open("a")
            print(f"[vote] starting new {MIX_MINUTES}-min mix", flush=True)
            proc = subprocess.Popen(
                [sys.executable, "dj.py", "--minutes", str(MIX_MINUTES)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            rc = proc.wait()
            log.write(f"vote-triggered generation exited {rc}\n")
            print(f"[vote] generation finished (exit {rc})", flush=True)
        finally:
            _gen_lock.release()
    threading.Thread(target=run, daemon=True).start()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>headless-dj</title>
<style>
body {{ font-family: sans-serif; max-width: 760px; margin: 2em auto; padding: 0 12px; box-sizing: border-box; background: #111; color: #ddd; }}
.row {{ display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #333; }}
.row div {{ flex: 1; min-width: 0; }} audio {{ width: 340px; }}
button {{ font-size: 1.6em; background: none; border: none; cursor: pointer; padding: 4px 10px; -webkit-tap-highlight-color: transparent; }}
button:hover {{ transform: scale(1.2); }} small {{ color: #888; word-wrap: break-word; }}
@media (max-width: 600px) {{
  body {{ margin: 1em auto; }}
  h1 {{ font-size: 1.4em; }}
  .row {{ flex-wrap: wrap; }}
  .row div {{ flex: 1 1 100%; }}
  audio {{ flex: 2 1 200px; width: auto; min-width: 200px; }}
  button {{ font-size: 2em; padding: 6px 14px; }}
}}
</style></head><body>
<h1>headless-dj</h1>
{rows}
<script>
function vote(mix, up) {{
  fetch('/vote', {{method:'POST', headers:{{'Content-Type':'application/json'}},
    body: JSON.stringify({{mix, up}})}})
    .then(r => r.json()).then(d =>
      document.getElementById('r-' + mix).textContent = d.msg);
}}
</script></body></html>"""


def index_html():
    rows = []
    for f in sorted(MIXES.glob("*.flac"), key=lambda p: p.stat().st_mtime,
                    reverse=True):
        gene = ""
        sidecar = f.with_suffix(".json")
        if sidecar.exists():
            g = json.loads(sidecar.read_text()).get("gene") or {}
            gene = f"gene #{g.get('id', '?')} ({g.get('plays', '?')} plays, " \
                   f"score {g.get('up', 0) - g.get('down', 0):+d}): {g.get('text', '')}"
        rows.append(
            f'<div class="row"><div><b>{f.name}</b><br><small>{gene}</small></div>'
            f'<audio controls preload="none" src="/mix/{f.name}"></audio>'
            f'<button onclick="vote(\'{f.name}\',1)">\U0001F44D</button>'
            f'<button onclick="vote(\'{f.name}\',0)">\U0001F44E</button>'
            f'<span id="r-{f.name}"></span></div>')
    return PAGE.format(rows="\n".join(rows))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mix_file(self, path):
        """Filename from /mix/<name>, traversal-safe."""
        name = path.removeprefix("/mix/")
        return MIXES / name if Path(name).name == name and name else None

    def do_GET(self):
        if self.path == "/":
            self._send(200, index_html().encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/mix/"):
            f = self._mix_file(self.path)
            if not f or not f.exists():
                return self._send(404, b"not found", "text/plain")
            self._send_flac(f)
        else:
            self._send(404, b"not found", "text/plain")

    def _send_flac(self, f):
        """Serve with Range support so the browser can seek in long mixes."""
        size = f.stat().st_size
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1) or 0)
            end = int(m.group(2)) if m.group(2) else size - 1
            start, end = min(start, end), min(end, size - 1)
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", "audio/flac")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):
        if self.path != "/vote":
            return self._send(404, b"not found", "text/plain")
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mix_name = data.get("mix", "")
        f = self._mix_file(f"/mix/{mix_name}")
        if not f or not f.exists():
            return self._send(400, json.dumps({"msg": "unknown mix"}).encode(),
                              "application/json")
        if mix_name in VOTED:
            return self._send(200, json.dumps({"msg": "already voted"}).encode(),
                              "application/json")
        gene = json.loads(f.with_suffix(".json").read_text()).get("gene")
        if not gene:
            return self._send(400,
                              json.dumps({"msg": "mix predates the gene pool; "
                                         "no gene to credit"}).encode(),
                              "application/json")
        genes.vote(genes.load(), gene["id"], bool(data.get("up")))
        VOTED.add(mix_name)
        kickoff_generation()  # evolve -> next: start the next 30-min mix now
        msg = f"scored gene #{gene['id']} · next mix generating"
        self._send(200, json.dumps({"msg": msg}).encode(), "application/json")


if __name__ == "__main__":
    print(f"headless-dj votes: http://localhost:{PORT}")
    # 0.0.0.0: reachable from the LAN and tailscale; it's a trusted network
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
