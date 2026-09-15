#!/usr/bin/env python3
"""Vote on mixes; votes score the gene pool (see genes.py).

  .venv/bin/python web.py            # http://localhost:8085

One 👍/👎 per mix (in-memory, resets on restart). A vote credits the gene that
generated the mix; when every gene has been tried at least once, the pool
evolves automatically.
"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import genes

PORT = 8085  # 8080 is taken by llama-server on this machine
MIXES = Path("mixes")
VOTED = set()  # mix filenames already voted on this server session

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>headless-dj</title>
<style>
body {{ font-family: sans-serif; max-width: 760px; margin: 2em auto; background: #111; color: #ddd; }}
.row {{ display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #333; }}
.row div {{ flex: 1; min-width: 0; }} audio {{ width: 340px; }}
button {{ font-size: 1.3em; background: none; border: none; cursor: pointer; }}
button:hover {{ transform: scale(1.2); }} small {{ color: #888; }}
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
        msg = f"scored gene #{gene['id']}"
        self._send(200, json.dumps({"msg": msg}).encode(), "application/json")


if __name__ == "__main__":
    print(f"headless-dj votes: http://localhost:{PORT}")
    # 0.0.0.0: reachable from the LAN and tailscale; it's a trusted network
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
