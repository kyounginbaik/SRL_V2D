"""Read-only local lab server with byte-range video streaming.

Only the generated site and two explicit dataset subfolders are served. There
are no shell-execution or job-launch endpoints. Bind defaults to loopback.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

WEB = Path(__file__).parent / 'web'
REPO = Path(__file__).resolve().parents[2]


def confined(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        return None
    return target


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, site, dataset, **kwargs):
        self.site, self.dataset = site, dataset
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        if len(args) > 1 and str(args[1]) not in ('200', '206', '304'):
            super().log_message(fmt, *args)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        path = unquote(urlsplit(self.path).path)
        if path == '/health':
            body = json.dumps({'ok': True, 'catalog': (self.site / 'catalog.json').exists()}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers()
            if not head:
                self.wfile.write(body)
            return
        if path.startswith('/data/'):
            target = confined(self.site, path[6:])
        elif path.startswith('/media/runs/'):
            target = confined(self.dataset / 'runs', path[len('/media/runs/'):])
        elif path.startswith('/media/track_3/public/'):
            target = confined(self.dataset / 'track_3/public', path[len('/media/track_3/public/'):])
        else:
            target = confined(WEB, 'index.html' if path == '/' else path.lstrip('/'))
        if target is None:
            self.send_error(404); return
        size = target.stat().st_size
        start, end, partial_content = 0, size - 1, False
        request_range = self.headers.get('Range')
        if request_range:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
            if not match or (not match[1] and not match[2]):
                self.send_error(416); return
            if match[1]:
                start = int(match[1]); end = min(end, int(match[2])) if match[2] else end
            else:
                start = max(0, size - int(match[2]))
            if start > end or start >= size:
                self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
            partial_content = True
        self.send_response(206 if partial_content else 200)
        content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
        if target.suffix == '.js':
            content_type = 'text/javascript'
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if partial_content:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:
            return
        try:
            with target.open('rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = f.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=REPO / '.ablation_lab/site')
    parser.add_argument('--dataset', type=Path, default=Path('/data1/video_to_data_challenge'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    handler = partial(Handler, site=args.site.resolve(), dataset=args.dataset.resolve())
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'V2D Ablation Lab: http://{args.host}:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
