"""Serve only the local review packet and the three allowlisted PDFs on loopback."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit
import json

ROOT=Path(__file__).resolve().parent.parent
PDFS={Path(d['file']).name:ROOT/d['file'] for d in json.loads((ROOT/'docs/evaluation/public_corpus_v1.json').read_text())['documents']}


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        route=unquote(urlsplit(self.path).path)
        if route in {'/','/review.html'}:
            p=ROOT/'data/evaluation/answer-quality-v1/review.html';mime='text/html; charset=utf-8'
        elif route.startswith('/pdf/') and route[5:] in PDFS:
            p=PDFS[route[5:]];mime='application/pdf'
        else:
            self.send_error(404);return
        if not p.is_file():self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(p.stat().st_size))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        try:
            with p.open('rb') as stream:
                while data:=stream.read(65536):self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass


if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',8768),Handler)
    print('Local review: http://127.0.0.1:8768/review.html',flush=True)
    server.serve_forever()
