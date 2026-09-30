#!/usr/bin/env python3
"""
Lightweight HTTP Server with full CORS and Range Request support for Rerun Web Viewer:
- Serves hot3d_nemo_benchmark.rrd with CORS headers so app.rerun.io can load it directly.
- Serves an interactive landing page at / with direct launch and download links.
"""

import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RerunHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for Rerun Web Viewer (app.rerun.io)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range, Content-Type, Accept')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Content-Length, Accept-Ranges')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()


def run_server(port=9090, directory=None):
    if directory:
        os.chdir(directory)
    print(f"Serving files from {os.getcwd()} on port {port} with CORS enabled...")
    httpd = HTTPServer(('0.0.0.0', port), RerunHTTPRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
    dir_path = sys.argv[2] if len(sys.argv) > 2 else "."
    run_server(port, dir_path)
