#!/usr/bin/env python3
"""
Run: python server.py
Opens the Bamberg Crowd Prediction Visualiser on http://localhost:5050
"""
import http.server
import socketserver
import webbrowser
import os

PORT = 5050
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence request logs

print(f"Visualiser running at http://localhost:{PORT}")
print("Press Ctrl+C to stop.\n")
webbrowser.open(f"http://localhost:{PORT}")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()