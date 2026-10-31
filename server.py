"""
Full-stack Dashboard Server for Speech Emotion Recognition.

Integrates infer.py directly with the interactive web dashboard:
- Serves the dashboard UI at http://127.0.0.1:8000/
- Multi-model switching:
    1. WavLM Base+ (Impro + Aug) [Champion 73.2% UA]
    2. Multimodal (WavLM + RoBERTa) [75.9% UA Audio+Text]
    3. WavLM Base+ (Full Dataset) [68.8% UA Audio-Only Ablation]

Usage:
  python server.py
  python run.py dashboard
"""
import io
import json
import os
import sys
import urllib.parse
import webbrowser
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from infer import EmotionPredictor

# Cache of loaded predictors (wavlm_impro, multimodal, wavlm)
PREDICTORS = {}
MODEL_LOCK = threading.Lock()


def get_predictor(model_name="wavlm_impro"):
    clean_name = model_name.lower().strip()
    if clean_name not in ["wavlm_impro", "multimodal", "wavlm"]:
        clean_name = "wavlm_impro"
    with MODEL_LOCK:
        if clean_name not in PREDICTORS:
            PREDICTORS[clean_name] = EmotionPredictor(model=clean_name)
        return PREDICTORS[clean_name]


class SERHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        dashboard_dir = BASE_DIR / "dashboard"
        super().__init__(*args, directory=str(dashboard_dir), **kwargs)

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.path = "/ser_dashboard.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                model_name = params.get("model", [self.headers.get("X-Model-Name", "wavlm_impro")])[0]
                text_input = params.get("text", [self.headers.get("X-Spoken-Text", "")])[0]

                content_len = int(self.headers.get("Content-Length", 0))
                if content_len == 0:
                    self._send_json({"status": "error", "error": "No audio data received"}, status=400)
                    return

                audio_bytes = self.rfile.read(content_len)

                # Get the appropriate predictor (cached in memory)
                predictor = get_predictor(model_name)
                with MODEL_LOCK:
                    result = predictor.predict(audio_bytes, text=text_input)
                self._send_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"status": "error", "error": str(e)}, status=500)
        else:
            self.send_error(404, "Endpoint not found")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Model-Name, X-Spoken-Text")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[VoxEmo Server] {args[0]} {args[1]}\n")


def find_free_port(start_port=8000, max_attempts=20):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    print("=" * 65)
    print("  VoxEmo — Speech Emotion Recognition Dashboard Server")
    print("=" * 65)

    # Pre-cache default champion model
    get_predictor("wavlm_impro")

    port = find_free_port(8000)
    server = ThreadingHTTPServer(("127.0.0.1", port), SERHandler)
    url = f"http://127.0.0.1:{port}/"

    print("\n" + "─" * 65)
    print(f"  🚀 Dashboard running at: {url}")
    print(f"  ⚡ Multi-Model Switcher: [wavlm_impro, multimodal, wavlm]")
    print(f"  🎙️ Real-time prediction ready on /api/predict")
    print("─" * 65)
    print("  Press Ctrl+C to stop the server\n")

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        server.server_close()


if __name__ == "__main__":
    main()
