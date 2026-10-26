"""
Servidor local del modelo Fine-Tuning (Ruta B). Corre en WSL, donde está la
GPU, y deja el modelo cargado en memoria; la interfaz de validación (que corre
en Windows) le envía las consultas por HTTP.

Uso (en WSL):
    export HSA_ENABLE_DXG_DETECTION=1
    ~/train-env/bin/python src/fase3_ruta_finetuning/servidor_ft.py

Endpoints:
    GET  /salud       -> {"ok": true}
    POST /responder   -> {"consulta": "...", "subset": "A"|"B"}  ->  {"respuesta": "..."}
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from fase3_ruta_finetuning.inference_ft import _responder_local

PUERTO = 8765
_gpu = threading.Lock()  # una generación a la vez


class Manejador(BaseHTTPRequestHandler):
    def _json(self, codigo: int, cuerpo: dict):
        datos = json.dumps(cuerpo, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def do_GET(self):
        self._json(200, {"ok": True}) if self.path == "/salud" else self._json(404, {"error": "no existe"})

    def do_POST(self):
        if self.path != "/responder":
            return self._json(404, {"error": "no existe"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            pedido = json.loads(self.rfile.read(n).decode("utf-8"))
            with _gpu:
                respuesta = _responder_local(pedido["consulta"], subset=pedido.get("subset", "A"))
            self._json(200, {"respuesta": respuesta})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, *args):  # sin ruido en consola
        pass


if __name__ == "__main__":
    print("Cargando el modelo Fine-Tuning en la GPU ...", flush=True)
    _responder_local("Prueba de arranque", subset="A")
    print(f"Listo: escuchando en http://0.0.0.0:{PUERTO}", flush=True)
    HTTPServer(("0.0.0.0", PUERTO), Manejador).serve_forever()
