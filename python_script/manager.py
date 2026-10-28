import time
import os
import signal
import requests
import torch
from ultralytics import YOLO

from realtime_fall_detection import FallDetectionWorker, mask_url

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
CAMERAS_API_URL = f"{API_BASE_URL}/api/cameras"
SYNC_INTERVAL = int(os.getenv("CAMERA_SYNC_INTERVAL_SECONDS", "10"))
DEBUG_GUI = os.getenv("DEBUG_GUI", "false").lower() in ("true", "1")

INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "rumi_internal_worker_secret_2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_shared_yolo_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(BASE_DIR, "..", "yolo11n-pose.pt")
    if not os.path.exists(model_path):
        model_path = "yolo11n-pose.pt"
    print(f"[SUPERVISOR] Memuat shared model YOLO ({device}): {model_path}")
    return YOLO(model_path).to(device)


def main():
    print("==================================================")
    print("   Rumi Multi-Stream Fall Detection Supervisor   ")
    print("==================================================")
    print(f"Target API Backend: {API_BASE_URL}")
    print(f"Sinkronisasi kamera tiap: {SYNC_INTERVAL} detik")

    shared_model = load_shared_yolo_model()
    active_workers = {}
    is_running = True

    def handle_shutdown(signum, frame):
        nonlocal is_running
        print("\n[SUPERVISOR] Sinyal shutdown diterima. Menghentikan seluruh worker...")
        is_running = False

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while is_running:
        try:
            headers = {"X-Internal-Token": INTERNAL_SERVICE_TOKEN}
            resp = requests.get(f"{CAMERAS_API_URL}?is_active=true", headers=headers, timeout=5)
            if resp.status_code == 200:
                cameras = resp.json()
                current_camera_ids = {c["id"] for c in cameras}

                # 1. Hentikan worker untuk kamera yang dihapus/dinonaktifkan
                for cam_id in list(active_workers.keys()):
                    if cam_id not in current_camera_ids:
                        print(f"[SUPERVISOR] Kamera {cam_id} dihapus. Menghentikan worker...")
                        active_workers[cam_id].stop()
                        del active_workers[cam_id]

                # 2. Mulai worker untuk kamera baru yang terdaftar
                for cam in cameras:
                    cam_id = cam["id"]
                    if cam_id not in active_workers:
                        masked_url = mask_url(cam["url"])
                        print(f"[SUPERVISOR] Mendaftarkan kamera baru: '{cam['name']}' ({masked_url})")

                        worker = FallDetectionWorker(
                            camera_id=cam_id,
                            camera_name=cam["name"],
                            location=cam["location"],
                            stream_url=cam["url"],
                            model_instance=shared_model,
                            debug_gui=DEBUG_GUI
                        )
                        worker.start()
                        active_workers[cam_id] = worker
            else:
                print(f"[SUPERVISOR WARNING] Gagal mengambil daftar kamera (HTTP {resp.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"[SUPERVISOR WARNING] Menunggu koneksi backend di {API_BASE_URL}... ({e})")

        # Sleep dengan polling per 1 detik agar responsif terhadap sinyal shutdown
        for _ in range(SYNC_INTERVAL):
            if not is_running:
                break
            time.sleep(1)

    print("[SUPERVISOR] Mematikan semua worker aktif...")
    for cam_id, worker in active_workers.items():
        worker.stop()
    print("[SUPERVISOR] Seluruh worker telah berhenti. Bye!")


if __name__ == "__main__":
    main()
