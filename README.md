# Rumi — Elderly Fall Detection System (Backend & AI Engine)

Backend service dan engine AI deteksi jatuh untuk sistem monitoring CCTV **Rumi**, dibangun menggunakan **FastAPI**, **PostgreSQL**, **YOLO11-pose**, dan WebSocket real-time event pipeline.

---

## Tech Stack
- **API Framework:** FastAPI, SQLAlchemy, WebSockets
- **Database:** PostgreSQL
- **AI Engine:** Ultralytics YOLO11 (`yolo11n-pose.pt` / `yolo11s-pose.pt`)
- **Computer Vision:** OpenCV, PyTorch (Auto CPU / CUDA fallback)
- **Video Processing:** `imageio-ffmpeg` (Browser-compatible H.264/AVC1 encoding)

---

## Fitur Utama

1. **Multi-Camera Management (`/api/cameras`):**
   - Mendukung skema HTTP/MJPEG dan RTSP.
   - Normalisasi otomatis endpoint IP Webcam Android ke `/video`.
   - Pelaporan status aktif & telemetri koneksi (`ONLINE` / `OFFLINE`).

2. **Incident Notification Pipeline (`/api/incident-alerts`):**
   - Penyimpanan bukti snapshot (JPEG) dan klip video forensik (H.264 MP4).
   - Tanpa status penyelesaian (patuh PRD Rumi: purely monitoring & incident evidence).
   - Pelacakan status baca (`is_read`) untuk mengendalikan titik merah unread di tab Riwayat.

3. **Real-time WebSocket Broadcast (`/ws/events`):**
   - Mengirim event `INCIDENT_DETECTED`, `CAMERA_STATUS_CHANGED`, `INCIDENT_READ`, dan `INCIDENTS_ALL_READ` secara instan ke frontend TanStack Start.

4. **Multi-Stream Supervisor (`python_script/manager.py`):**
   - Threaded frame reader dengan auto-reconnect backoff.
   - Throttled inference (frame sampling di ~10 FPS, capture di 25–30 FPS) untuk efisiensi CPU/GPU.
   - Transcoding buffer berputar (rolling circular buffer) pre & post-fall.
   - Masking kredensial URL RTSP pada log terminal.

---

## Cara Menjalankan

### 1. Setup Virtual Environment & Dependencies
```bash
# Aktifkan virtual environment
.\venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
pip install imageio-ffmpeg
```

### 2. Jalankan FastAPI Backend
```bash
uvicorn main:app --reload --port 8000
```
*API docs otomatis dapat diakses di: `http://localhost:8000/docs`*
*WebSocket real-time channel: `ws://localhost:8000/ws/events`*

### 3. Menjalankan Engine AI

#### Opsi A: Multi-Stream Supervisor (Rekomendasi Produksi)
Otomatis membaca seluruh kamera aktif dari backend database dan memantau beberapa kamera secara bersamaan:
```bash
python python_script/manager.py
```

#### Opsi B: Standalone Single-Camera / Video Demo
Menjalankan deteksi langsung pada satu sumber kamera atau file video demo (`test_fall.mp4`):
```bash
python python_script/realtime_fall_detection.py
```
*(Opsional: set `DEBUG_GUI=true` di `.env` jika ingin melihat visualisasi OpenCV secara lokal)*
