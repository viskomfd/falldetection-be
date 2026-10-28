import os
import re
import time
import subprocess
import urllib.parse
from datetime import datetime
from typing import List, Optional
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Form,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Header
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import desc
import cv2

import models, schemas, database

# Buat tabel otomatis jika belum ada
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Rumi Fall Detection API", version="2.0.0")

# Setup CORS agar TanStack Start / Next.js / Vite bisa akses (strictly whitelist trusted local origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pastikan folder penyimpanan media tersedia
os.makedirs("storage/thumbnails", exist_ok=True)
os.makedirs("storage/clips", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")


# --- WEBSOCKET REAL-TIME BROADCASTER ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Tetap jaga koneksi aktif
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- HELPER UTILITIES & SECURITY CONTROLS ---

INTERNAL_SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "rumi_internal_worker_secret_2026")

def mask_camera_url(url: str) -> str:
    """Masks username and password in rtsp://user:password@host -> rtsp://user:***@host"""
    return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1***\3', str(url))

def safe_delete_storage_file(relative_or_url_path: Optional[str]) -> bool:
    """
    Safely deletes a file only if it resides strictly inside the storage directory,
    preventing path traversal attacks.
    """
    if not relative_or_url_path:
        return False

    clean_path = relative_or_url_path.lstrip("/").replace("\\", "/")
    if clean_path.startswith("storage/"):
        rel_target = clean_path
    else:
        rel_target = os.path.join("storage", clean_path)

    base_storage_dir = os.path.abspath("storage")
    target_abs_path = os.path.abspath(rel_target)

    # Path traversal guard
    if not target_abs_path.startswith(base_storage_dir):
        return False

    if os.path.exists(target_abs_path) and os.path.isfile(target_abs_path):
        try:
            os.remove(target_abs_path)
            return True
        except OSError:
            return False
    return False

def normalize_camera_url(raw_url: str, protocol_hint: Optional[str] = None) -> tuple[str, str]:
    """
    Normalisasi URL kamera sesuai PRD §8.5:
    - URL dengan skema rtsp:// diberi protocol RTSP.
    - URL HTTP tanpa path (misal http://192.168.1.10:8080) otomatis ditambah /video untuk IP Webcam.
    """
    url = raw_url.strip()
    if url.lower().startswith("rtsp://"):
        return url, "RTSP"

    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        url = f"http://{url}"

    parsed = urllib.parse.urlparse(url)
    if parsed.path in ("", "/"):
        normalized = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            "/video",
            parsed.params,
            parsed.query,
            parsed.fragment
        ))
        return normalized, "HTTP"

    protocol = protocol_hint or ("HTTPS" if parsed.scheme == "https" else "HTTP")
    return url, protocol


# --- LIVE MJPEG STREAM FOR DEMO VIDEO ---

@app.get("/api/stream/video")
def stream_test_video():
    """
    Menyediakan MJPEG stream dari test_fall.mp4 secara looping
    agar elemen <img src="/api/stream/video" /> di dashboard dapat menampilkan
    video bergerak secara langsung.
    """
    def iter_frames():
        video_path = os.path.join("python_script", "test_fall.mp4")
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.04)
                continue

            success, jpeg = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
            time.sleep(0.033)

    return StreamingResponse(
        iter_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- ROOT & HEALTH ---

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Rumi Fall Detection API",
        "version": "2.0.0"
    }


# --- CAMERA ENDPOINTS ---

@app.get("/api/cameras", response_model=List[schemas.CameraResponse])
def get_cameras(
    is_active: Optional[bool] = None,
    x_internal_token: Optional[str] = Header(None),
    db: Session = Depends(database.get_db)
):
    query = db.query(models.Camera)
    if is_active is not None:
        query = query.filter(models.Camera.is_active == is_active)
    cams = query.order_by(models.Camera.created_at.desc()).all()

    if x_internal_token == INTERNAL_SERVICE_TOKEN:
        return cams

    # Sanitasi URL untuk konsumsi frontend / publik
    return [
        schemas.CameraResponse(
            id=c.id,
            name=c.name,
            location=c.location,
            url=mask_camera_url(c.url),
            protocol=c.protocol,
            status=c.status,
            is_active=c.is_active,
            last_seen=c.last_seen,
            created_at=c.created_at
        )
        for c in cams
    ]


@app.post("/api/cameras", response_model=schemas.CameraResponse)
async def create_camera(
    camera_in: schemas.CameraCreate,
    x_internal_token: Optional[str] = Header(None),
    db: Session = Depends(database.get_db)
):
    normalized_url, detected_protocol = normalize_camera_url(
        camera_in.url,
        camera_in.protocol
    )

    new_camera = models.Camera(
        name=camera_in.name,
        location=camera_in.location,
        url=normalized_url,
        protocol=detected_protocol,
        status="OFFLINE",
        is_active=True
    )
    db.add(new_camera)
    db.commit()
    db.refresh(new_camera)

    await manager.broadcast({
        "event": "CAMERA_ADDED",
        "data": {
            "id": new_camera.id,
            "name": new_camera.name,
            "location": new_camera.location,
            "url": mask_camera_url(new_camera.url),
            "protocol": new_camera.protocol,
            "status": new_camera.status
        }
    })

    if x_internal_token == INTERNAL_SERVICE_TOKEN:
        return new_camera

    return schemas.CameraResponse(
        id=new_camera.id,
        name=new_camera.name,
        location=new_camera.location,
        url=mask_camera_url(new_camera.url),
        protocol=new_camera.protocol,
        status=new_camera.status,
        is_active=new_camera.is_active,
        last_seen=new_camera.last_seen,
        created_at=new_camera.created_at
    )


@app.get("/api/cameras/{camera_id}", response_model=schemas.CameraResponse)
def get_camera(
    camera_id: str,
    x_internal_token: Optional[str] = Header(None),
    db: Session = Depends(database.get_db)
):
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if x_internal_token == INTERNAL_SERVICE_TOKEN:
        return camera

    return schemas.CameraResponse(
        id=camera.id,
        name=camera.name,
        location=camera.location,
        url=mask_camera_url(camera.url),
        protocol=camera.protocol,
        status=camera.status,
        is_active=camera.is_active,
        last_seen=camera.last_seen,
        created_at=camera.created_at
    )


@app.patch("/api/cameras/{camera_id}/status", response_model=schemas.CameraResponse)
async def update_camera_status(
    camera_id: str,
    status_in: schemas.CameraStatusUpdate,
    db: Session = Depends(database.get_db)
):
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera.status = status_in.status.upper()
    camera.last_seen = datetime.utcnow()
    db.commit()
    db.refresh(camera)

    await manager.broadcast({
        "event": "CAMERA_STATUS_CHANGED",
        "data": {
            "camera_id": camera.id,
            "status": camera.status,
            "last_seen": camera.last_seen.isoformat() if camera.last_seen else None
        }
    })

    return camera


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str, db: Session = Depends(database.get_db)):
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    db.delete(camera)
    db.commit()

    await manager.broadcast({
        "event": "CAMERA_DELETED",
        "data": {"camera_id": camera_id}
    })

    return {"message": "Camera deleted successfully"}


# --- INCIDENT ALERTS ENDPOINTS ---

@app.get("/api/incident-alerts", response_model=List[schemas.IncidentResponse])
def get_incidents(
    camera_id: Optional[str] = None,
    location: Optional[str] = None,
    unread_only: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(database.get_db)
):
    query = db.query(models.Incident)
    if camera_id:
        query = query.filter(models.Incident.camera_id == camera_id)
    if location:
        query = query.filter(models.Incident.location.ilike(f"%{location}%"))
    if unread_only:
        query = query.filter(models.Incident.is_read == False)
    if start_date:
        query = query.filter(models.Incident.detected_at >= start_date)
    if end_date:
        query = query.filter(models.Incident.detected_at <= end_date)

    incidents = query.order_by(desc(models.Incident.detected_at)).limit(limit).all()
    return incidents


@app.post("/api/incident-alerts", response_model=schemas.IncidentResponse)
async def create_incident(
    camera_id: Optional[str] = Form(None),
    device_id: Optional[int] = Form(None), # Legacy fallback
    confidence: float = Form(...),
    camera_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    clip_duration_seconds: Optional[int] = Form(10),
    thumbnail: UploadFile = File(None),
    clip: UploadFile = File(None),
    db: Session = Depends(database.get_db)
):
    # Resolusi ID kamera
    target_camera_id = camera_id or (f"cam_dev_{device_id}" if device_id else None)
    if not target_camera_id:
        # Buat kamera default jika tidak ada kamera sama sekali
        target_camera_id = "default_cam"

    # Cari referensi kamera di database
    camera = db.query(models.Camera).filter(models.Camera.id == target_camera_id).first()
    if not camera:
        # Auto-create fallback record agar integritas relasi terjaga
        camera = models.Camera(
            id=target_camera_id,
            name=camera_name or f"Kamera {target_camera_id}",
            location=location or "Ruangan Utama",
            url="http://localhost:8080/video",
            protocol="HTTP",
            status="ONLINE"
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)

    resolved_camera_name = camera_name or camera.name
    resolved_location = location or camera.location
    safe_camera_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(target_camera_id))

    # Simpan Thumbnail (dengan sanitasi nama & batas ukuran 10MB)
    ts_now = int(time.time() * 1000)
    snapshot_url = ""
    if thumbnail:
        thumb_bytes = await thumbnail.read()
        if len(thumb_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Thumbnail melebihi batas 10MB")
        thumb_filename = f"thumb_{safe_camera_id}_{ts_now}.jpg"
        thumb_path = os.path.join("storage", "thumbnails", thumb_filename)
        with open(thumb_path, "wb") as f:
            f.write(thumb_bytes)
        snapshot_url = f"/storage/thumbnails/{thumb_filename}"
    else:
        snapshot_url = "/storage/thumbnails/placeholder.jpg"

    # Simpan Clip (dengan sanitasi nama & batas ukuran 60MB)
    clip_url = None
    if clip:
        clip_bytes = await clip.read()
        if len(clip_bytes) > 60 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Klip video melebihi batas 60MB")
        clip_filename = f"clip_{safe_camera_id}_{ts_now}.mp4"
        clip_path = os.path.join("storage", "clips", clip_filename)
        with open(clip_path, "wb") as f:
            f.write(clip_bytes)
        clip_url = f"/storage/clips/{clip_filename}"

    # Simpan record Incident (PRD compliant: tanpa field status penanganan)
    new_incident = models.Incident(
        camera_id=camera.id,
        camera_name=resolved_camera_name,
        location=resolved_location,
        confidence=confidence,
        detected_at=datetime.utcnow(),
        snapshot_url=snapshot_url,
        clip_url=clip_url,
        clip_duration_seconds=clip_duration_seconds or 10,
        is_read=False
    )
    db.add(new_incident)
    db.commit()
    db.refresh(new_incident)

    # Broadcast event real-time ke web client via WebSocket
    incident_payload = {
        "id": new_incident.id,
        "camera_id": new_incident.camera_id,
        "camera_name": new_incident.camera_name,
        "location": new_incident.location,
        "confidence": new_incident.confidence,
        "detected_at": new_incident.detected_at.isoformat(),
        "snapshot_url": new_incident.snapshot_url,
        "clip_url": new_incident.clip_url,
        "clip_duration_seconds": new_incident.clip_duration_seconds,
        "is_read": new_incident.is_read
    }

    await manager.broadcast({
        "event": "INCIDENT_DETECTED",
        "data": incident_payload
    })

    return new_incident


@app.patch("/api/incident-alerts/{incident_id}/read", response_model=schemas.IncidentResponse)
async def mark_incident_read(incident_id: str, db: Session = Depends(database.get_db)):
    incident = db.query(models.Incident).filter(models.Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident.is_read = True
    db.commit()
    db.refresh(incident)

    await manager.broadcast({
        "event": "INCIDENT_READ",
        "data": {"incident_id": incident.id}
    })

    return incident


@app.post("/api/incident-alerts/mark-all-read")
async def mark_all_incidents_read(db: Session = Depends(database.get_db)):
    db.query(models.Incident).filter(models.Incident.is_read == False).update({"is_read": True})
    db.commit()

    await manager.broadcast({
        "event": "INCIDENTS_ALL_READ",
        "data": {}
    })

    return {"message": "All incidents marked as read"}


@app.delete("/api/incident-alerts/{incident_id}")
def delete_incident(incident_id: str, db: Session = Depends(database.get_db)):
    incident = db.query(models.Incident).filter(models.Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Bersihkan file fisik snapshot dan clip dengan proteksi path traversal
    safe_delete_storage_file(incident.snapshot_url)
    safe_delete_storage_file(incident.clip_url)

    db.delete(incident)
    db.commit()
    return {"message": "Incident deleted successfully"}


@app.delete("/api/incident-alerts")
async def delete_all_incidents(db: Session = Depends(database.get_db)):
    incidents = db.query(models.Incident).all()
    for incident in incidents:
        safe_delete_storage_file(incident.snapshot_url)
        safe_delete_storage_file(incident.clip_url)

    db.query(models.Incident).delete()
    db.commit()

    await manager.broadcast({
        "event": "INCIDENTS_ALL_DELETED",
        "data": {}
    })

    return {"message": "All incidents deleted successfully"}


# --- ENGINE PROCESS CONTROLLER (UI BUTTON TEST RUNNER) ---

detection_process = None

@app.post("/api/detection/start")
def start_detection():
    global detection_process
    if detection_process and detection_process.poll() is None:
        return {
            "status": "already_running",
            "is_running": True,
            "message": "Detection script is already running",
            "pid": detection_process.pid
        }

    # Jalankan langsung realtime_fall_detection.py (menggunakan test_fall.mp4)
    script_path = os.path.join("python_script", "realtime_fall_detection.py")

    python_executable = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_executable):
        python_executable = "python"

    detection_process = subprocess.Popen([python_executable, script_path])
    return {
        "status": "started",
        "is_running": True,
        "message": f"Detection worker started with test video ({script_path})",
        "pid": detection_process.pid
    }


@app.post("/api/detection/stop")
def stop_detection():
    global detection_process
    if detection_process and detection_process.poll() is None:
        detection_process.terminate()
        detection_process = None
        return {"status": "stopped", "is_running": False, "message": "Detection script stopped"}
    detection_process = None
    return {"status": "not_running", "is_running": False, "message": "Detection script is not running"}


@app.get("/api/detection/status")
def get_detection_status():
    global detection_process
    is_running = detection_process is not None and detection_process.poll() is None
    return {
        "is_running": is_running,
        "status": "running" if is_running else "stopped",
        "pid": detection_process.pid if is_running else None
    }
