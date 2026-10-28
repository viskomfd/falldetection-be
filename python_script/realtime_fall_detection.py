import time
import math
from collections import deque
import os
import re
import threading
import subprocess
from typing import Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import torch
import imageio_ffmpeg

# =============================================================================
# KONFIGURASI GLOBAL & THRESHOLD HEURISTIK JATUH
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_SOURCE = os.path.join(BASE_DIR, "test_fall.mp4")

# Ambang batas deteksi fisik (Kalibrasi 3D Grid Search + Kompensasi Perspektif CCTV)
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.15"))
ASPECT_RATIO_THRESHOLD = 0.55        # Lebar vs Tinggi kotak untuk indikasi rebah menyamping
LONGITUDINAL_AR_THRESHOLD = 0.40     # Ambang rasio saat orang rebah membujur
TORSO_ANGLE_THRESHOLD = 45.0         # Kemiringan torso (> 45 derajat dari vertikal)
VERTICAL_VELOCITY_THRESHOLD = 8.0    # Kecepatan turun instan antar-frame (px/frame)
SLOW_FALL_DROP_THRESHOLD = 28.0      # Total penurunan vertikal (px) dalam window
INACTIVITY_SECONDS = float(os.getenv("INACTIVITY_SECONDS", "0.8"))
POSTURE_WINDOW = 5                   # Window voting
POSTURE_VOTE_RATIO = 0.4             # Minimal 2 dari 5 frame konsisten
SAME_SPOT_COOLDOWN = 4.0             # Mencegah spam alert pada titik koordinat yang sama

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ALERT_API_URL = f"{API_BASE_URL}/api/incident-alerts"
CAMERA_STATUS_URL = f"{API_BASE_URL}/api/cameras"

DEBUG_GUI = os.getenv("DEBUG_GUI", "true").lower() in ("true", "1")
INFERENCE_FPS = int(os.getenv("INFERENCE_TARGET_FPS", "10"))
CLIP_BUFFER_SECONDS = 5  # 5 detik sebelum dan 5 detik sesudah jatuh

# Index keypoint COCO Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

torch.set_num_threads(4)


# =============================================================================
# HELPER MATEMATIKA & POSE
# =============================================================================

def mask_url(url: str) -> str:
    """Masks password in rtsp://user:password@host -> rtsp://user:***@host"""
    return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1***\3', str(url))


def calc_torso_angle(shoulder_mid, hip_mid):
    """Menghitung sudut kemiringan torso terhadap sumbu vertikal (0 = tegak, 90 = mendatar)."""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def calc_shoulder_tilt(l_sh, r_sh):
    """Menghitung kemiringan garis bahu terhadap sumbu horizontal."""
    dx = abs(r_sh[0] - l_sh[0])
    dy = abs(r_sh[1] - l_sh[1])
    return math.degrees(math.atan2(dy, dx + 1e-6))


def calc_aspect_ratio(box):
    """Menghitung rasio lebar dibanding tinggi bounding box."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    return w / (h + 1e-6)


def get_box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def get_valid_point(kpts, person_kpts, idx1, idx2, conf_thresh=0.25):
    """Mengambil titik tengah jika kedua keypoint valid, atau salah satu yang valid."""
    c1 = person_kpts[idx1][2] if len(person_kpts) > idx1 else 0
    c2 = person_kpts[idx2][2] if len(person_kpts) > idx2 else 0
    if c1 > conf_thresh and c2 > conf_thresh:
        return ((kpts[idx1][0] + kpts[idx2][0]) / 2, (kpts[idx1][1] + kpts[idx2][1]) / 2)
    elif c1 > conf_thresh:
        return (kpts[idx1][0], kpts[idx1][1])
    elif c2 > conf_thresh:
        return (kpts[idx2][0], kpts[idx2][1])
    return None


# =============================================================================
# EXPORT VIDEO BROWSER-COMPATIBLE (H.264 / AVC1)
# =============================================================================

def export_browser_compatible_clip(frames: list, output_path: str, fps: int = 20) -> bool:
    """
    Meng-encode kumpulan frame menjadi video MP4 dengan codec H.264 / yuv420p
    menggunakan static binary FFmpeg dari imageio-ffmpeg.
    Dipastikan langsung dapat diputar pada elemen <video> browser tanpa kendala codec.
    """
    if not frames:
        return False
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        h, w = frames[0].shape[:2]
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(max(1, fps)),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        print(f"[ERROR] Transcoding video clip gagal: {e}")
        return False


def dispatch_alert_async(
    api_url: str,
    camera_id: str,
    camera_name: str,
    location: str,
    confidence: float,
    snapshot_frame: np.ndarray,
    clip_frames: list,
    fps: int = 20
):
    """Mengirim alert beserta snapshot dan video clip ke FastAPI secara non-blocking."""
    def _worker():
        ts = int(time.time() * 1000)
        thumb_file = f"temp_thumb_{camera_id}_{ts}.jpg"
        clip_file = f"temp_clip_{camera_id}_{ts}.mp4"
        files = {}
        opened_handles = []

        try:
            cv2.imwrite(thumb_file, snapshot_frame)
            if os.path.exists(thumb_file):
                f_thumb = open(thumb_file, "rb")
                opened_handles.append(f_thumb)
                files["thumbnail"] = ("thumb.jpg", f_thumb, "image/jpeg")

            has_clip = export_browser_compatible_clip(clip_frames, clip_file, fps=fps)
            if has_clip and os.path.exists(clip_file):
                f_clip = open(clip_file, "rb")
                opened_handles.append(f_clip)
                files["clip"] = ("clip.mp4", f_clip, "video/mp4")

            payload = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "location": location,
                "confidence": round(confidence, 2),
                "clip_duration_seconds": int(len(clip_frames) / max(1, fps)) if has_clip else 0
            }

            resp = requests.post(api_url, data=payload, files=files, timeout=12)
            if resp.status_code == 200:
                print(f"[ALERT DISPATCHED] Kamera '{camera_name}' (ID: {camera_id}) -> Insiden tercatat (HTTP 200)")
            else:
                print(f"[ALERT ERROR] Server merespon {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[ALERT EXCEPTION] Gagal mengirim alert ke backend: {e}")
        finally:
            for handle in opened_handles:
                try:
                    handle.close()
                except Exception:
                    pass
            if os.path.exists(thumb_file):
                try:
                    os.remove(thumb_file)
                except OSError:
                    pass
            if os.path.exists(clip_file):
                try:
                    os.remove(clip_file)
                except OSError:
                    pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# =============================================================================
# PERSON TRACKER
# =============================================================================

class PersonTracker:
    def __init__(self, display_id, history_len=20):
        self.display_id = display_id
        self.hip_y_history = deque(maxlen=history_len)
        self.box_y_history = deque(maxlen=history_len)
        self.box_h_history = deque(maxlen=history_len)
        self.posture_flags = deque(maxlen=POSTURE_WINDOW)
        self.fall_suspected_since = None
        self.last_center = None
        self.last_box = None
        self.max_upright_height = 0.0
        self.last_seen_time = time.time()
        self.alert_sent = False

    def update_posture_flag(self, is_fall_posture):
        self.posture_flags.append(is_fall_posture)
        if len(self.posture_flags) < 2:
            return is_fall_posture
        true_ratio = sum(self.posture_flags) / len(self.posture_flags)
        return true_ratio >= POSTURE_VOTE_RATIO

    def update_state(self, box, hip_pos=None, is_upright_normal=False):
        center = get_box_center(box)
        box_h = box[3] - box[1]
        self.box_y_history.append(center[1])
        self.box_h_history.append(box_h)

        if is_upright_normal and box_h > self.max_upright_height:
            self.max_upright_height = box_h

        if hip_pos is not None:
            self.hip_y_history.append(hip_pos[1])

        self.last_center = center
        self.last_box = box
        self.last_seen_time = time.time()

    def vertical_velocity(self):
        """Kecepatan turun instan per-frame (px/frame)."""
        if len(self.hip_y_history) >= 2:
            instant_v = self.hip_y_history[-1] - self.hip_y_history[-2]
            window_v = (self.hip_y_history[-1] - self.hip_y_history[0]) / max(1, len(self.hip_y_history) - 1)
            return max(instant_v, window_v)
        if len(self.box_y_history) >= 2:
            instant_v = self.box_y_history[-1] - self.box_y_history[-2]
            window_v = (self.box_y_history[-1] - self.box_y_history[0]) / max(1, len(self.box_y_history) - 1)
            return max(instant_v, window_v)
        return 0.0

    def total_vertical_drop(self):
        """Total penurunan posisi Y dalam window 20 frame."""
        drop_hip = (self.hip_y_history[-1] - min(self.hip_y_history)) if len(self.hip_y_history) >= 3 else 0.0
        drop_box = (self.box_y_history[-1] - min(self.box_y_history)) if len(self.box_y_history) >= 3 else 0.0
        return max(drop_hip, drop_box)

    def is_inactive(self, current_box, movement_threshold=65):
        if self.last_center is None:
            return True
        current_center = get_box_center(current_box)
        dist = math.dist(current_center, self.last_center)
        return dist < movement_threshold


# =============================================================================
# THREADED CAMERA STREAM READER
# =============================================================================

class ThreadedCamera:
    """
    Membaca frame secara terus menerus dalam thread terpisah agar buffer tidak lag
    dan menangani auto-reconnect dengan exponential backoff jika stream terputus.
    """
    def __init__(self, source, on_status_change=None):
        self.source = int(source) if str(source).isdigit() else source
        self.on_status_change = on_status_change
        self.cap = None
        self.frame = None
        self.fps = 25.0
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _notify_status(self, connected: bool):
        if self.is_connected != connected:
            self.is_connected = connected
            if self.on_status_change:
                self.on_status_change("ONLINE" if connected else "OFFLINE")

    def _capture_loop(self):
        backoff = 2
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                masked = mask_url(str(self.source))
                print(f"[STREAM] Mencoba koneksi ke {masked}...")
                self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps_prop and 5.0 <= fps_prop <= 60.0:
                        self.fps = fps_prop
                    self._notify_status(True)
                    backoff = 2
                    print(f"[STREAM] Berhasil tersambung ke {masked} ({self.fps:.1f} FPS)")
                else:
                    self._notify_status(False)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Jika sumber adalah file video lokal (test_fall.mp4), loop otomatis dari awal
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.04)
                    continue

                self._notify_status(False)
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            with self.lock:
                self.frame = frame

            time.sleep(1.0 / (self.fps * 1.5))

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self._notify_status(False)


# =============================================================================
# WORKER DETEKSI JATUH PER KAMERA
# =============================================================================

class FallDetectionWorker:
    """
    Worker deteksi jatuh independen per kamera.
    Mendukung frame sampling, circular buffer forensic MP4, dan auto-reconnect.
    """
    def __init__(
        self,
        camera_id: str = "default_cam",
        camera_name: str = "Kamera Utama",
        location: str = "Ruangan",
        stream_url: str = DEFAULT_VIDEO_SOURCE,
        model_instance: Optional[YOLO] = None,
        debug_gui: bool = False
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.location = location
        self.stream_url = stream_url
        self.debug_gui = debug_gui
        self.is_running = False
        self.worker_thread = None

        # Shared YOLO model atau inisialisasi lokal
        if model_instance is not None:
            self.model = model_instance
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = os.path.join(BASE_DIR, "..", "yolo11n-pose.pt")
            if not os.path.exists(model_path):
                model_path = "yolo11n-pose.pt"
            self.model = YOLO(model_path).to(device)

    def _report_status(self, status: str):
        masked = mask_url(self.stream_url)
        print(f"[STATUS] Kamera '{self.camera_name}' ({self.camera_id}): {status}")
        try:
            requests.patch(
                f"{CAMERA_STATUS_URL}/{self.camera_id}/status",
                json={"status": status},
                timeout=3
            )
        except Exception:
            pass

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        return self

    def stop(self):
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

    def _run(self):
        camera = ThreadedCamera(self.stream_url, on_status_change=self._report_status).start()
        trackers = {}
        last_alert_time = 0.0
        last_alert_pos = None
        prev_small_gray = None

        # Circular buffer untuk klip forensik (e.g. 5 detik sebelum jatuh)
        buffer_len = int(camera.fps * CLIP_BUFFER_SECONDS)
        pre_fall_buffer = deque(maxlen=max(30, buffer_len))
        post_fall_recording = False
        post_fall_frames = []
        pending_incident = None

        frame_count = 0
        inference_interval = max(1, int(camera.fps / INFERENCE_FPS))

        print(f"[WORKER] FallDetectionWorker dimulai untuk '{self.camera_name}' (ID: {self.camera_id})")

        while self.is_running:
            frame = camera.read()
            if frame is None:
                time.sleep(0.04)
                continue

            frame_count += 1
            pre_fall_buffer.append(frame)

            # Jika sedang merekam frame setelah jatuh
            if post_fall_recording:
                post_fall_frames.append(frame)
                if len(post_fall_frames) >= int(camera.fps * CLIP_BUFFER_SECONDS):
                    post_fall_recording = False
                    if pending_incident is not None:
                        combined_frames = list(pending_incident["pre_frames"]) + post_fall_frames
                        dispatch_alert_async(
                            api_url=ALERT_API_URL,
                            camera_id=self.camera_id,
                            camera_name=self.camera_name,
                            location=self.location,
                            confidence=pending_incident["confidence"],
                            snapshot_frame=pending_incident["snapshot_frame"],
                            clip_frames=combined_frames,
                            fps=int(camera.fps)
                        )
                    pending_incident = None
                    post_fall_frames = []

            # Frame Sampling: Jalankan inferensi YOLO hanya setiap N frame untuk hemat compute
            if frame_count % inference_interval != 0:
                time.sleep(0.005)
                continue

            # 1. SCENE CUT DETECTION
            small_gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
            if prev_small_gray is not None:
                scene_diff = np.mean(cv2.absdiff(small_gray, prev_small_gray))
                if scene_diff > 32.0:
                    trackers.clear()
                    last_alert_pos = None
                    last_alert_time = 0.0
            prev_small_gray = small_gray

            # 2. INFERENSI POSE & TRACKING
            results = self.model.track(
                frame,
                imgsz=640,
                persist=True,
                conf=CONF_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            annotated_frame = results[0].plot() if self.debug_gui else None
            current_time = time.time()
            active_ids = set()
            frame_h, frame_w = frame.shape[:2]

            if results[0].boxes.id is not None and results[0].keypoints is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()
                keypoints_data = results[0].keypoints.data.cpu().numpy()

                for i, (box, raw_track_id, kpts) in enumerate(zip(boxes, track_ids, keypoints_xy)):
                    box_center = get_box_center(box)
                    box_w = box[2] - box[0]
                    box_h = box[3] - box[1]

                    # 3. SPATIAL ID RE-LINKING
                    if raw_track_id not in trackers:
                        inherited_tracker = None
                        min_dist = 145.0
                        for old_id, old_tracker in list(trackers.items()):
                            if old_id not in track_ids and (current_time - old_tracker.last_seen_time) < 2.5:
                                if old_tracker.last_center is not None:
                                    dist = math.dist(box_center, old_tracker.last_center)
                                    if dist < min_dist:
                                        min_dist = dist
                                        inherited_tracker = (old_id, old_tracker)

                        if inherited_tracker is not None:
                            old_id, tracker_obj = inherited_tracker
                            del trackers[old_id]
                            trackers[raw_track_id] = tracker_obj
                        else:
                            trackers[raw_track_id] = PersonTracker(display_id=raw_track_id)

                    tracker = trackers[raw_track_id]
                    active_ids.add(raw_track_id)

                    raw_posture_flag = False
                    person_kpts = keypoints_data[i]
                    aspect_ratio = calc_aspect_ratio(box)

                    nose_conf = person_kpts[NOSE][2] if len(person_kpts) > NOSE else 0
                    l_sh_conf = person_kpts[L_SHOULDER][2] if len(person_kpts) > L_SHOULDER else 0
                    r_sh_conf = person_kpts[R_SHOULDER][2] if len(person_kpts) > R_SHOULDER else 0

                    hip_pos = get_valid_point(kpts, person_kpts, L_HIP, R_HIP, 0.30)
                    if hip_pos is not None and hip_pos[1] > (frame_h * 0.94):
                        hip_pos = None

                    shoulder_pos = get_valid_point(kpts, person_kpts, L_SHOULDER, R_SHOULDER, 0.25)
                    knee_pos = get_valid_point(kpts, person_kpts, L_KNEE, R_KNEE, 0.25)

                    is_upper_body_upright = False
                    if nose_conf > 0.40 and shoulder_pos is not None:
                        nose_y = kpts[NOSE][1]
                        if (shoulder_pos[1] - nose_y) > (box_h * 0.12):
                            if l_sh_conf > 0.35 and r_sh_conf > 0.35:
                                sh_tilt = calc_shoulder_tilt(kpts[L_SHOULDER], kpts[R_SHOULDER])
                                if sh_tilt < 35:
                                    is_upper_body_upright = True
                            else:
                                is_upper_body_upright = True

                    inactive = tracker.is_inactive(box)
                    tracker.update_state(box, hip_pos)
                    v_velocity = tracker.vertical_velocity()
                    total_drop = tracker.total_vertical_drop()

                    # 4. LOGIKA EVALUASI JATUH
                    if hip_pos is None:
                        if is_upper_body_upright:
                            raw_posture_flag = False
                            tracker.posture_flags.clear()
                        else:
                            if aspect_ratio > 0.85 or total_drop > SLOW_FALL_DROP_THRESHOLD or tracker.fall_suspected_since is not None:
                                raw_posture_flag = True
                    else:
                        if shoulder_pos is not None:
                            torso_angle = calc_torso_angle(shoulder_pos, hip_pos)
                            torso_len = math.dist(shoulder_pos, hip_pos)

                            hip_near_ground = False
                            if knee_pos is not None and abs(hip_pos[1] - knee_pos[1]) < (box_h * 0.24):
                                hip_near_ground = True

                            head_collapsed = (nose_conf > 0.30 and kpts[NOSE][1] >= (shoulder_pos[1] - box_h * 0.04))
                            body_foreshortened = (torso_len < box_w * 0.75) and (aspect_ratio > LONGITUDINAL_AR_THRESHOLD)
                            in_floor_zone = (box[1] > frame_h * 0.35)

                            is_longitudinal_floor_pose = (
                                in_floor_zone
                                and aspect_ratio > LONGITUDINAL_AR_THRESHOLD
                                and (head_collapsed or body_foreshortened or hip_near_ground or torso_angle > 22.0)
                            )
                            is_slow_fall = (total_drop > SLOW_FALL_DROP_THRESHOLD) and (
                                aspect_ratio > LONGITUDINAL_AR_THRESHOLD or torso_angle > 22.0 or hip_near_ground
                            )

                            if (
                                torso_angle < 18.0
                                and aspect_ratio < LONGITUDINAL_AR_THRESHOLD
                                and not hip_near_ground
                                and not head_collapsed
                            ):
                                raw_posture_flag = False
                                tracker.posture_flags.clear()
                            elif torso_angle > TORSO_ANGLE_THRESHOLD:
                                raw_posture_flag = True
                            elif aspect_ratio > ASPECT_RATIO_THRESHOLD and (
                                torso_angle > 20.0 or hip_near_ground or v_velocity > VERTICAL_VELOCITY_THRESHOLD or in_floor_zone
                            ):
                                raw_posture_flag = True
                            elif v_velocity > VERTICAL_VELOCITY_THRESHOLD and (
                                aspect_ratio > LONGITUDINAL_AR_THRESHOLD or torso_angle > 22.0 or hip_near_ground
                            ):
                                raw_posture_flag = True
                            elif is_slow_fall or is_longitudinal_floor_pose:
                                raw_posture_flag = True
                        else:
                            if aspect_ratio > LONGITUDINAL_AR_THRESHOLD and (box[1] > frame_h * 0.35):
                                raw_posture_flag = True

                    posture_indicates_fall = tracker.update_posture_flag(raw_posture_flag)

                    status_text = "Normal"
                    if posture_indicates_fall:
                        status_text = "Suspected Fall"
                        if tracker.fall_suspected_since is None:
                            tracker.fall_suspected_since = current_time

                        duration = current_time - tracker.fall_suspected_since

                        if duration >= INACTIVITY_SECONDS and inactive:
                            status_text = "FALL DETECTED"
                            if not tracker.alert_sent:
                                is_same_spot_spam = (
                                    last_alert_pos is not None
                                    and math.dist(box_center, last_alert_pos) < 100
                                    and (current_time - last_alert_time) < SAME_SPOT_COOLDOWN
                                )
                                if not is_same_spot_spam:
                                    print(f"[FALL DETECTED] Person #{tracker.display_id} di '{self.camera_name}'!")
                                    # Snapshot dengan bounding box
                                    snapshot_img = frame.copy()
                                    x1, y1, x2, y2 = [int(v) for v in box]
                                    cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                    cv2.putText(
                                        snapshot_img,
                                        f"FALL DETECTED: ID {tracker.display_id}",
                                        (x1, max(25, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 0, 255),
                                        2
                                    )

                                    # Siapkan perekaman post-fall context untuk klip video
                                    post_fall_recording = True
                                    post_fall_frames = []
                                    pending_incident = {
                                        "confidence": 0.92,
                                        "snapshot_frame": snapshot_img,
                                        "pre_frames": list(pre_fall_buffer)
                                    }

                                    last_alert_time = current_time
                                    last_alert_pos = box_center
                                tracker.alert_sent = True
                        elif tracker.alert_sent:
                            status_text = "FALL DETECTED"
                    else:
                        tracker.fall_suspected_since = None
                        tracker.alert_sent = False

                    if self.debug_gui and annotated_frame is not None:
                        x1, y1 = int(box[0]), int(box[1])
                        text_y = y1 - 10 if y1 > 30 else y1 + 25
                        color = (0, 0, 255) if "FALL" in status_text else ((0, 165, 255) if "Suspected" in status_text else (0, 255, 0))
                        cv2.putText(
                            annotated_frame,
                            f"ID {tracker.display_id}: {status_text}",
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                        )

            # Bersihkan tracker kadaluarsa
            for raw_id, tracker in list(trackers.items()):
                if raw_id not in active_ids:
                    if (current_time - tracker.last_seen_time) > 2.5:
                        del trackers[raw_id]

            if self.debug_gui and annotated_frame is not None:
                cv2.imshow(f"Rumi Vision - {self.camera_name}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        camera.stop()
        if self.debug_gui:
            cv2.destroyAllWindows()
        print(f"[WORKER] FallDetectionWorker untuk '{self.camera_name}' dihentikan.")


# =============================================================================
# STANDALONE EXECUTION (Untuk pengujian lokal)
# =============================================================================

def main():
    print("=== Rumi Fall Detection Engine (Standalone Mode) ===")
    worker = FallDetectionWorker(
        camera_id="cam_demo_01",
        camera_name="Ruang Tamu",
        location="Lantai 1",
        stream_url=DEFAULT_VIDEO_SOURCE,
        debug_gui=DEBUG_GUI
    )
    try:
        worker.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMenghentikan worker...")
        worker.stop()
        print("Selesai.")


if __name__ == "__main__":
    main()
