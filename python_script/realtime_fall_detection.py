import time
import math
from collections import deque
import os

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import torch

# =========================
# KONFIGURASI
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gunakan 0 untuk webcam, atau os.path.join(BASE_DIR, "test_fall.mp4") untuk video
VIDEO_SOURCE = os.path.join(BASE_DIR, "test_fall.mp4")
CONF_THRESHOLD = 0.25                # Confidence deteksi orang

# Ambang batas deteksi fisik
ASPECT_RATIO_THRESHOLD = 0.50        # Lebar vs Tinggi kotak untuk indikasi rebah
TORSO_ANGLE_THRESHOLD = 40           # Kemiringan torso (> 35 derajat dari vertikal)
VERTICAL_VELOCITY_THRESHOLD = 10     # Kecepatan turun (px)

INACTIVITY_SECONDS = 0.8             # Durasi diam (detik) untuk konfirmasi jatuh
POSTURE_WINDOW = 5                   # Window voting
POSTURE_VOTE_RATIO = 0.4             # Minimal 2 dari 5 frame konsisten

# Cooldown pendek khusus di titik koordinat yang sama (mencegah spam saat ID kedip)
SAME_SPOT_COOLDOWN = 4.0

ALERT_API_URL = "http://127.0.0.1:8000/api/incident-alerts"

# Index keypoint COCO Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14

torch.set_num_threads(4)


def calc_torso_angle(shoulder_mid, hip_mid):
    """Menghitung sudut kemiringan torso terhadap sumbu vertikal (0 = tegak, 90 = mendatar)."""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def calc_shoulder_tilt(l_sh, r_sh):
    """Menghitung kemiringan garis bahu terhadap sumbu horizontal (0 = bahu rata/tegak)."""
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


def get_valid_point(kpts, person_kpts, idx1, idx2, conf_thresh=0.30):
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


class PersonTracker:
    def __init__(self, display_id, history_len=8):
        self.display_id = display_id
        self.hip_y_history = deque(maxlen=history_len)
        self.box_y_history = deque(maxlen=history_len)
        self.posture_flags = deque(maxlen=POSTURE_WINDOW)
        self.fall_suspected_since = None
        self.last_center = None
        self.last_box = None
        self.last_seen_time = time.time()
        self.alert_sent = False

    def update_posture_flag(self, is_fall_posture):
        self.posture_flags.append(is_fall_posture)
        if len(self.posture_flags) < 2:
            return is_fall_posture
        true_ratio = sum(self.posture_flags) / len(self.posture_flags)
        return true_ratio >= POSTURE_VOTE_RATIO

    def update_state(self, box, hip_pos=None):
        center = get_box_center(box)
        self.box_y_history.append(center[1])
        if hip_pos is not None:
            self.hip_y_history.append(hip_pos[1])
        else:
            self.hip_y_history.clear()

        self.last_center = center
        self.last_box = box
        self.last_seen_time = time.time()

    def vertical_velocity(self):
        if len(self.hip_y_history) >= 3:
            return self.hip_y_history[-1] - min(self.hip_y_history)
        if len(self.box_y_history) >= 3:
            return self.box_y_history[-1] - min(self.box_y_history)
        return 0.0

    def is_inactive(self, current_box, movement_threshold=65):
        if self.last_center is None:
            return True
        current_center = get_box_center(current_box)
        dist = math.dist(current_center, self.last_center)
        return dist < movement_threshold


def send_alert(frame, display_id, confidence):
    filename = f"temp_alert_{display_id}_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    payload = {"device_id": 1, "confidence": round(confidence, 2)}

    try:
        with open(filename, "rb") as img_file:
            response = requests.post(
                ALERT_API_URL, data=payload, files={"thumbnail": img_file}, timeout=3
            )
        if response.status_code == 200:
            print(f"[ALERT SENT] Berhasil kirim alert untuk ID #{display_id}")
        else:
            print(f"[WARNING] Server error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Gagal koneksi ke FastAPI: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Gunakan yolo11n-pose.pt agar selalu ringan dan kencang (high FPS)
    MODEL_PATH = "yolo11n-pose.pt"
    print(f"[INFO] Device: {device} | Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    trackers = {}
    last_alert_time = 0.0
    last_alert_pos = None
    prev_small_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video selesai atau gagal membaca frame.")
            break

        # =====================================================================
        # 1. DETEKSI PERGANTIAN KLIP / SCENE CUT (Mencegah ID Bocor ke Klip Baru)
        # =====================================================================
        small_gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
        if prev_small_gray is not None:
            scene_diff = np.mean(cv2.absdiff(small_gray, prev_small_gray))
            if scene_diff > 32.0:  # Layar berubah drastis (ganti video/adegan)
                print(f"[SCENE CUT] Pergantian adegan terdeteksi (diff={scene_diff:.1f}). Reset tracker.")
                trackers.clear()
                last_alert_pos = None
                last_alert_time = 0.0
        prev_small_gray = small_gray

        # =====================================================================
        # 2. INFERENSI CEPAT: imgsz=480 & bytetrack.yaml (Mengembalikan FPS)
        # =====================================================================
        results = model.track(
            frame,
            imgsz=480,
            persist=True,
            conf=CONF_THRESHOLD,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        annotated_frame = results[0].plot()
        current_time = time.time()
        active_ids = set()

        if results[0].boxes.id is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_data = results[0].keypoints.data.cpu().numpy()

            for i, (box, raw_track_id, kpts) in enumerate(zip(boxes, track_ids, keypoints_xy)):
                box_center = get_box_center(box)
                box_h = box[3] - box[1]

                # =============================================================
                # 3. SPATIAL ID RE-LINKING KETAT (Hanya untuk Orang yang Sama)
                # =============================================================
                if raw_track_id not in trackers:
                    inherited_tracker = None
                    min_dist = 95.0  # Radius diperketat (95 px) & waktu hilang < 1.2 detik

                    for old_id, old_tracker in list(trackers.items()):
                        if old_id not in track_ids and (current_time - old_tracker.last_seen_time) < 1.2:
                            if old_tracker.last_center is not None and old_tracker.last_box is not None:
                                dist = math.dist(box_center, old_tracker.last_center)
                                old_h = old_tracker.last_box[3] - old_tracker.last_box[1]
                                # Pastikan ukuran tinggi kotak tidak berubah ekstrem (bukan orang berbeda)
                                size_ratio = min(box_h, old_h) / (max(box_h, old_h) + 1e-6)
                                if dist < min_dist and size_ratio > 0.5:
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

                # Keypoints confidence
                nose_conf = person_kpts[NOSE][2] if len(person_kpts) > NOSE else 0
                l_sh_conf = person_kpts[L_SHOULDER][2] if len(person_kpts) > L_SHOULDER else 0
                r_sh_conf = person_kpts[R_SHOULDER][2] if len(person_kpts) > R_SHOULDER else 0

                frame_h = frame.shape[0]
                hip_pos = get_valid_point(kpts, person_kpts, L_HIP, R_HIP, 0.40)
                if hip_pos is not None and hip_pos[1] > (frame_h * 0.93):
                    hip_pos = None

                shoulder_pos = get_valid_point(kpts, person_kpts, L_SHOULDER, R_SHOULDER, 0.30)
                knee_pos = get_valid_point(kpts, person_kpts, L_KNEE, R_KNEE, 0.30)

                # Upright Check untuk Close-up Webcam
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

                # =============================================================
                # 4. LOGIKA EVALUASI JATUH
                # =============================================================
                if hip_pos is None:
                    if is_upper_body_upright:
                        raw_posture_flag = False
                        tracker.posture_flags.clear()
                    else:
                        if aspect_ratio > 1.15 or (tracker.fall_suspected_since is not None and aspect_ratio > 0.80):
                            raw_posture_flag = True
                else:
                    if shoulder_pos is not None:
                        torso_angle = calc_torso_angle(shoulder_pos, hip_pos)
                        hip_near_ground = False
                        if knee_pos is not None and abs(hip_pos[1] - knee_pos[1]) < (box_h * 0.22):
                            hip_near_ground = True

                        # Jika orang kembali berdiri tegak (torso lurus & rasio tinggi), langsung reset status jatuh!
                        if torso_angle < 20 and aspect_ratio < 0.65 and not hip_near_ground:
                            raw_posture_flag = False
                            tracker.posture_flags.clear()
                        elif torso_angle > TORSO_ANGLE_THRESHOLD:
                            raw_posture_flag = True
                        elif aspect_ratio > ASPECT_RATIO_THRESHOLD and (
                            torso_angle > 22 or hip_near_ground or v_velocity > VERTICAL_VELOCITY_THRESHOLD
                        ):
                            raw_posture_flag = True
                        elif v_velocity > (VERTICAL_VELOCITY_THRESHOLD * 1.5) and aspect_ratio > 0.60:
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
                            # Anti-spam hanya jika koordinat & waktu sangat berdekatan (< 4 detik & < 100 px)
                            is_same_spot_spam = (
                                last_alert_pos is not None
                                and math.dist(box_center, last_alert_pos) < 100
                                and (current_time - last_alert_time) < SAME_SPOT_COOLDOWN
                            )
                            if not is_same_spot_spam:
                                send_alert(frame, tracker.display_id, confidence=0.9)
                                last_alert_time = current_time
                                last_alert_pos = box_center
                            tracker.alert_sent = True
                    elif tracker.alert_sent:
                        status_text = "FALL DETECTED"
                else:
                    tracker.fall_suspected_since = None
                    tracker.alert_sent = False

                x1, y1 = int(box[0]), int(box[1])
                text_y = y1 - 10 if y1 > 30 else y1 + 25
                color = (
                    (0, 0, 255)
                    if "FALL" in status_text
                    else ((0, 165, 255) if "Suspected" in status_text else (0, 255, 0))
                )
                cv2.putText(
                    annotated_frame,
                    f"ID {tracker.display_id}: {status_text}",
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

        # Bersihkan tracker yang sudah tidak aktif (> 1.5 detik)
        for raw_id, tracker in list(trackers.items()):
            if raw_id not in active_ids:
                if (current_time - tracker.last_seen_time) > 1.5:
                    del trackers[raw_id]

        cv2.imshow("Fall Detection - press 'q' to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()