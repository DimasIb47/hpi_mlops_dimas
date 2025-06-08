from ultralytics import YOLO

# =============================
# Kelas Deteksi Kendaraan (YOLO)
# =============================

class VehicleDetector:
    def __init__(self, model_path):
        # Load model YOLO sesuai path
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame, allowed_classes):
        # Deteksi kendaraan di frame, filter hanya kelas yang diizinkan
        detections = []
        results = self.model(frame, verbose=False)[0]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            # Threshold confidence khusus motor lebih rendah
            conf_thresh = 0.15 if class_id == 3 else 0.25
            # Heuristik: kalau bus/truk kecil, anggap mobil
            if class_id in [5, 7]:
                width = x2 - x1
                height = y2 - y1
                if width < 40 or height < 40:
                    class_id = 2
            if class_id in allowed_classes and confidence >= conf_thresh:
                detections.append([x1, y1, x2, y2, confidence, class_id])
        return detections