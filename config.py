import numpy as np

VIDEO_PATH = 'traffic_video.mp4' 
MODEL_PATH = 'yolov8x.pt'

# =============================
# Konfigurasi Sistem Monitoring
# =============================

import numpy as np

# Path video dan model YOLO yang digunakan
VIDEO_PATH = 'traffic_video.mp4' 
MODEL_PATH = 'yolov8x.pt'

# Kelas kendaraan yang dideteksi: 2=Mobil, 3=Motor, 5=Bus, 7=Truk
ALLOWED_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {2: 'Mobil', 3: 'Sepeda Motor', 5: 'Bus', 7: 'Truk'}

# --- Area Garis Start/End untuk Hitung & Speed ---
# Garis atas untuk mulai hitung kecepatan (Lajur 1-3)
START_LINE_Y = 300
# Garis atas untuk mulai hitung kecepatan (Lajur 4)
START_LINE_Y_LANE4 = 500
# Garis bawah sebagai trigger akhir hitung (Lajur 1-3)
END_LINE_Y = 420
# Garis bawah sebagai trigger akhir hitung (Lajur 4)
END_LINE_Y_LANE4 = 620

# --- Kalibrasi Jarak untuk Speed ---
# Jarak sebenarnya (meter) antara garis start dan end
# Ganti sesuai kondisi lapangan agar hasil speed akurat
DISTANCE_METERS = 4.5

# --- Definisi Poligon Tiap Lajur (Lane) ---
# Setiap poligon mewakili area lajur pada frame video
LANE_POLYGONS = [
    # Lajur 1 (paling kiri)
    np.array([[160, 420], [310, 420], [375, 300], [250, 300]], np.int32),
    # Lajur 2
    np.array([[310, 420], [490, 420], [500, 300], [375, 300]], np.int32),
    # Lajur 3
    np.array([[540, 420], [720, 420], [720, 300], [550, 300]], np.int32),
    # Lajur 4 (kanan, area merge, diperbesar vertikal)
    np.array([[800, 700], [1150, 700], [1200, 500], [900, 500]], np.int32)
]
