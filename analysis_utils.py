import cv2
import collections

# =============================
# Fungsi Analisis & Utilitas
# =============================

def get_vehicle_lane(point, lane_polygons):
    """
    Cek di lajur mana sebuah titik kendaraan berada.
    point: (x, y) koordinat kendaraan
    lane_polygons: list poligon lajur
    Return: nomor lajur (1 dst), None jika di luar lajur
    """
    for i, polygon in enumerate(lane_polygons):
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return i + 1
    return None

def estimate_speed(entry_frame, current_frame, fps, distance_meters):
    """
    Hitung estimasi kecepatan (km/jam) dari dua crossing frame.
    """
    if fps == 0:
        return 0
    time_in_frames = current_frame - entry_frame
    time_in_seconds = time_in_frames / fps
    if time_in_seconds <= 0:
        return 0
    speed_mps = distance_meters / time_in_seconds  # m/s
    speed_kmh = speed_mps * 3.6  # ke km/jam
    return speed_kmh

def count_lane4_crossing(path, start_y, end_y, polygon):
    """
    Cek crossing khusus lajur 4: crossing path dari atas start ke bawah end, dan titik crossing ada di poligon lajur 4.
    Return True jika crossing valid.
    """
    import cv2
    for i in range(1, len(path)):
        if path[i-1][1] < start_y <= path[i][1] and path[i][1] >= end_y:
            if cv2.pointPolygonTest(polygon, path[i], False) >= 0:
                return True
    return False

def get_lane_by_point(point, lane_polygons):
    """
    Cek index lajur (1-based) dari titik tertentu.
    Return None jika di luar semua poligon.
    """
    for idx, poly in enumerate(lane_polygons, 1):
        if cv2.pointPolygonTest(poly, point, False) >= 0:
            return idx
    return None

def get_dominant_lane(path, lane_polygons, num_last_points=10):
    """
    Cari lajur dominan dari histori path kendaraan (biar nggak salah assign kalau nempel garis).
    """
    if len(path) < num_last_points:
        points_to_check = path
    else:
        points_to_check = path[-num_last_points:]
    lane_votes = []
    for point in points_to_check:
        lane = get_vehicle_lane(point, lane_polygons)
        if lane is not None:
            lane_votes.append(lane)
    if not lane_votes:
        return None
    # Ambil lajur yang paling sering muncul
    counter = collections.Counter(lane_votes)
    dominant_lane = counter.most_common(1)[0][0]
    return dominant_lane