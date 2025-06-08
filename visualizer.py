import cv2
import config

import cv2
import config

# =============================
# Fungsi Visualisasi Frame Video
# =============================

def draw_lanes_and_line(frame, polygons, start_y, end_y, start_y_lane4=None, end_y_lane4=None):
    """
    Gambar semua poligon lajur dan garis start/end pada frame.
    Lajur 4 pakai warna beda biar jelas.
    """
    overlay = frame.copy()
    # Poligon lajur 1-3 (warna biru kemerahan)
    for polygon in polygons[:3]:
        cv2.fillPoly(overlay, [polygon], color=(0, 100, 255))
    # Poligon lajur 4 (oranye)
    cv2.fillPoly(overlay, [polygons[3]], color=(255, 140, 0))
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    # Outline tiap lajur
    for i, polygon in enumerate(polygons):
        color = (0, 100, 255) if i < 3 else (255, 140, 0)
        cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)
    height, width, _ = frame.shape
    # Garis start/end lajur 1-3
    cv2.line(frame, (0, start_y), (width, start_y), (0, 255, 0), 2)
    cv2.line(frame, (0, end_y), (width, end_y), (0, 0, 255), 2)
    # Garis start/end lajur 4
    if start_y_lane4 is not None and end_y_lane4 is not None:
        cv2.line(frame, (0, start_y_lane4), (width, start_y_lane4), (0, 255, 255), 2)
        cv2.line(frame, (0, end_y_lane4), (width, end_y_lane4), (0, 165, 255), 2)
    return frame

def draw_tracked_vehicles(frame, tracked_objects, vehicle_data, class_names):
    """
    Gambar kotak, ID, dan info tiap kendaraan yang sedang dilacak.
    """
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        data = vehicle_data.get(obj_id)
        if data:
            vehicle_type = class_names.get(data['type_id'], 'Unknown')
            label = f"ID:{obj_id} {vehicle_type}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Tampilkan speed jika sudah ada
            if data['speed'] > 0:
                speed_label = f"{data['speed']:.1f} km/j"
                cv2.putText(frame, speed_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_statistics(frame, stats):
    """Menampilkan papan statistik di pojok kiri atas."""
    y_offset = 30
    cv2.putText(frame, f"Total Kendaraan: {stats['total']}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, f"Total Kendaraan: {stats['total']}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    type_text = (f"Mobil: {stats['by_type'].get('Mobil', 0)}, "
                 f"Sepeda Motor: {stats['by_type'].get('Sepeda Motor', 0)}, "
                 f"Bus: {stats['by_type'].get('Bus', 0)}, "
                 f"Truk: {stats['by_type'].get('Truk', 0)}")
    cv2.putText(frame, type_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(frame, type_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30

    for i in range(1, len(config.LANE_POLYGONS) + 1):
        lane_text = f"Lajur {i}: {stats['by_lane'].get(i, 0)}"
        cv2.putText(frame, lane_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, lane_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30