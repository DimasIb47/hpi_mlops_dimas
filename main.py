import cv2
import numpy as np
from sort import Sort
import collections
import time
import config
from vehicle_detector import VehicleDetector
import analysis_utils
import visualizer
import json
from datetime import datetime

def main():
    # Inisialisasi detektor, tracker, dan video
    detector = VehicleDetector(config.MODEL_PATH)
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    cap = cv2.VideoCapture(config.VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = 'traffic_output.mp4' # Output video hasil analisis
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Mulai analisis dengan logika dua garis (start-end)
    print(f"Memulai analisis dengan Logika Dua Garis...")

    # Variabel untuk data kendaraan dan statistik
    vehicle_data = {}
    counted_ids = set()
    recently_counted = []  # Buffer untuk menghindari double count
    stats = {
        'total': 0,
        'by_type': collections.defaultdict(int),
        'by_lane': collections.defaultdict(int)
    }
    
    frame_count = 0
    start_time = time.time()

    # List untuk hasil JSON
    json_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Tampilkan progress analisis setiap 20 frame
        if frame_count % 20 == 0:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.2f}% | Frame: {frame_count}/{total_frames} | ETA: {eta:.2f} detik", end="")

        detections = detector.detect_vehicles(frame, config.ALLOWED_CLASSES)
        detections_np = np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
        tracked_objects = tracker.update(detections_np)

        # Vehicle tracking and statistics
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)
            center_x, center_y = (x1 + x2) // 2, y2
            if obj_id not in vehicle_data:
                vehicle_data[obj_id] = {'path': [], 'crossed_start': False, 'start_time': 0, 'speed': 0, 'type_id': -1}
            # Maintain a buffer of the last 5 positions
            vehicle_data[obj_id]['path'].append((center_x, center_y))
            if len(vehicle_data[obj_id]['path']) > 5:
                vehicle_data[obj_id]['path'].pop(0)
            if vehicle_data[obj_id]['type_id'] == -1:
                for det in detections:
                    dx1, dy1, dx2, dy2, _, class_id = det
                    if abs(x1 - dx1) < 20 and abs(y1 - dy1) < 20:
                        vehicle_data[obj_id]['type_id'] = class_id
                        break
            # After assignment, never update type_id again
            path = vehicle_data[obj_id]['path']
            # Determine if this vehicle is in lane 4 polygon at the current position
            # Use bottom center for motorcycles, centroid for others
            x1, y1, x2, y2, _ = map(int, obj)
            if vehicle_data[obj_id]['type_id'] == 3:
                current_point = ((x1 + x2) // 2, y2)
            else:
                current_point = path[-1]
            in_lane4 = cv2.pointPolygonTest(config.LANE_POLYGONS[3], current_point, False) >= 0
            # Set start/end lines based on lane
            if in_lane4:
                start_y = config.START_LINE_Y_LANE4
                end_y = config.END_LINE_Y_LANE4
            else:
                start_y = config.START_LINE_Y
                end_y = config.END_LINE_Y
            # Check start line (across all pairs in buffer)
            if not vehicle_data[obj_id]['crossed_start'] and len(path) > 1:
                for i in range(1, len(path)):
                    if path[i-1][1] < start_y <= path[i][1]:
                        vehicle_data[obj_id]['crossed_start'] = True
                        vehicle_data[obj_id]['start_time'] = frame_count - (len(path) - 1 - i)
                        break
            # Check end line (across all pairs in buffer)
            if vehicle_data[obj_id]['crossed_start'] and obj_id not in counted_ids and len(path) > 1:
                for i in range(1, len(path)):
                    if path[i-1][1] < end_y <= path[i][1]:
                        # Check if this box is close to a recently counted box (avoid double count)
                        this_box = (x1, y1, x2, y2)
                        is_duplicate = False
                        for prev_box, prev_frame in recently_counted:
                            # Compute IOU
                            xx1 = max(this_box[0], prev_box[0])
                            yy1 = max(this_box[1], prev_box[1])
                            xx2 = min(this_box[2], prev_box[2])
                            yy2 = min(this_box[3], prev_box[3])
                            w = max(0, xx2 - xx1)
                            h = max(0, yy2 - yy1)
                            inter = w * h
                            area1 = (this_box[2] - this_box[0]) * (this_box[3] - this_box[1])
                            area2 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                            union = area1 + area2 - inter
                            iou = inter / union if union > 0 else 0
                            if iou > 0.6 and abs(frame_count - prev_frame) < 10:
                                is_duplicate = True
                                break
                        if is_duplicate:
                            break
                        counted_ids.add(obj_id)
                        recently_counted.append((this_box, frame_count))
                        if len(recently_counted) > 20:
                            recently_counted.pop(0)
                        start_frame = vehicle_data[obj_id]['start_time']
                        duration_frames = frame_count - (len(path) - 1 - i) - start_frame
                        duration_seconds = duration_frames / fps if fps > 0 else 0
                        if duration_seconds > 0:
                            speed_kmh = (config.DISTANCE_METERS / duration_seconds) * 3.6
                            vehicle_data[obj_id]['speed'] = speed_kmh
                        # Dedicated function for lane 4 counting
                        type_id = vehicle_data[obj_id]['type_id']
                        if analysis_utils.count_lane4_crossing(path, config.START_LINE_Y_LANE4, config.END_LINE_Y_LANE4, config.LANE_POLYGONS[3]):
                            final_lane = 4
                        else:
                            crossing_point = ((x1 + x2) // 2, y2) if type_id == 3 else path[i]
                            final_lane = analysis_utils.get_lane_by_point(crossing_point, config.LANE_POLYGONS)
                            # If not found, fallback to nearest polygon by x-distance
                            if final_lane is None:
                                px = crossing_point[0]
                                min_dist = float('inf')
                                for idx, poly in enumerate(config.LANE_POLYGONS, 1):
                                    poly_x = np.mean([pt[0] for pt in poly])
                                    dist = abs(px - poly_x)
                                    if dist < min_dist:
                                        min_dist = dist
                                        final_lane = idx

                        # Simpan data kendaraan ke list untuk output JSON
                        stats['total'] += 1
                        vehicle_type = config.CLASS_NAMES.get(type_id, 'Unknown')
                        stats['by_type'][vehicle_type] += 1
                        if final_lane is not None:
                            stats['by_lane'][final_lane] += 1
                        # Hitung timestamp detik dari frame
                        waktu_detik = frame_count / fps if fps > 0 else 0
                        # Simpan hasil ke list
                        if 'json_results' not in locals():
                            json_results = []
                        json_results.append({
                            'vehicle_id': int(obj_id),
                            'timestamp': round(waktu_detik, 2),
                            'tipe': vehicle_type,
                            'kecepatanperjam': round(vehicle_data[obj_id]['speed'], 2),
                            'lajur': int(final_lane) if final_lane is not None else None
                        })
                        break

        # --- VISUALIZATION ---
        frame = visualizer.draw_lanes_and_line(
            frame, config.LANE_POLYGONS,
            config.START_LINE_Y, config.END_LINE_Y,
            config.START_LINE_Y_LANE4, config.END_LINE_Y_LANE4
        )
        visualizer.draw_tracked_vehicles(frame, tracked_objects, vehicle_data, config.CLASS_NAMES)
        visualizer.draw_statistics(frame, stats)

        cv2.imshow("Realtime Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        vehicle_data[obj_id]['path'].append((center_x, center_y))
        if vehicle_data[obj_id]['type_id'] == -1:
            for det in detections:
                 dx1, dy1, dx2, dy2, _, class_id = det
            if abs(x1 - dx1) < 20 and abs(y1 - dy1) < 20:
                    vehicle_data[obj_id]['type_id'] = class_id
                    break

            path = vehicle_data[obj_id]['path']
            
            # 1. Cek lintasan START_LINE
            if not vehicle_data[obj_id]['crossed_start'] and len(path) > 1 and path[-2][1] < config.START_LINE_Y <= path[-1][1]:
                vehicle_data[obj_id]['crossed_start'] = True
                vehicle_data[obj_id]['start_time'] = frame_count

            # 2. Cek lintasan END_LINE (jika sudah melewati START)
            if vehicle_data[obj_id]['crossed_start'] and obj_id not in counted_ids:
                if len(path) > 1 and path[-2][1] < config.END_LINE_Y <= path[-1][1]:
                    counted_ids.add(obj_id)

                    start_frame = vehicle_data[obj_id]['start_time']
                    duration_frames = frame_count - start_frame
                    duration_seconds = duration_frames / fps if fps > 0 else 0

                    if duration_seconds > 0:
                        speed_kmh = (config.DISTANCE_METERS / duration_seconds) * 3.6
                        vehicle_data[obj_id]['speed'] = speed_kmh
                    
                    final_lane = analysis_utils.get_dominant_lane(path, config.LANE_POLYGONS)
                    
                    stats['total'] += 1
                    vehicle_type = config.CLASS_NAMES.get(vehicle_data[obj_id]['type_id'], 'Unknown')
                    stats['by_type'][vehicle_type] += 1
                    if final_lane is not None:
                        stats['by_lane'][final_lane] += 1

        # --- GAMBAR VISUALISASI ---
        # CUKUP PANGGIL SATU FUNGSI INI DARI VISUALIZER
        visualizer.draw_lanes_and_line(frame, config.LANE_POLYGONS, config.START_LINE_Y, config.END_LINE_Y)
        
        # Gambar info kendaraan dan statistik
        visualizer.draw_tracked_vehicles(frame, tracked_objects, vehicle_data, config.CLASS_NAMES)
        visualizer.draw_statistics(frame, stats)

        cv2.imshow("Realtime Monitoring", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

        video_writer.write(frame)

    end_time = time.time()
    total_processing_time = end_time - start_time
    print(f"\nAnalisis selesai dalam {total_processing_time:.2f} detik.")

    # Setelah selesai, simpan hasil JSON ke file
    with open('hasil_analisis.json', 'w', encoding='utf-8') as fjson:
        json.dump(json_results, fjson, ensure_ascii=False, indent=2)
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()