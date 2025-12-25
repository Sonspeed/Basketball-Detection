# track.py
import supervision as sv
import numpy as np
import pandas as pd
from utils.stub_utils import read_stub, save_stub


class BallTracker:
    def __init__(self, model):
        self.model = model

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections.extend(self.model.predict(batch, conf=0.5))
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        cached = read_stub(read_from_stub, stub_path)
        if cached is not None and len(cached) == len(frames):
            return cached

        detections = self.detect_frames(frames)
        tracks = []

        for idx, det in enumerate(detections):
            class_map = det.names
            inv_map = {v: k for k, v in class_map.items()}
            det_sv = sv.Detections.from_ultralytics(det)

            tracks.append({})
            best_box = None
            best_conf = 0

            for obj in det_sv:
                box = obj[0].tolist()
                conf = obj[2]
                cls_id = obj[3]

                if cls_id == inv_map.get("Ball", -1) and conf > best_conf:
                    best_box = box
                    best_conf = conf

            if best_box is not None:
                tracks[idx][1] = {"bbox": best_box}

        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, positions):
        max_dist = 25
        last_idx = -1

        for i in range(len(positions)):
            curr = positions[i].get(1, {}).get("bbox", [])

            if not curr:
                continue
            if last_idx == -1:
                last_idx = i
                continue

            prev = positions[last_idx].get(1, {}).get("bbox", [])
            gap = i - last_idx
            allowed = max_dist * gap

            if np.linalg.norm(np.array(prev[:2]) - np.array(curr[:2])) > allowed:
                positions[i] = {}
            else:
                last_idx = i

        return positions

    def interpolate_ball_positions(self, positions):
        boxes = [f.get(1, {}).get("bbox", []) for f in positions]
        df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]


class PlayerTracker:
 
    def __init__(self, model):
      
        self.model = model 
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20):
    
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
       
        tracks = read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)

        tracks=[]

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox}
        
        save_stub(stub_path,tracks)
        return tracks

    