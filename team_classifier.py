import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils.stub_utils import read_stub, save_stub


class TeamClassifier:
    def __init__(self):
        self.team_kmeans = None
        self.team_colors = [] 

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        w = x2 - x1
        h = y2 - y1
        
        # Center Crop: 20% height, 40% width
        center_y = y1 + h // 2
        center_x = x1 + w // 2
        crop_h = int(h * 0.2) 
        crop_w = int(w * 0.4) 

        y_start = max(0, center_y - crop_h // 2)
        y_end = min(frame.shape[0], center_y + crop_h // 2)
        x_start = max(0, center_x - crop_w // 2)
        x_end = min(frame.shape[1], center_x + crop_w // 2)

        crop = frame[y_start:y_end, x_start:x_end]
        
        if crop.size == 0: return None

        kmeans = self.get_clustering_model(crop)
        labels = kmeans.labels_
        clustered_img = kmeans.cluster_centers_.astype(int)
        label_counts = np.bincount(labels)
        dominant_label = np.argmax(label_counts)
        
        return clustered_img[dominant_label]

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        return interArea / boxAArea

    def classify_teams(self, frames, tracks, read_from_stub=False, stub_path=None):
        team_labels = read_stub(read_from_stub, stub_path)
        if team_labels is not None:
            for frame_tracks in tracks:
                for tid in frame_tracks:
                    if tid in team_labels:
                        frame_tracks[tid]['team'] = team_labels[tid]
            return team_labels

        print("Starting team classification...")
        
        player_clean_colors = {}  
        player_dirty_colors = {}  

        for frame_num, frame_tracks in enumerate(tracks):
            frame = frames[frame_num]
            
            all_bboxes = []
            for tid, info in frame_tracks.items():
                if 'bbox' in info:
                    all_bboxes.append((tid, info['bbox']))

            for track_id, track_info in frame_tracks.items():
                if "bbox" not in track_info: continue
                
                current_bbox = track_info['bbox']
                
                # --- CHECK OVERLAP ---
                is_overlapping = False
                for other_tid, other_bbox in all_bboxes:
                    if track_id == other_tid: continue
                    if self.compute_iou(current_bbox, other_bbox) > 0.3: 
                        is_overlapping = True
                        break
                
                color = self.get_player_color(frame, current_bbox)
                if color is None: continue

                # --- CLASSIFY DATA ---
                if is_overlapping:
                    if track_id not in player_dirty_colors:
                        player_dirty_colors[track_id] = []
                    player_dirty_colors[track_id].append(color)
                else:
                    if track_id not in player_clean_colors:
                        player_clean_colors[track_id] = []
                    player_clean_colors[track_id].append(color)

        # 1. Training KMeans 
        player_avg_colors = []
        clean_track_ids = []

        for track_id, color_list in player_clean_colors.items():
            if len(color_list) > 0:
                avg_color = np.mean(color_list, axis=0)
                player_avg_colors.append(avg_color)
                clean_track_ids.append(track_id)
        
        if len(player_avg_colors) < 2:
            print("Not enough clean data to classify teams.")
            return {}

        player_avg_colors = np.array(player_avg_colors)
        self.team_kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        self.team_kmeans.fit(player_avg_colors)
        self.team_colors = self.team_kmeans.cluster_centers_

        # Predict teams for clean data
        clean_predictions = self.team_kmeans.predict(player_avg_colors)
        track_team_map = {tid: int(pred)+1 for tid, pred in zip(clean_track_ids, clean_predictions)}

        # 2. Fallback for dirty data
        print(f"Classified {len(track_team_map)} players from clean data. Proceeding with fallback for dirty data...")
        
        # Find unassigned track IDs
        all_dirty_ids = set(player_dirty_colors.keys())
        assigned_ids = set(track_team_map.keys())
        unassigned_ids = all_dirty_ids - assigned_ids
        
        count_fallback = 0
        for tid in unassigned_ids:
            colors = player_dirty_colors[tid]
            if len(colors) == 0: continue
            
            # Avg color from dirty data
            dirty_avg_color = np.mean(colors, axis=0).reshape(1, -1)
            
            #Predict team
            pred_team = self.team_kmeans.predict(dirty_avg_color)[0]
            
            track_team_map[tid] = int(pred_team) + 1
            count_fallback += 1

        print(f"Finished team classification. Total players classified: {len(track_team_map)} (including {count_fallback} by fallback).")

        # 3. Assign teams back to tracks
        for frame_tracks in tracks:
            for tid in frame_tracks:
                if tid in track_team_map:
                    frame_tracks[tid]['team'] = track_team_map[tid]
                    frame_tracks[tid]['team_color'] = self.team_colors[track_team_map[tid]-1]

        if stub_path:
            save_stub(stub_path, track_team_map)

        return track_team_map