import cv2
import numpy as np
import mediapipe as mp
from utils.bbox_utils import get_center_of_bbox, compute_containment
from collections import deque


class BallHolderDetector:
    def __init__(self,
                 wrist_dist_thresh=30,
                 distance_thresh=45,
                 max_missed_frames=3,    
                 containment_threshold=0.8,
                 ball_velocity_thresh=12,
                 velocity_buffer_size=3): 
        
        self.wrist_dist_thresh = wrist_dist_thresh
        self.distance_thresh = distance_thresh
        self.max_missed_frames = max_missed_frames
        self.containment_threshold = containment_threshold
        self.ball_velocity_thresh = ball_velocity_thresh

        # MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, 
                                      model_complexity=1, 
                                      min_detection_confidence=0.5)

        # State Tracking
        self.current_holder = -1
        self.consecutive_count = 0      
        self.missed_count = 0           
        
        # Velocity Tracking
        self.prev_ball_center = None
        self.velocity_buffer = deque(maxlen=velocity_buffer_size)

    def detect_single_pose(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None

        h, w, _ = crop.shape
        lm = results.pose_landmarks.landmark
        
        wrists = []
        for i in [15, 16]:
            if lm[i].visibility > 0.35: 
                wx = int(lm[i].x * w) + x1
                wy = int(lm[i].y * h) + y1
                wrists.append((wx, wy))
        
        return wrists if wrists else None

    def get_smoothed_velocity(self, current_center):
        if self.prev_ball_center is None:
            self.prev_ball_center = current_center
            return 0
        
        dist = np.linalg.norm(np.array(current_center) - np.array(self.prev_ball_center))
        self.velocity_buffer.append(dist)
        self.prev_ball_center = current_center
        
        # Return average velocity
        return np.mean(self.velocity_buffer)

    def find_best_candidate(self, frame, players, ball_box, ball_center, ball_speed):
        is_ball_flying = ball_speed > self.ball_velocity_thresh
        
        containment_candidates = []
        wrist_candidates = []
        distance_candidates = []

        # 1. Initial Filtering
        potential_pose_candidates = []

        for pid, pdata in players.items():
            p_bbox = pdata["bbox"]
            
            # 1.1 Check Containment
            if not is_ball_flying:
                ratio = compute_containment(p_bbox, ball_box)
                if ratio >= self.containment_threshold:
                    containment_candidates.append(pid)

            # 1.2 Distance Pre-check 
            dist_rough = np.linalg.norm(np.array(ball_center) - np.array(get_center_of_bbox(p_bbox)))
            
            if dist_rough < self.distance_thresh * 3:
                potential_pose_candidates.append(pid)
            
           # 1.3 Distance Candidates 
            if not is_ball_flying and dist_rough < self.distance_thresh:
                distance_candidates.append((pid, dist_rough))

        # 3. Check Wrist
        current_wrist_thresh = self.wrist_dist_thresh if not is_ball_flying else (self.wrist_dist_thresh * 0.7)
        
        for pid in potential_pose_candidates:
            wrists = self.detect_single_pose(frame, players[pid]["bbox"])
            if wrists:
                dists = [np.linalg.norm(np.array(ball_center) - np.array(w)) for w in wrists]
                if min(dists) < current_wrist_thresh:
                    wrist_candidates.append(pid)

        # --- RANKING ---
        # Priority 1: Wrist
        if wrist_candidates:
            if self.current_holder in wrist_candidates:
                return self.current_holder
            return wrist_candidates[0] 

        # Priority 2: Containment
        if containment_candidates and not is_ball_flying:
            if self.current_holder in containment_candidates:
                return self.current_holder
            return containment_candidates[0]

        # Priority 3: Distance
        if distance_candidates and not is_ball_flying:
            # Sort by distance min
            distance_candidates.sort(key=lambda x: x[1])
            best_pid = distance_candidates[0][0]
            if self.current_holder == best_pid:
                return self.current_holder
            return best_pid

        return -1

    def detect_ball_holders(self, frames, player_tracks, ball_tracks):
        num_frames = len(frames)
        holders_result = [-1] * num_frames
        
        # Reset state 
        self.prev_ball_center = None
        self.velocity_buffer.clear()
        self.current_holder = -1
        self.consecutive_count = 0
        self.missed_count = 0

        for f_idx, frame in enumerate(frames):
            players = player_tracks[f_idx]
            ball_info = ball_tracks[f_idx]

            # 1. Missed Ball
            if 1 not in ball_info or not ball_info[1].get("bbox"):
                # Keep previous holder if within missed frame limit
                if self.current_holder != -1 and self.missed_count < self.max_missed_frames:
                    self.missed_count += 1
                    holders_result[f_idx] = self.current_holder
                else:
                    self.current_holder = -1
                    self.consecutive_count = 0
                    holders_result[f_idx] = -1
                
                self.prev_ball_center = None # Reset velocity calc logic
                self.velocity_buffer.clear()
                continue

            # 2. Ball Present
            ball_box = ball_info[1]["bbox"]
            ball_center = get_center_of_bbox(ball_box)
            
            # Smooth velocity
            ball_speed = self.get_smoothed_velocity(ball_center)

            candidate = self.find_best_candidate(frame, players, ball_box, ball_center, ball_speed)

            # 3. Logic update State 
            if candidate != -1:
                # Find a candidate
                if candidate == self.current_holder:
                    self.consecutive_count += 1
                    self.missed_count = 0 # Reset missed count
                # New candidate
                else:
                    self.consecutive_count = 1
                    self.current_holder = candidate 
                    self.missed_count = 0
            else:
                # No candidate found
                if ball_speed > self.ball_velocity_thresh * 1.5:
                    # Flying ball 
                    self.current_holder = -1
                    self.consecutive_count = 0
                else:
                    # Stationary ball
                    if self.current_holder != -1 and self.missed_count < self.max_missed_frames:
                        self.missed_count += 1
                        # Keep previous holder
                    else:
                        self.current_holder = -1
                        self.consecutive_count = 0

            holders_result[f_idx] = self.current_holder

        return holders_result
