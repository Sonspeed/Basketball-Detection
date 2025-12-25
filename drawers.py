# drawers.py
from utils import draw_ellipse, draw_traingle
import numpy as np
import cv2
from collections import deque


class PlayerTracksDrawer:
    def __init__(self, team_1_color=[255,245,238], team_2_color=[128,0,0], holder_color=[0,0,255]):
        self.default_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color
        self.holder_color = holder_color 

    def draw(self, frames, tracks, holders=None):
      
        output_frames = []

        for i, frame in enumerate(frames):
            frame_copy = frame.copy()
            player_dict = tracks[i]
            holder_id = None
            if holders is not None:
                holder_id = holders[i]
            else:
                print("Warning: holders is None, skipping ball holder drawing.")

            for track_id, player in player_dict.items():
                team_id = player.get('team', self.default_team_id)
                color = self.team_1_color if team_id==1 else self.team_2_color
                frame_copy = draw_ellipse(frame_copy, player['bbox'], color, track_id)

                if track_id == holder_id:
                    frame_copy = draw_traingle(frame_copy, player['bbox'], self.holder_color)

            output_frames.append(frame_copy)

        return output_frames


class BallTracksDrawer:
    def __init__(self, color=(0, 255, 0)):
        self.ball_color = color

    def draw(self, frames, tracks):
        output_frames = []
        for i, frame in enumerate(frames):
            frame_copy = frame.copy()
            balls = tracks[i]

            for _, ball in balls.items():  
                if ball.get("bbox") is not None:
                    frame_copy = draw_traingle(frame_copy, ball["bbox"], self.ball_color)

            output_frames.append(frame_copy)  

        return output_frames

class TeamControlBallDrawer:
    def __init__(self, window_size=300, easing=0.1):
        # Sliding window to track recent possession
        self.window_size = window_size
        self.control_history = deque(maxlen=window_size)

        # Smoothed values used only for rendering (animation easing)
        self.display_t1 = 50.0
        self.display_t2 = 50.0
        self.easing = easing

    # Color gradient
    def _red_gradient(self, strength):
        base = np.array([60, 60, 180], dtype=np.float32)
        strong = np.array([0, 0, 255], dtype=np.float32)
        color = base + strength * (strong - base)
        return tuple(int(c) for c in color)
    
    def _blue_gradient(self, strength):
        base = np.array([180, 60, 60], dtype=np.float32)
        strong = np.array([255, 0, 0], dtype=np.float32)
        color = base + strength * (strong - base)
        return tuple(int(c) for c in color)

  
    def draw(self, frames, player_tracks, holders):

        output_frames = []

        # Reset history for a new video
        self.control_history.clear()
        self.display_t1 = 50.0
        self.display_t2 = 50.0

        for frame_idx, frame in enumerate(frames):
            frame_copy = frame.copy()

            # 1. Determine current team in control
            holder_id = holders[frame_idx]
            current_team_id = -1

            if holder_id != -1:
                frame_players = player_tracks[frame_idx]
                if holder_id in frame_players:
                    current_team_id = frame_players[holder_id].get("team", -1)

            # 2. Update sliding window (ball alive only)
            if current_team_id in [1, 2]:
                self.control_history.append(current_team_id)

            # 3. Compute possession percentage
            t1_pct, t2_pct = 50.0, 50.0

            if len(self.control_history) > 0:
                history = list(self.control_history)
                t1_count = history.count(1)
                t2_count = history.count(2)
                total = t1_count + t2_count

                if total > 0:
                    t1_pct = (t1_count / total) * 100
                    t2_pct = (t2_count / total) * 100

            # 4. Apply animation easing (EMA)
            self.display_t1 += self.easing * (t1_pct - self.display_t1)
            self.display_t2 += self.easing * (t2_pct - self.display_t2)

            # 5. Draw UI
            frame_copy = self.draw_scoreboard(
                frame_copy,
                self.display_t1,
                self.display_t2
            )

            output_frames.append(frame_copy)

        return output_frames
    
    # Scoreboard rendering
    def draw_scoreboard(self, frame, t1_pct, t2_pct):

        h, w = frame.shape[:2]

        # UI layout configuration
        bar_width = 300
        bar_height = 30
        center_x = w // 2
        start_y = h - 80

        x1 = center_x - bar_width // 2
        x2 = center_x + bar_width // 2
        y1 = start_y
        y2 = start_y + bar_height

        overlay = frame.copy()

        # Background bar
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (40, 40, 40),
            -1
        )

        # Dominance strength for gradient intensity
        dominance = abs(t1_pct - 50.0) / 50.0
        dominance = np.clip(dominance, 0.0, 1.0)

        # Gradient colors
        t1_color = self._red_gradient(dominance)
        t2_color = self._blue_gradient(dominance)

        # Split position based on Team 1 percentage
        split_x = int(x1 + bar_width * (t1_pct / 100.0))

        # Team 1 bar
        cv2.rectangle(
            overlay,
            (x1, y1),
            (split_x, y2),
            t1_color,
            -1
        )

        # Team 2 bar
        cv2.rectangle(
            overlay,
            (split_x, y1),
            (x2, y2),
            t2_color,
            -1
        )

        # 50% reference line
        cv2.line(
            overlay,
            (center_x, y1),
            (center_x, y2),
            (255, 255, 255),
            2
        )

        # Alpha blending
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Text labels
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            frame,
            f"{t1_pct:.0f}%",
            (x1 - 60, y1 + 25),
            font,
            0.8,
            t1_color,
            2
        )

        cv2.putText(
            frame,
            f"{t2_pct:.0f}%",
            (x2 + 10, y1 + 25),
            font,
            0.8,
            t2_color,
            2
        )

        # Subtitle
        cv2.putText(
            frame,
            "Possession (Last 10s)",
            (center_x - 80, y2 + 20),
            font,
            0.5,
            (200, 200, 200),
            1
        )

        return frame
