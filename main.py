from ultralytics import YOLO
from utils import load_video, write_video
from trackers import BallTracker
from trackers import PlayerTracker
from utils import read_stub, save_stub
from drawers import PlayerTracksDrawer, BallTracksDrawer, TeamControlBallDrawer
from team_classifier import TeamClassifier
from ballholder_detector import BallHolderDetector


def main():

    player_model = YOLO("model/player_detector.pt")
    ball_model = YOLO("model/ball_detector.pt")

    video_frames, fps = load_video("input_videos/video_1.mp4")

    ball_tracker = BallTracker(ball_model)
    player_tracker = PlayerTracker(player_model)

    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/player_tracks_video_1.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path="stubs/ball_tracks_video_1.pkl")
    print(player_tracks)
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    team_classifier = TeamClassifier()
    team_classifier.classify_teams(video_frames, player_tracks, read_from_stub=False, stub_path="stubs/team_labels_video_1.pkl")
    holder_detector = BallHolderDetector(
        wrist_dist_thresh=30,
        containment_threshold=0.8,
        distance_thresh=45,
    )
    holders = holder_detector.detect_ball_holders(video_frames, player_tracks, ball_tracks)
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_control_drawer = TeamControlBallDrawer()

    output_video_frames = player_tracks_drawer.draw(video_frames, 
                                                    player_tracks, holders=holders)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)
    output_video_frames = team_control_drawer.draw(output_video_frames, player_tracks, holders)

    write_video(output_video_frames, "output_videos/original_video_1.avi")

if __name__ == '__main__':
    main()
