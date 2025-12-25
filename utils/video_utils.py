import cv2
import os

def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Video FPS: {fps}")

    frames = []
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()

    cap.release()
    return frames, fps


def write_video(frames, save_path, fps=24, codec='XVID'):
    if len(frames) == 0:
        raise ValueError("Empty frame list, nothing to write.")
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Cannot write video to: {save_path}")
    for frm in frames:
        writer.write(frm)
    writer.release()
