import cv2
import numpy as np

def write_video(frames, output_path, fps=24):
    """
    frames: list of numpy arrays (H x W x 3)
    output_path: output .mp4 path
    fps: frames per second
    """

    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()
