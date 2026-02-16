import numpy as np

def generate_frames(num_frames, width=512, height=512):
    frames = []

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # simple animation example
        x = int((i / num_frames) * width)
        frame[:, x:x+10] = (0, 255, 0)

        frames.append(frame)

    return frames
