import os
import sys
import cv2
import numpy as np

image_folder = sys.argv[1]
video_name = sys.argv[2]

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
print(sorted(os.listdir(image_folder)))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

