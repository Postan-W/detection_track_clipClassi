import cv2
import os
from PIL import Image
import numpy as np
# Path to the directory containing images
image_dir = "C:/Users/wmingdru/Desktop/temp"

# Get the list of image files
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# Sort the image files
image_files.sort()

# Get the first image to get the dimensions
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape

# Define the output video file
output_video = './videos/pose/image_compose.avi'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

# Read and write images to the video
for image_file in image_files:
    frame = cv2.imread(image_file)
    image = Image.fromarray(frame)
    image = image.resize((width, height))
    frame = np.array(image)
    out.write(frame)

# Release everything when done
out.release()
cv2.destroyAllWindows()