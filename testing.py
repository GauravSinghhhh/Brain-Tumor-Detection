import cv2
import os

# Check if the image file exists
image_path = 'datasets\\no\\no0.jpg'
if not os.path.exists(image_path):
    print(f"Error: The image file at {image_path} does not exist.")
else:
    # Try to read the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error: Unable to read the image file at {image_path}.")
    else:
        # If the image is read successfully, get the dimensions
        height, width, channels = image.shape
        num_pixels = height * width
        print(f"The number of pixels in the image is: {num_pixels}")
