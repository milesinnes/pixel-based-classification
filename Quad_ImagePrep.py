# Script for pre-processing quadrat photos for pixel analysis.
# Off-nadir and non-centered images are cropped to a square and selected corners and distorted (perspective).
# Instructions: Configure photo input/output folders. Click internal quadrat corners (clockwise) starting at top-left. Repeat.
# Miles Innes. February 17, 2026.

### Imports
import cv2
import numpy as np
import os

### Configuration (ADD FILE PATHS)
INPUT_FOLDER = '' # Add path to unprocessed photos
OUTPUT_FOLDER = '' # Add output folder location for cropped and transformed photos
OUTPUT_SIZE = 1000  # Pixels for the final square (e.g., 1000x1000)

### Confirm file path
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

### Point-and-Click Function (manually click quadrat corners)
points = []

def click_event(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN: #If left-mouse button clicked...
        points.append((x, y)) #Save those coordinates
        # Draw a small circle to show where you clicked
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Warping Tool", img_display)



### Parse through imagery, defining corners for perspective distortion
# Get list of images
images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in images:
    img_path = os.path.join(INPUT_FOLDER, img_name)
    img = cv2.imread(img_path)
    if img is None: continue

    # Resize for display if the image is huge (keeps aspect ratio)
    scale = 1000 / img.shape[0]
    img_display = cv2.resize(img, None, fx=scale, fy=scale)

    points = []
    cv2.imshow("Warping Tool", img_display)
    cv2.setMouseCallback("Warping Tool", click_event)

    print(f"Processing {img_name}. Click 4 corners: TL, TR, BR, BL. Press 'r' to reset, 'q' to quit.")

    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Reset clicks for this image
            img_display = cv2.resize(img, None, fx=scale, fy=scale)
            points = []
            cv2.imshow("Warping Tool", img_display)
        if key == ord('q'):
            exit()

    # Convert display points back to original image scale
    src_pts = np.float32(points) / scale

    # Define destination: a perfect square (the locations that selected corners will be shifted to)
    dst_pts = np.float32([
        [0, 0],
        [OUTPUT_SIZE, 0],
        [OUTPUT_SIZE, OUTPUT_SIZE],
        [0, OUTPUT_SIZE]
    ])

    # Calculate transformation and warp
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts) # Defines output matrix
    warped = cv2.warpPerspective(img, matrix, (OUTPUT_SIZE, OUTPUT_SIZE)) # Perform warp transformation (make nadir)

    # Save output
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{img_name}"), warped)
    print(f"Saved rectified {img_name}")

cv2.destroyAllWindows()
print("Batch complete!")
