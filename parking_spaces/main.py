import cv2
import pickle
import numpy as np

# Load video feed
cap = cv2.VideoCapture('parking_lot.mp4')

# Load parking spot positions
with open('CarParkPos', 'rb') as f:
    pos_list = pickle.load(f)

# Define width and height of parking spots
width = 43
height = 95

# Function to draw a gradient rectangle
def draw_gradient_rectangle(img, start_point, end_point, start_color, end_color):
    x1, y1 = start_point
    x2, y2 = end_point
    for i in range(y1, y2):
        alpha = (i - y1) / (y2 - y1)
        color = tuple([int(start_color[j] * (1 - alpha) + end_color[j] * alpha) for j in range(3)])
        cv2.line(img, (x1, i), (x2, i), color, 1)

# Function to check parking space occupancy
def check_parking_space(img_pro, img):
    space_counter = 0
    total_spots = 0
    
    # Check each defined parking spot
    for pos in pos_list:
        x, y = pos
        
        # Check if the rectangle is within image boundaries
        if x >= 0 and y >= 0 and x + width < img_pro.shape[1] and y + height < img_pro.shape[0]:
            img_crop = img_pro[y:y + height, x:x + width]
            count = cv2.countNonZero(img_crop)

            if count < 860:
                color = (0, 255, 0)
                thickness = 5
                space_counter += 1
            else:
                color = (0, 0, 255)
                thickness = 2

            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
            total_spots += 1

    # Calculate dimensions for centering gradient box and text
    box_width = 350
    box_height = 70
    start_point = ((img.shape[1] - box_width) // 2, (img.shape[0] - box_height) // 2)
    end_point = (start_point[0] + box_width, start_point[1] + box_height)

    # Draw the gradient box
    draw_gradient_rectangle(img, start_point, end_point, (0, 200, 0), (0, 100, 0))
    cv2.rectangle(img, start_point, end_point, (50, 50, 50), 2, cv2.LINE_AA)

    # Draw the counter with adjusted size and position
    text = f'Free: {space_counter}/{total_spots}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Adjust font scale for smaller text
    text_thickness = 2  # Adjust text thickness
    text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
    
    text_x = start_point[0] + (box_width - text_size[0]) // 2
    text_y = start_point[1] + (box_height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

# Process frames from the video feed and call the function from above 
while True:
    # Check if the current position is the last frame and restart video if true
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()

    # Convert image to gray, blur, and convert to inverse binary to detect non-zero pixels
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 16)

    # Remove stray dots to clean the image
    img_median = cv2.medianBlur(img_threshold, 5)

    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img_median, kernel, iterations=1)

    # Call function to check parking space occupancy
    check_parking_space(img_dilate, img)

    # Display image with updated parking spot information
    cv2.imshow("Image", img)
    cv2.waitKey(10)