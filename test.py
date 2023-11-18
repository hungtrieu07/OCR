# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd

# # Load the image in grayscale
# img = cv2.imread('body.jpg', cv2.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"

# cropped = img[65:387, 74:1553]

# cv2.imwrite('cropped.jpg', cropped)

# # 74, 74
# # 1553, 387
# # Get the width and height of the image
# height, width = cropped.shape

# # Initialize an array to store the column-wise counts
# column_counts = np.zeros(width, dtype=int)

# # Iterate over each column and count black pixels
# for x in range(height):
#     for y in range(width):
#         if img[x, y] == 0:
#             column_counts[y] += 1

# # Set the window size for the moving average
# window_size = 3  # Adjust this as needed for smoothing

# # Calculate the moving average
# smoothed_array = np.convolve(column_counts, np.ones(window_size)/window_size, mode='valid')

# # Create a list of x-axis values for the bar chart
# x_values = np.arange(len(smoothed_array))

# # Create subplots in a single figure
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Plot the smoothed data as a bar chart on the first subplot
# ax1.bar(x_values, smoothed_array)
# ax1.set_xlabel('X-axis')
# ax1.set_ylabel('Y-axis')
# ax1.set_title('Smoothed Bar Chart')

# # Plot the column-wise black pixel count chart on the second subplot
# ax2.bar(range(width), column_counts, color='b', alpha=0.7)
# ax2.set_title('Column-Wise Black Pixel Counts')
# ax2.set_xlabel('Column Index')
# ax2.set_ylabel('Number of Black Pixels')
# ax2.grid(axis='y')

# # Adjust the space between the subplots
# plt.tight_layout()

# # Show the combined figure with both charts
# plt.show()

import cv2

# Sample bounding box coordinates and corresponding content
bounding_boxes = [(1114.0, 4.0, 1237.0, 29.0), (1359.0, 0.0, 1450.0, 26.0), (3.0, 13.0, 262.0, 37.0),
                  (402.0, 14.0, 533.0, 39.0), (607.0, 13.0, 941.0, 33.0), (1.0, 39.0, 219.0, 66.0),
                  (602.0, 39.0, 727.0, 63.0), (3.0, 72.0, 285.0, 92.0), (2.0, 98.0, 393.0, 122.0),
                  (0.0, 125.0, 29.0, 150.0), (0.0, 155.0, 117.0, 180.0), (0.0, 183.0, 62.0, 208.0),
                  (2.0, 211.0, 166.0, 236.0), (2.0, 240.0, 191.0, 264.0), (3.0, 295.0, 304.0, 319.0)]

# Group bounding boxes based on x-axis proximity
grouped_boxes = []
current_group = []

# Sorting bounding boxes based on their x-coordinate
sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0])

# Set a threshold for x-axis proximity, adjust as needed
x_proximity_threshold = 20

# Iterate through sorted boxes and group them
for i in range(len(sorted_boxes) - 1):
    current_box = sorted_boxes[i]
    next_box = sorted_boxes[i + 1]

    # Check if the next box is within the x-axis proximity threshold
    if next_box[0] - current_box[2] <= x_proximity_threshold:
        current_group.append(current_box)
    else:
        current_group.append(current_box)
        grouped_boxes.append(current_group)
        current_group = []

# Add the last box to the last group
current_group.append(sorted_boxes[-1])
grouped_boxes.append(current_group)

# Merge bounding boxes within each group (considering y-axis proximity as well)
merged_boxes = []
for group in grouped_boxes:
    min_x = min(group, key=lambda x: x[0])[0]
    min_y = min(group, key=lambda x: x[1])[1]
    max_x = max(group, key=lambda x: x[2])[2]
    max_y = max(group, key=lambda x: x[3])[3]

    merged_boxes.append((min_x, min_y, max_x, max_y))

# Draw bounding boxes based on grouping
image = cv2.imread("cropped.jpg")
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0), (255, 0, 255)]  # Colors for different groups

for i, box in enumerate(bounding_boxes):
    x1, y1, x2, y2 = map(int, box)
    for j, merged_box in enumerate(merged_boxes):
        if x1 >= merged_box[0] and x2 <= merged_box[2] and y1 >= merged_box[1] and y2 <= merged_box[3]:
            color = colors[j % len(colors)]  # Cycle through colors for different groups
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

cv2.imwrite("result.jpg", image)

