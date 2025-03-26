import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET  # Assuming you have the model definition
from utils import load_checkpoint, save_checkpoint
import os
import csv
from collections import deque
# Hyperparameters
IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Single frame processing at a time

# Define the transformations for the video frames
transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# Load the pre-trained model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)  # Load the trained model
model.eval()  # Set the model to evaluation mode

# Function to calculate area and write to CSV
# def write_area_to_csv(frame_number, mask, csv_file_path, [area, EF]):
#     # Calculate the area of the mask (number of non-zero pixels)
#     area = np.sum(mask)  # Sum of pixels where mask value is 1 , repeated here 
    
#     # Write to CSV file
#     with open(csv_file_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         # Check if it's the first row, if so, write headers
#         if file.tell() == 0:
#             writer.writerow(["Frame Number", "Area (pixels)"])
#         writer.writerow([frame_number, area])

import csv
import numpy as np

import csv
import numpy as np

def write_area_to_csv(frame_number, mask, csv_file_path, additional_data=None, header_names=None):
    """
    Writes the frame number, area, and additional data (like EF) to a CSV file with custom headers.

    Parameters:
        frame_number (int): The frame number in the video.
        mask (numpy.ndarray): The binary mask (where object pixels are 1 and background is 0).
        csv_file_path (str): Path to the CSV file where data will be written.
        additional_data (list, optional): List of additional data to append as columns, like [area, EF]. Defaults to None.
        header_names (list, optional): List of custom headers to use for the CSV columns. If None, default headers are used.
    """
    # Calculate the area of the mask (number of non-zero pixels)
    area = np.sum(mask)  # Sum of pixels where mask value is 1

    # If no additional data is provided, use an empty list
    if additional_data is None:
        additional_data = []

    # Append area to the additional data
    additional_data.append(area)

    # Check if header names are provided, otherwise use default headers
    if header_names is None:
        header_names = ["Frame Number", "Area (pixels)"] + [f"Column {i+1}" for i in range(len(additional_data)-2)]

    # Write to CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if it's the first row, if so, write headers
        if file.tell() == 0:
            writer.writerow(header_names)

        # Write the data for the current frame
        writer.writerow([frame_number] + additional_data)




# Function to process and segment each frame
def segment_frame(frame):
    # Apply the transformations
    frame = transform(image=frame)["image"]
    frame = frame.unsqueeze(0).to(DEVICE)  # Add batch dimension and move to the device

    # Make predictions
    with torch.no_grad():
        output = model(frame)
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities

    # Convert output to binary mask (1 for the object, 0 for the background)
    output = output.squeeze().cpu().numpy()
    output = (output > 0.5).astype(np.uint8)  # Threshold the output

    return output

areas = [0, 0]  # currMaxArea, currMinArea
areaBuffer = []

# def getEF(area,period):
    
#     areaBuffer.push(area)
#     if areaBuffer.length > period:
#         areaBuffer.pop first element
#     global areas  # Access the global 'areas' variable
#     currMax = areas[0]
#     currMin = areas[1]

#     if area > currMax:
#         currMax = area
#         if currMin == 0:  # Check if currMin is 0, then set it to the first area value
#             currMin = area  
#     elif area < currMin:
#         currMin = area
    
#     # Update the global areas list
#     areas = [currMax, currMin]

#     print(f"Current Max Area: {currMax}")
#     print(f"Current Min Area: {currMin}")

#     return ((currMax-currMin)/currMax)
    # EF = 0
    # if currMax <= area:
    #     currMax = area
    #     currMin = currMax
    # elif currMin >= area:
    #     currMin = area
    # else :
    #    EF = (currMax - currMin)/currMax
    #    currMax = 0
    # areas = [currMax, currMin]
    # print(areas)
    # return EF

def getEF(area, area_buffer):
    area_buffer.append(area)  # Automatically removes the oldest element if full
    currMax =  max(area_buffer)
    currMin =  min(area_buffer)
    
     # Prevent division by zero
    currEF = (currMax - currMin) / currMax if currMax != 0 else 0  
    return (currEF, area_buffer)  # Return EF and updated buffer


# Load the video
# "0X1F3A4CF3F7A776F2"
filename = "0X2CBCF722CBB5B80D"
video_path = f'../Videos/{filename}.avi'
cap = cv2.VideoCapture(video_path)

# Check if video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties for saving output video
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ensure the output directory exists
output_dir = "saved_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = "saved_csv_files"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create a VideoWriter object to save the output video
out = cv2.VideoWriter(f"saved_videos/{filename}_segmented.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# Read and process each frame
frame_number = 0
areas = [0, 0] # currMaxArea, currMinArea
maxEF = 0
prevEF = 0
currEF = 0.00001
i = 0
period = 40
area_buffer = deque(maxlen=period)  # Fixed-size buffer
# area_buffer = getEF(10, area_buffer)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Segment the current frame
    segmented_mask = segment_frame(frame)
    # Example usage in the loop:
    area = np.sum(segmented_mask)
    
    # EF = max(maxEF,getEF(area))
    # maxEF = EF

    EF,area_buffer = getEF(area, area_buffer) # getEF(area)
    write_area_to_csv(frame_number,segmented_mask,f"saved_csv_files/{filename}_area.csv", additional_data=[EF], header_names=["Frame","EF","Area"])
    # write_area_to_csv(frame_number,segmented_mask,f"saved_csv_files/{filename}_area.csv")
    frame_number += 1

    # Define the color for the mask (e.g., red)
    mask_color = [255, 0, 255]  # purple in BGR

    # Convert the binary mask (0, 1) to a colored mask (e.g., red for object)
    mask_rgb = np.stack([segmented_mask] * 3, axis=-1)  # Create a 3-channel mask
    mask_rgb = mask_rgb * np.array(mask_color)  # Apply the color to the mask

    # Convert the mask back to the desired color (e.g., red)
    output_frame = cv2.addWeighted(frame, 1, mask_rgb.astype(np.uint8), 0.5, 0)

    # Write the frame to the output video
    out.write(output_frame)

# Release the video capture and writer objects
cap.release()
out.release()

print("Video processing complete!")
