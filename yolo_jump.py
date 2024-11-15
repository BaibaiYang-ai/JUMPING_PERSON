import cv2
from PIL import Image
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.signal import find_peaks
import subprocess
import math


import subprocess
import os
import math
import re


# 0, split video, in case the video is too long to be processed by Yolo

def split_video(input_video, video_path, split_duration=90):
    # Get video duration in seconds
    s_video = os.path.join(video_path, input_video)
    result = subprocess.run(['ffmpeg', '-i', s_video], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    
    # Extract duration in seconds from the ffmpeg output
    match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', output)
    
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        total_duration = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Could not parse the video duration.")
    
    # Calculate the number of segments based on the desired split duration
    num_segments = math.ceil(total_duration / split_duration)

    

    # Create output directory
    output_dir = "videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the video into smaller parts
    if num_segments >1:
        for i in range(num_segments):
            start_time = i * split_duration
            output_filename = os.path.join(output_dir, f"{os.path.splitext(i_video)[0]}_{i:02d}.mp4")
            
            
            # Run ffmpeg to extract each part
            subprocess.run([
                'ffmpeg', '-i', s_video, '-ss', str(start_time), '-t', str(split_duration),
                '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', output_filename
            ])

        try:
            # Check if the file exists
            if os.path.exists(s_video):
                # Remove the file
                os.remove(s_video)
                print(f"File {s_video} has been deleted successfully.")
            else:
                print(f"The file {s_video} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")


    else:
        output_filename = os.path.join(output_dir, f"{os.path.splitext(i_video)[0]}_00.mp4")
        try:
            # Check if the old file exists
            if os.path.exists(s_video):
                # Rename the file
                os.rename(s_video, output_filename)
                print(f"File renamed successfully to {output_filename}")
            else:
                print(f"The file {s_video} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
            
        print(f"Created: {output_filename}")
        
    print("Video split completed.")
    # return num_segments







#### 1. Yolo detect

model = YOLO("yolo11n.pt")
# ANSI escape codes for colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'  # Reset color to default

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("https://youtu.be/GJyD4fnR6HQ?si=KwFvCHPceoAdW5Pd")
current_dir = os.getcwd()


video_folder = "videos"
video_path = os.path.join(current_dir, video_folder)

all_files = os.listdir(video_path)



# Define common video file extensions
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']

# Filter the files that have a video extension
video_files = [file for file in all_files if file.lower().endswith(tuple(video_extensions))]

# Sort video files alphabetically (you can also use other sorting criteria)
sorted_video_files = sorted(video_files)

# for i_video in sorted_video_files:
#     input_video = i_video  # Replace with your video file path
#     split_video(input_video, video_path, split_duration=90)  # 90 seconds = 1.5 minutes



# all_files = os.listdir(video_path)

# # Filter the files that have a video extension
# video_files = [file for file in all_files if file.lower().endswith(tuple(video_extensions))]

# # Sort video files alphabetically (you can also use other sorting criteria)
# sorted_video_files = sorted(video_files)



for i_video in sorted_video_files:

    url= os.path.join(video_path, i_video)
    results = []

    results = model.predict(source=url, save=False)  # save plotted images
    saved_folder = "jumping_people"
    length_results = len(results)

    saved_folder = os.path.join(current_dir, saved_folder)

    # Check if the file exists
    if not os.path.exists(saved_folder):
        # Create the file if it does not exist
        os.makedirs(saved_folder)
        print(f"File '{saved_folder}' created.")
    else:
        print(f"File '{saved_folder}' already exists.")



    w_list = []
    h_list = []
    i_w = []
    i_h = []
    


    for i_result in range(length_results):
        i_w = [i_result, 0]
        i_h = [i_result, 0]
        i_xywh = results[i_result].boxes.xywh
        i_conf = results[i_result].boxes.conf # Boxes object for bounding box outputs
        classes = results[i_result].boxes.cls # classes
        indices = ((classes == 0) & (i_conf > 0.5)).nonzero(as_tuple=True)[0] 
        xywh_person = i_xywh[indices]
        # print(i_xywh)
        # print(classes)
        try:
            # print(xywh_person[0])
            if xywh_person[0][2]/xywh_person[0][3] > 0.25:
                print(xywh_person[0][2]/xywh_person[0][3])
                print(xywh_person[0][2].cpu().item())
                i_w[1] = xywh_person[0][2].cpu().item()
                i_h[1] = xywh_person[0][3].cpu().item()
                
                print(f"{GREEN}Find the one person{RESET}")
        except IndexError:
            print("No person in this frame")
        
        w_list.append(i_w)
        h_list.append(i_h)

### 2. filtering

    data1 = np.array(h_list)

    # Split the data into x and y arrays
    x_data1 = data1[:, 0]
    y_data1 = data1[:, 1]

    # Filter out rows where y == 0
    mask = y_data1 != 0
    x_data1 = x_data1[mask]
    y_data1 = y_data1[mask]
    x_data1 = x_data1[30:]
    y_data1 = y_data1[30:]

    # Find local minima (lower peaks)
    inverted_y = -y_data1  # Invert y values to find minima as peaks
    minima_indices, _ = find_peaks(inverted_y, distance=30)

    # Filter minima where y is smaller than the left and right 8 points
    filtered_minima_indices = []
    for idx in minima_indices:
        if idx >= 50 and idx <= len(y_data1) - 51:
            left_values_50 = np.mean(y_data1[idx-50:idx])
            right_values_50 = np.mean(y_data1[idx+1:idx+51])
            avg_distance = (left_values_50 + right_values_50) / 2

        # if idx >= 8 and idx <= len(y_data1) - 9:  # Ensure there are enough points on both sides
            left_values = y_data1[idx-8:idx]
            right_values = y_data1[idx+1:idx+9]
            if y_data1[idx] < left_values.min() and y_data1[idx] < right_values.min():
                # Calculate average distance from y_data1[idx] to surrounding points
                avg_distance_left = np.mean(np.abs(y_data1[idx] - left_values))
                avg_distance_right = np.mean(np.abs(y_data1[idx] - right_values))

                if avg_distance_left > avg_distance/7 and avg_distance_right >avg_distance/7:
                    filtered_minima_indices.append(idx)

    lenght_filtered = len(filtered_minima_indices)
    new_filtered_minima_indices = []
    for i_filtered in range(lenght_filtered):
        if filtered_minima_indices[i_filtered] >= 50 and filtered_minima_indices[i_filtered] <= len(y_data1) - 51:
            left_values1_50 = np.mean(y_data1[filtered_minima_indices[i_filtered]-50:filtered_minima_indices[i_filtered]])
            right_values1_50 = np.mean(y_data1[filtered_minima_indices[i_filtered]+1:filtered_minima_indices[i_filtered]+51])
            avg_distance1 = (left_values1_50 + right_values1_50) / 2
        else:
            avg_distance1 = 210

        if i_filtered == 0 and lenght_filtered >1:
            if x_data1[filtered_minima_indices[i_filtered+1]] - x_data1[filtered_minima_indices[i_filtered]] <150 and abs(y_data1[filtered_minima_indices[i_filtered+1]] - y_data1[filtered_minima_indices[i_filtered]]) < avg_distance1/9:
                new_filtered_minima_indices.append(filtered_minima_indices[i_filtered])
        elif i_filtered < lenght_filtered -1 and lenght_filtered >2:
            if x_data1[filtered_minima_indices[i_filtered+1]] - x_data1[filtered_minima_indices[i_filtered]] <150 and abs(y_data1[filtered_minima_indices[i_filtered+1]] - y_data1[filtered_minima_indices[i_filtered]]) < avg_distance1/9:
                new_filtered_minima_indices.append(filtered_minima_indices[i_filtered])

            elif x_data1[filtered_minima_indices[i_filtered]] - x_data1[filtered_minima_indices[i_filtered-1]] <150 and abs(y_data1[filtered_minima_indices[i_filtered]] - y_data1[filtered_minima_indices[i_filtered-1]]) < avg_distance1/9:
                new_filtered_minima_indices.append(filtered_minima_indices[i_filtered])

        elif i_filtered == lenght_filtered -1 and lenght_filtered >1:  
            if x_data1[filtered_minima_indices[i_filtered]] - x_data1[filtered_minima_indices[i_filtered-1]] <150 and abs(y_data1[filtered_minima_indices[i_filtered]] - y_data1[filtered_minima_indices[i_filtered-1]]) < avg_distance1/9:
                new_filtered_minima_indices.append(filtered_minima_indices[i_filtered])
        else:
            print("the point is not selected")


    # Extract the filtered minima values for x and y
    filtered_x_minima = x_data1[new_filtered_minima_indices]
    filtered_y_minima = y_data1[new_filtered_minima_indices]

    # # Plot data with filtered minima
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_data1, y_data1, label='Dataset 1', marker='o')
    # plt.plot(filtered_x_minima, filtered_y_minima, 'ro', label='Filtered Lower Peaks')

    # # Add labels and title
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Filtered Lower Peaks of Dataset 1')
    # plt.legend()
    # plt.grid(True)

    # # Show plot
    # plt.show()

    # Print filtered minima values
    print("Filtered Lower Peaks:")
    for x_val, y_val in zip(filtered_x_minima, filtered_y_minima):
        print(f"x: {x_val}, y: {y_val}")








### 3. pick the images

    # Load video file
    cap = cv2.VideoCapture(url)

    # Define frame selection criteria
    filtered_x_minima = filtered_x_minima.astype(int) # Example list of specific frame numbers to save
    # Extend each element by 5 elements on the left and right
    extended_indices = []
    for idx in filtered_x_minima:
        extended_indices.extend(range(idx - 3, idx + 4))

    # Remove duplicates and sort
    extended_indices = sorted(set(extended_indices))

    selected_frames = extended_indices

    # Create an output folder for extracted images
    output_folder = 'extracted_images'
    output_yolo_folder = 'extracted_images_yolo'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_yolo_folder, exist_ok=True)

    for i_result in range(length_results):
        if i_result in selected_frames:

            new_path = os.path.join(output_yolo_folder, f"yolo_jump{os.path.splitext(i_video)[0]}_{i_result:05d}.jpg")
            results[i_result].save(filename=new_path)



    # Initialize frame count
    frame_count = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if the frame is one of the selected frames
        if frame_count in selected_frames:
            # Save the frame as an image
            output_path = os.path.join(output_folder, f"jump{os.path.splitext(i_video)[0]}_{frame_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            print(f'Saved {output_path}')
        
        frame_count += 1

    # Release video capture object
    cap.release()
    print("Extraction complete.")





"""





"""




### 4. generate video

def generate_video(ge_video_folder, video_name):
    # Define the folder containing your images
    image_folder = os.path.join(current_dir, ge_video_folder)  # Update this with your folder path
    output_video = video_name  # Output video filename
    fps = 30  # Frames per second in the output video
    video_resolution = (1920, 1080)  # Desired resolution (width, height)

    # Get list of all images in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    # Sort the images by name (or you can sort by date using os.path.getmtime if needed)
    image_files.sort()  # You can change this to sort by modification time, e.g. sorted(image_files, key=lambda x: os.path.getmtime(x))

    # Check if there are any images in the folder
    if not image_files:
        print("No images found in the folder.")
        exit()

    # Read the first image to get the dimensions (width and height)
    first_image = Image.open(os.path.join(image_folder, image_files[0]))
    first_image = first_image.resize(video_resolution)  # Resize to desired resolution
    frame_width, frame_height = first_image.size

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Loop through the sorted images, convert each to a frame, and write to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Open the image
        img = cv2.imread(image_path)
        
        # Resize image to match the video resolution (if needed)
        img_resized = cv2.resize(img, (frame_width, frame_height))
        
        # Write the image frame to the video
        video_writer.write(img_resized)

    # Release the video writer object
    video_writer.release()

    print(f"Video created successfully: {output_video}")



generate_video(output_folder, "generate_pure_jump.mp4")
generate_video(output_yolo_folder, "generate_yolo_jump.mp4")
















# Now we will iterate over the loop and then fetch each of the result. Each result contains the following attributes.

# orig_img: This is the original image as an numpy array.
# orig_shape: This tuple contains the frame shape.
# boxes: This is a array of Box objectsof Ultralytics containing bounding box. Each box has following parameters:
    # xyxy: It contains the coordinates according to frame and we are going to use this for this tutorial.
    # conf: It is the confidence value of the bounding box or the detected object.
    # cls: It is the class of object. There are total 80 classes.
    # id: It is the ID of the box.
    # xywh: Returns the bounding box in xywh format.
    # xyxyn: It returns the bounding box in xyxy format but in normalized form that is from 0 to 1.
    # xywhn: It returns the bounding box in xywh format but in normalized form that is from 0 to 1.
# masks: It is the collection of Mask objects for detection mask.
# probs: It contains the probability of each classification.
# keypoints: It is the collection of Keypoints which stores the collection of each keypoint.
# obb: It contains OBB object containing oriented bounding boxes.
# speed: It returns the speed of prediction. Faster computer with GPU gives better speed.
# names: It is the dictionary of classes with names and it is irrespective of prediction. The cls that we get in box can be used here to get the respective name.
# path: It is the path to image file.
