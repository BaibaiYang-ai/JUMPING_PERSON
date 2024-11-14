import cv2
from PIL import Image
from ultralytics import YOLO
import os

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

for i_video in sorted_video_files:

    url= os.path.join(video_path, i_video)
    results = model.predict(source=url, save=True)  # save plotted images
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







    for i_result in range(length_results):
        i_xywh = results[i_result].boxes.xywh
        i_conf = results[i_result].boxes.conf # Boxes object for bounding box outputs
        classes = results[i_result].boxes.cls # classes
        indices = ((classes == 0) & (i_conf > 0.5)).nonzero(as_tuple=True)[0] 
        xywh_person = i_xywh[indices]
        # print(i_xywh)
        # print(classes)
        try:
            # print(xywh_person[0])
            if xywh_person[0][2]/xywh_person[0][3] > 1.1:
                print(f"{GREEN}Find the jumping person{RESET}")
                new_path = os.path.join(saved_folder, f"jump{os.path.splitext(i_video)[0]}{i_result}.jpg")
                results[i_result].save(filename=new_path)

        except IndexError:
            print("No person in this frame")


    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk








# Define the folder containing your images
image_folder = os.path.join(current_dir, "jumping_people")  # Update this with your folder path
output_video = 'generated.mp4'  # Output video filename
fps = 15  # Frames per second in the output video
video_resolution = (1080, 1920)  # Desired resolution (width, height)

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



















# for i_result in range(length_results):
#     boxes = results[i_result].boxes  # Boxes object for bounding box outputs
#     # masks = result.masks  # Masks object for segmentation masks outputs
#     # keypoints = result.keypoints  # Keypoints object for pose outputs
#     # probs = result.probs  # Probs object for classification outputs
#     # obb = result.obb  # Oriented boxes object for OBB outputs
#     print(boxes.xywh)
#     print(boxes.boxes.cls)
#     # result.show()  # display to screen
#     # result.save(filename="result.jpg")  # save to disk

# from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])




# from ultralytics import YOLO
# # Load a model
# model = YOLO("PATH TO MODEL")
# # Run batched inference on a list of images
# results = model(["im1.jpg", "im2.jpg"])  # return a list of Results objects
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk




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

