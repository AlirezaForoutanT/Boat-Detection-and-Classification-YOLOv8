from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import piexif
import matplotlib.pyplot as plt
import os

# Get all image files in the directory
image_directory = 'Mapping Lerins/2023 09 ZONE 2'
image_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

model_path = 'Mooring_Project/code/runs/detect/train/weights/last.pt'
mob_counts=mb_counts=sb_counts = 0

boat_areas = []

def load_image(image_path):
    # Loading the image to get its size
    image = Image.open(image_path)
    img_array = np.array(image)

    # Extract focal length and altitude from EXIF metadata
    imgsize = image.size
    exif_data = piexif.load(image.info["exif"])
    focal_length = exif_data["Exif"][piexif.ExifIFD.FocalLength][0] / exif_data["Exif"][piexif.ExifIFD.FocalLength][1]
    altitude = exif_data["GPS"][piexif.GPSIFD.GPSAltitude][0] / exif_data["GPS"][piexif.GPSIFD.GPSAltitude][1]
    sensor_size = (13.3, 8.8)  # in millimeters

    return img_array, imgsize, focal_length, altitude, sensor_size


def adjust_parameters_for_brightness(gray_roi):
    # Calculate average brightness
    brightness = np.mean(gray_roi)
    print(brightness)
    
    # Adjust parameters based on brightness
    if  120 < brightness< 190:  
        kernel_size = (5, 5)  # Larger kernel for brighter regions
        threshold_val = 170  # Higher threshold for bright areas
    elif brightness <= 120:
        kernel_size = (3, 3)  
        threshold_val = 100  
    else:
        kernel_size = (2, 2)  
        threshold_val = 220  

    return kernel_size, threshold_val


def Morphology(x_min, y_min, x_max, y_max, img):

    roi = img[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Adjust morphology parameters based on brightness
    kernel_size, threshold = adjust_parameters_for_brightness(gray_roi)
    kernel = np.ones(kernel_size, np.uint8)
    # kernel = np.ones((5,5), np.uint8)

    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 1)

    opening = cv2.morphologyEx(blurred_roi, cv2.MORPH_OPEN, kernel)

    # morphological operations 
    dilated_roi = cv2.dilate(opening, kernel, iterations=3)
    eroded_roi = cv2.erode(dilated_roi, kernel, iterations=3)

    # Threshold the HSV image 
    mask = cv2.inRange(eroded_roi, threshold , 255)


    # # Display the resulting image
    # cv2.imshow('opening', opening)
    # cv2.waitKey(0)
    # # Display the resulting image
    # cv2.imshow('eroded_roi', eroded_roi)
    # cv2.waitKey(0)
    # # Display the resulting image
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    return mask


def Area_pixels(yolo_coordinates, imgsize, altitude, sensor_size, focal_length, img):
    x_min, y_min, x_max, y_max = map(int, yolo_coordinates)
    
    res =  Morphology(x_min, y_min, x_max, y_max, img)

    # Calculating boat area
    boat_area_pixel = np.count_nonzero(res)  

    gsd = (sensor_size[0] * altitude) / (focal_length * imgsize[0])
   
    # Convert pixel coordinates to real-world coordinates
    eara = boat_area_pixel * gsd*gsd
    if eara>270: print("*****************************************************************************************************************************************",eara)

    return int(eara)


def Area_boxes(yolo_coordinates, imgsize, altitude, sensor_size, focal_length):
    x_min, y_min, x_max, y_max = map(int, yolo_coordinates)

    # Access individual elements of the sensor_size tuple
    sensor_width, sensor_height = sensor_size

    # Calculate Ground Sample Distance (GSD)
    gsd_x = (sensor_width * altitude) / (focal_length * imgsize[0])
    gsd_y = (sensor_height * altitude) / (focal_length * imgsize[1])

    # Convert pixel coordinates to real-world coordinates
    real_width = (x_max - x_min) * gsd_x
    real_height = (y_max - y_min) * gsd_y

    # Calculate real area in square meters
    real_area = real_width * real_height

    return int(real_area)

def plots():

    # Calculate the boat counts for each class
    boat_counts = [len(class_averages[i]) for i in range(len(class_intervals) - 1)]

    # Calculate the total number of boats
    total_boats = sum(boat_counts)

    # Create a subplot with 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the boat counts per class in the first subplot
    axs[0].bar(range(len(class_intervals) - 1), boat_counts)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Number of Boats')
    axs[0].set_title(f'Total Number of Boats: {total_boats}')

    # Set x-axis labels based on size categories and numerical intervals
    x_labels = [f'{class_intervals[i]} - {class_intervals[i + 1]}' for i in range(len(class_intervals) - 1)]
    axs[0].set_xticks(range(len(class_intervals) - 1))
    axs[0].set_xticklabels(x_labels, ha='center', fontsize=9)

    axs[0].set_ylim([0, max(boat_counts) * 1.1])

    # Display boat counts above each bar
    for i, count in enumerate(boat_counts):
        axs[0].text(i, count + max(boat_counts) * 0.05, str(count), ha='center', va='top')

    # Plot the second bar chart in the second subplot
    axs[1].bar(['Moving Boats', 'Motor Boats', 'Sailing Boats'], [mob_counts, mb_counts, sb_counts])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Count')
    axs[1].set_title('')

    # Display the number of boats above each bar in the second subplot
    for i, count in enumerate([mob_counts, mb_counts, sb_counts]):
        axs[1].text(i, count + max([mob_counts, mb_counts, sb_counts]) * 0.05, str(count), ha='center', va='top')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

    areas=[]
    for i in boat_areas:
        if i<300:
            areas.append(i)

    #Plot histogram of boat sizes
    plt.hist(areas, bins=100,color='blue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Boat Sizes')
    plt.xlabel('Boat Size (Area in pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return 


def set_thresholds(boat_areas):

    # Automatically set class intervals and thresholds based on the distribution
    percentile_25 = np.percentile(boat_areas, 25)
    percentile_50 = np.percentile(boat_areas, 50)
    percentile_75 = np.percentile(boat_areas, 75)

    class_intervals = [0, int(percentile_25), int(percentile_50), int(percentile_75), np.inf]
    print("Automatically set class intervals based on percentiles:")
    print(f"Class 1: 0 - {int(percentile_25)} ")
    print(f"Class 2: {int(percentile_25)} - {int(percentile_50)} ")
    print(f"Class 3: {int(percentile_50)} - {int(percentile_75)} ")
    print(f"Class 4: {int(percentile_75)}+ \n")

    with open('class_intervals.txt', 'w') as file:
        for i, interval in enumerate(class_intervals[:-1]):
            file.write(f"Class {i + 1}: {interval} - {class_intervals[i + 1]}\n")
            
    return class_intervals


method=input('Analysis using:\n1) Pixels \n2) boxes\n   Your chois: ')
if(method !='1' and method !='2'):exit() 

for image_path in image_paths:
    img_array, imgsize, focal_length, altitude,sensor_size = load_image(image_path)
    
    # Load the model
    model = YOLO(model_path)
    results = model(image_path, conf=0.4, save=False, save_txt=False) 
    res = list(results)[0]

    xyxy = res.boxes.xyxy
    if(method =='1'):
        for box in xyxy:
            area = Area_pixels(box, imgsize, altitude, sensor_size, focal_length, img_array)
            boat_areas.append(area) 
    
    elif(method =='2'):
        for box in xyxy:
            area = Area_boxes(box, imgsize, altitude, sensor_size, focal_length)
            boat_areas.append(area) 
    
    for r in results:       
        for box in r.boxes:
            class_id = box.cls
            if class_id==0:mb_counts+=1
            elif class_id==2:sb_counts+=1
            else: mob_counts+=1 

            
class_intervals = set_thresholds(boat_areas)
class_averages = [[] for _ in range(len(class_intervals) - 1)]

for j in range(len(boat_areas)):
    for i in range(len(class_intervals) - 1):
        if class_intervals[i] <= boat_areas[j] < class_intervals[i + 1]:
            class_averages[i].append(boat_areas)
            break     

plots()
    
