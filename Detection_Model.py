# This script detects objects in an image using a YOLOv5 model from Roboflow, 
# crops the point cloud based on the image boundaries and object locations,

import os
from roboflow import Roboflow
from PIL import Image
import numpy as np
import Crop_Coordinate


# Increase the maximum image size
Image.MAX_IMAGE_PIXELS = None

def extract_coordinates_and_dimensions_and_class(json_data):# Organize predictions by class into class_data dict

    predictions_by_class = {}
    image_width = int(json_data["image"]["width"])  # Extract image width from JSON data
    image_height = int(json_data["image"]["height"])  # Extract image height from JSON data
    predictions_by_class = {}

    # Iterate through the predictions and store them by class
    for prediction in json_data['predictions']:
        class_name = prediction["class"]
        
        if class_name not in predictions_by_class:
            predictions_by_class[class_name] = []

        predictions_by_class[class_name].append(prediction)

    # Create a dictionary for each class with two arrays: one for the (x, y) coordinates and another for the (width, height) dimensions
    class_data = {}

    for class_name, class_predictions in predictions_by_class.items():
        coordinates = []
        dimensions = []

        for prediction in class_predictions:
            x = prediction["x"]
            y = prediction["y"]
            width = prediction["width"]
            height = prediction["height"]

            coordinates.append((x, y))
            dimensions.append((width, height))

        class_data[class_name] = {"coordinates": coordinates, "dimensions": dimensions}

    # Store the image width and height as a tuple
    image_dimensions = (image_width, image_height)
    return image_dimensions, class_data

def crop_center(img, percent):
    
    img_width, img_height = img.size
    image_original_size = (img_width,img_height) #Get the size of the original image
    print("Original size:", img_width, "x", img_height)

    #get the percent% from the central of the image
    left = img_width * ((1 - percent) / 2)
    upper = img_height * ((1 - percent) / 2)
    right = img_width * ((1 + percent) / 2)
    lower = img_height * ((1 + percent) / 2)
    
    img_cropped = img.crop((left, upper, right, lower))
    
    #Decompress the photo 
    new_width = 1250
    new_height = 1250
    print("New size:", new_width, "x", new_height)
    
    resized_image = img_cropped.resize((new_width, new_height))
    
    # Save the resized image
    #resized_image.save("resized_image.tif")
    
    #Save as jpg image
    img_rgb = resized_image.convert('RGB')
    img_rgb.save("resized_image.jpg", 'JPEG')

    
    
    return image_original_size

def receive_file(file_paths):
    global tfw_file_path
    tfw_file_path = file_paths[2]    # Boundary of the photo 
    
    global point_cloud
    point_cloud = file_paths[0]      # Point Cloud FIle
    
    global xyz_file_name
    xyz_file_name = file_paths[3]    # points coordinates on the point cloud
   
    global percent #Image cropping percent
    
    img = Image.open(file_paths[1])
    
   # Set the default percent
    percent = 0.6

    # Check the image dimensions and adjust the percent accordingly
    if img.size[0] < 13000 and img.size[1] < 13000:
        percent = 0.80
    elif img.size[0] < 12000 and img.size[1] < 12000:
        percent = 0.85
        
    image_original_size = crop_center(img, percent)
    
    #Yolo Object Detection Model
    rf = Roboflow(api_key="0CgMo7f3BUDqqEW2mxwo")
    project = rf.workspace().project("plot-feature-detection-version_2")
    model = project.version(2).model

    # infer on a local image
    json_data = model.predict("resized_image.jpg", confidence=40, overlap=30).json()

    # Analyze the return json file and extract the key information
    image_dimensions, class_data = extract_coordinates_and_dimensions_and_class(json_data)
    
    return class_data, image_dimensions, image_original_size


def read_tfw_file(tfw_file_path):  #xyz file
    with open(tfw_file_path,'r') as twf_file:
        tfw_lines = twf_file.readlines()
        
    # Remove any leading or trailing whitespace from each line
    # Convert each line to a floating point number
    tfw_values = [float(line.strip()) for line in tfw_lines]
    
    return tfw_values

def box_coordinate_extract(xyz_file_name): #tfw file
    # Open the file in read mode
    with open(xyz_file_name, "r") as file:
        # Load the data from the file, assuming it is delimited by commas
        data = np.loadtxt(file, delimiter=",")

    # Extract the first column of the data (x coordinates)
    x = data[:, 0]

    # Extract the second column of the data (y coordinates)
    y = data[:, 1]

    # Extract the third column of the data (z coordinates)
    z = data[:, 2]
        
    # Find the maximum value in the z coordinates
    z_max = max(z)

    # Find the minimum value in the z coordinates
    z_min = min(z)
        
    # Find the minimum x and y coordinates to determine the bottom left of the bounding box
    box_bottom_left = np.array([np.min(x), np.min(y)])
        
    # Return the bottom left coordinates of the bounding box, and the max and min z coordinates
    return box_bottom_left, z_max, z_min


def cropping(class_data,image_dimensions,image_original_size):
    
    '''# Read the LAS file
    las = pylas.read(point_cloud)

    # Extract Z coordinates
    Z_copy = las.Z

    # Compute the mean of Z coordinates
    z_mean = np.mean(Z_copy)
    
    # Compute the standard deviation of Z coordinates
    z_standard_deviation = np.std(Z_copy)

    # Identify indices of points where Z coordinates are within the range of (mean - 3*std_dev) and (mean + 2*std_dev)
    valid_indices = np.where((Z_copy <= z_mean + 2*z_standard_deviation) & (Z_copy >= z_mean - 3*z_standard_deviation))[0]

    # Update the point cloud by keeping only the points that fall within the specified range
    las.points = las.points[valid_indices]'''
    
    # Define the base directory name
    directory="PointCloud"
    
   # Use a timestamp to create a unique directory name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    new_directory = f"{directory}_{timestamp}"

    os.makedirs(new_directory)
    
    #output_file = os.path.join(new_directory, f"PointCloud_clean.las")
    #las.write(output_file)
    #del las

    
    tfw_values = read_tfw_file(tfw_file_path)
    
    # Calculate the size of the photo when projected onto the map. This is done by multiplying 
    # the original image size by the scale factors (tfw_values[0] for x, -tfw_values[3] for y)
    photo_projected_old = np.array([tfw_values[0] * image_original_size[0], -tfw_values[3] * image_original_size[1]])
    
    photo_projected = photo_projected_old * percent
    
    # Extract the bottom left coordinate, maximum z and minimum z from the XYZ file
    box_bottom_left, z_max, z_min = box_coordinate_extract(xyz_file_name) #xyz File
    
    # Calculate the bottom left coordinate of the photo.
    # This is done by shifting the original bottom left coordinates (tfw_values[4] for x, tfw_values[5] for y)
    # by a certain percentage of the photo's projected size.
    photo_bottom_left = np.array([tfw_values[4] + (1 - percent) / 2 * photo_projected_old[0] , 
                                  tfw_values[5]-photo_projected_old[1] + (1 - percent) / 2 * photo_projected_old[1]])

    
    # Print the data
    print("Class data:")
    for class_name, data in class_data.items():
        print(f"{class_name}:")
        print(f"  Coordinates: {data['coordinates']}")
        print(f"  Dimensions: {data['dimensions']}")
        
        Crop_Coordinate.crop_coordinate(data['coordinates'],data['dimensions'],image_dimensions,
                                        class_name,point_cloud,new_directory,
                                        photo_projected,photo_bottom_left,box_bottom_left,z_max, z_min)

    print("\nImage Size:", image_dimensions)