import numpy as np
import pylas
import statistics as stat
import os

# Define the minimum and maximum RGB values for filtering
rgb_min = np.array([0.99, 0.99, 0.99])  # similar to white
rgb_max = np.array([1.0, 1.0, 1.0])  # white


def crop_las_file(point_cloud, projected_reference_center,reference_box,class_name,z_max,z_min,new_directory):
    
    X_drone, Y_drone, Z_drone = [], [], []
    
    for index,coordinates in enumerate(projected_reference_center):
        
        las = pylas.read(point_cloud)
        points = np.vstack((las.x, las.y, las.z)).T
        
        
        x_half_range = reference_box[index][0] / 2
        y_half_range = reference_box[index][1] / 2
        
        # Define the minimum and maximum bounds for the bounding box
        min_bounds = np.array([coordinates[0] - x_half_range, coordinates[1] - y_half_range, z_min])
        max_bounds = np.array([coordinates[0] + x_half_range,coordinates[1] + y_half_range, z_max])
    
        
        # Find the points that fall within the bounding box
        mask = np.all((min_bounds <= points) & (points <= max_bounds), axis=1)
        
         # Apply the mask to keep only the points within the bounding box
        las.points = las.points[mask]
        
        if class_name != "House":
            
            # Doing color-based point cloud segmentation, extract the white points
            colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            valid_indices = np.where((colors >= rgb_min) & (colors <= rgb_max))[0]
            las.points = las.points[valid_indices]
            
            X_copy = las.X
            Y_copy = las.Y
            Z_copy = las.Z
            
            if X_copy.size == 0:
                print(f"No points found in bounding box for index {index}. Skipping...and class name is {class_name}")
                continue
            
            X_range_copy = max(X_copy) - min(X_copy)
            Y_range_copy = max(Y_copy) - min(Y_copy)

            #Outlier Removal
            points = np.column_stack((X_copy, Y_copy, Z_copy))
            z_mean = stat.mean(Z_copy)
            z_standard_deviation = stat.stdev(Z_copy)
            valid_indices = np.where((Z_copy <= z_mean + 2*z_standard_deviation))[0]
            
            las.points = las.points[valid_indices]
            
                
            if class_name == "Drone":    
                
                #Calculate the center point of the drone(After the segmentation)
                center_copy = np.array([min(X_copy) + X_range_copy / 2, max(Y_copy) - Y_range_copy / 2])
                
                las = pylas.read(point_cloud)
                X = las.X
                Y = las.Y
                
                # Calculate the squared Euclidean distance from each point in the point cloud to the previously computed center.
                dist = (X - center_copy[0]) ** 2 + (Y - center_copy[1]) ** 2
                
                # Find the indices of the 500 points that are closest to the center.
                valid_index = np.argpartition(dist, 500)[:500]
                
                # Update the point cloud by keeping only the 500 points that are closest to the center.
                las.points = las.points[valid_index]
            
                X_drone.append(las.X)
                Y_drone.append(las.Y)
                Z_drone.append(las.Z)
                
                
            
            
        # Write the cropped LAS file
        output_file = os.path.join(new_directory, f"cropped_output_{class_name}_{index}.las")
        las.write(output_file)
        
    if len(X_drone) > 0:
        
        # Concatenate all coordinates
        X = np.concatenate(X_drone)
        Y = np.concatenate(Y_drone)
        Z = np.concatenate(Z_drone)

        # Create a new LAS object to store the merged point cloud
        merged_las = pylas.create()

        # Assign the merged coordinates to the new LAS object
        merged_las.X = X
        merged_las.Y = Y
        merged_las.Z = Z
        
        output_file = os.path.join(new_directory, f"cropped_output_{class_name}_droneMerge.las")
        merged_las.write(output_file)

    
def crop_coordinate(xy_coordinates, wh_dimensions, image_size, class_name,  
                    point_cloud, new_directory, photo_projected, photo_bottom_left,
                    box_bottom_left, z_max, z_min):

    # Convert lists of x,y coordinates, width-height dimensions, and image size to NumPy arrays for efficient computation
    xy_coordinates = np.array(xy_coordinates) 
    wh_dimensions = np.array(wh_dimensions)
    photo = np.array(image_size)

    # Precompute
    support, local_min_corner = divmod(box_bottom_left, 1000)
    
    # Calculate the difference between the x coordinates of the photo's bottom left corner and the box's bottom left corner
    left_diff = photo_bottom_left[0] - box_bottom_left[0]
    
    # Calculate the difference between the y coordinates of the photo's bottom left corner and the box's bottom left corner
    bottom_diff = photo_bottom_left[1] - box_bottom_left[1]

    # Calculate the projected x,y coordinates of the center corresponding to the bounding box of reference object in the point cloud.
    projected_x = xy_coordinates[:,0] / photo[0] * photo_projected[0] + left_diff
    projected_y = photo_projected[1] - xy_coordinates[:,1] / photo[1] * photo_projected[1] + bottom_diff
    projected_reference_center = np.column_stack([projected_x, projected_y]) + local_min_corner + support * 1000

    # Calculate the height and width of the bounding box when projected on the point cloud
    reference_box = wh_dimensions / photo * photo_projected
    
    

    # Print output
    for i in range(len(projected_reference_center)):
        print(f"Point {i+1} coordinate:")
        print(f"X: {projected_reference_center[i][0]}, Range: {reference_box[i][0]}") 
        print(f"Y: {projected_reference_center[i][1]}, Range: {reference_box[i][1]}")

    crop_las_file(point_cloud, projected_reference_center, reference_box,  
                  class_name, z_max, z_min, new_directory)
    


    