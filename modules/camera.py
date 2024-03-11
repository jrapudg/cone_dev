import  numpy as np
import cv2
import carla

####################################################
#    Camera projection
####################################################

def project_vertices_to_image(vertices, edges,  img, K, world_2_camera):
    for verts in vertices:
        for edge in edges:
            p1 = get_image_point(verts[edge[0]], K, world_2_camera)
            p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
            cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)
    return img


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]   

####################################################
#    Process semantic
####################################################

def process_semantic_image(image):
    """
    Convert CARLA semantic segmentation image to a 2D array where each pixel value represents a semantic tag.
    """
    # Convert the raw data into a numpy array and reshape it to image dimensions
    image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_data = np.reshape(image_data, (image.height, image.width, 4))  # images are BGRA
    # The red channel contains the semantic information
    semantic_image = image_data[:, :, 2]
    return semantic_image

def generate_custom_mask(image, label_mappings):
    """
    Generate a mask with custom pixel values for different categories.

    Args:
    - image: numpy array of the semantic segmentation image (labels).
    - label_mappings: Dictionary mapping original label values to new custom values.

    Returns:
    - A numpy array with the custom mask applied.
    """
    # Initialize a mask with zeros (background)
    custom_mask = np.zeros(image.shape, dtype=np.uint8)

    # Apply the mappings to set custom values for each category
    for original_label, new_value in label_mappings.items():
        custom_mask[image == original_label] = new_value

    return custom_mask

def process_semantic_labels(labels):
    """
    Convert CARLA image to numpy array and generate a mask with custom values for categories.
    """
    # Define mappings from original labels to new values
    label_mappings = {
        1: 1,  # Road
        24: 2,  # Road Markings (if applicable)
        14: 3, # Vehicles
        21: 4,  # Construction Zone
        6: 5 # Pole
        #8: 5 # TrafficSign
        #11: 5 # TrafficLight
        # Add more mappings as needed
    }

    # Generate the custom mask
    custom_mask = generate_custom_mask(labels, label_mappings)
    return custom_mask

def get_drivable_surface(labels, include_construction=False):
    """
    Convert CARLA image to numpy array and generate a mask with custom values for categories.
    """
    # Define mappings from original labels to new values
    if include_construction:
        label_mappings = {
            1: 1,  # Road
            24: 1,  # Road Markings (if applicable)
            21: 1  # Construction Zone
            # Add more mappings as needed
        }
    else:
        label_mappings = {
            1: 1,  # Road
            24: 1  # Road Markings (if applicable)
            # Add more mappings as needed
        }

    # Generate the custom mask
    custom_mask = generate_custom_mask(labels, label_mappings)
    return custom_mask

def recolor_custom_mask(custom_mask, value_to_color):
    """
    Recolor a custom mask with specific RGB colors.

    Args:
    - custom_mask: numpy array of the custom mask with unique values for different categories.
    - value_to_color: Dictionary mapping mask values to RGB colors.

    Returns:
    - An RGB numpy array representing the colored image.
    """
    # Initialize an RGB image with the same dimensions as the custom_mask, filled with black
    rgb_image = np.zeros((custom_mask.shape[0], custom_mask.shape[1], 3), dtype=np.uint8)

    # Apply the color mappings
    for value, color in value_to_color.items():
        rgb_image[custom_mask == value] = color

    return rgb_image

def visualize_custom_mask(custom_mask):
    """
    Visualize a custom mask by applying predefined RGB colors to each category, with the construction zone in orange.
    """
    # Define the RGB color mappings for each category, with the construction zone set to orange ([255, 165, 0])
    value_to_color = {
        0: [0, 0, 0],       # Background: Black
        1: [128, 128, 128], # Road: Gray
        2: [255, 255, 255], # Road Markings: White
        3: [0, 0, 255],     # Vehicles: Blue
        4: [255, 165, 0],    # Construction Zone: Orange
        5: [255, 0, 0]    # Construction Zone: Red
        # Add more mappings as needed
    }

    # Recolor the custom mask to an RGB image
    colored_mask = recolor_custom_mask(custom_mask, value_to_color)
    return colored_mask

def project_trajectory_to_image(traj, img, K, world_2_camera):
    in_drivable = 0
    width = img.shape[1]
    height = img.shape[0]
    
    for i in range(traj.shape[0]-1):
        p1_3d = carla.Location(x=traj[i][0], y=traj[i][1], z=0)
        p2_3d = carla.Location(x=traj[i+1][0], y=traj[i+1][1], z=0)
        p1 = get_image_point(p1_3d, K, world_2_camera)
        p2 = get_image_point(p2_3d,  K, world_2_camera)
        
        if np.all(p1 > 0) and np.all(p2 > 0):
            u1, v1 = (int(p1[0]),int(p1[1]))
            u2, v2 = (int(p2[0]),int(p2[1]))
            if u1 < width and v1 < height and u2 < width and v2 < height:
                cv2.line(img, (u1, v1), (u2, v2), (255,0,0, 255), 1)
    return img

def count_points_in_mask(traj, img, K, world_2_camera):
    in_drivable = 0
    width = img.shape[1]
    height = img.shape[0]
    iters = traj.shape[0]-1
    for i in range(iters):
        p1_3d = carla.Location(x=traj[i][0], y=traj[i][1], z=0)
        p2_3d = carla.Location(x=traj[i+1][0], y=traj[i+1][1], z=0)
        p1 = get_image_point(p1_3d, K, world_2_camera)
        p2 = get_image_point(p2_3d,  K, world_2_camera)
        
        if np.all(p1 > 0) and np.all(p2 > 0):
            u1, v1 = (int(p1[0]),int(p1[1]))
            u2, v2 = (int(p2[0]),int(p2[1]))
            
            if u1 < width and v1 < height and u2 < width and v2 < height:
                in_drivable += img[v2, u2] > 0 
    return np.mean(in_drivable)

def check_endpoint_in_mask(traj, img, K, world_2_camera):
    in_drivable = 0
    width = img.shape[1]
    height = img.shape[0]
    last = traj.shape[0]-1
    
    p1_3d = carla.Location(x=traj[last][0], y=traj[last][1], z=0)
    p1 = get_image_point(p1_3d, K, world_2_camera)
        
    if np.all(p1 > 0):
        u1, v1 = (int(p1[0]),int(p1[1]))
            
        if u1 < width and v1 < height:
            return img[v1, u1] > 0
        
    return False