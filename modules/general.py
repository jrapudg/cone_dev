import carla
import numpy as np
import random
import sys 
import math
import time
import datetime
import os
from PIL import Image
from modules.transformations import get_rotation_matrix, rotation_yaw, rotation_yaw_3d, radian2degree
from modules.camera import process_semantic_image
import imageio
sys.path.append('/home/juan/Documents/simulators/carla/CARLA_0.9.14/PythonAPI/carla/')
from agents.navigation.global_route_planner import GlobalRoutePlanner

####################################################
#    General Simulation
####################################################

def initialize_random_town(client):
    # Randomly select a map
    available_maps = ['Town10HD',
                      'Town10HD', # Urban Town
                      '/Game/Carla/Maps/Town05', # Double lane
                      '/Game/Carla/Maps/Town01', # Small town single lane
                      '/Game/Carla/Maps/Town06' # Highway
                      ]
    
    random_map = random.choice(available_maps)
    world = client.load_world(random_map)

    # Set random weather
    weather_options = [carla.WeatherParameters.ClearNoon]
    random_weather = random.choice(weather_options)
    world.set_weather(random_weather)
    return random_map

def destroy_all_actors(world):
    actor_list = world.get_actors()
    # Destroy only cones, barriers, and barrels
    for actor in actor_list:  # Adjust this based on the actual blueprint names in your setup
        if actor.is_alive:
            actor.destroy()

def destroy_all_static_objects(world):
    actor_list = world.get_actors()
    # Destroy only cones, barriers, and barrels
    for actor in actor_list.filter('static.prop.*'):  # Adjust this based on the actual blueprint names in your setup
        if actor.is_alive:
            actor.destroy()

def destroy_all_vehicle_objects(world):
    actor_list = world.get_actors()
    # Destroy only cones, barriers, and barrels
    for actor in actor_list.filter('vehicle.*'):  # Adjust this based on the actual blueprint names in your setup
        if actor.is_alive:
            actor.destroy()

def every_tick(snapshot):
    frame = snapshot.frame
    timestamp = snapshot.timestamp.elapsed_seconds
    print(f"Frame: {frame}, Timestamp: {timestamp}")

####################################################
#    Control and Measurements
####################################################

def get_vehicle_current_pose(vehicle):
    vehicle_transform = vehicle.get_transform()
    location = vehicle_transform.location
    rotation = vehicle_transform.rotation
    return location, rotation

def get_vehicle_measurement(vehicle):
    location, rotation  = get_vehicle_current_pose(vehicle)
    return location.x, location.y, location.z, rotation.yaw*math.pi/180

def get_vehicle_speed(vehicle):
    # Get the vehicle's velocity (in meters/second)
    velocity = vehicle.get_velocity()

    # Calculate the speed using the velocity's magnitude
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
    return speed

def send_control_command(world, vehicle, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, 
                                   hand_brake=hand_brake, reverse=reverse)
    vehicle.apply_control(control)
    world.tick()
    time.sleep(0.001)

####################################################
#    Waypoints
####################################################

def get_route_to_random_location(world, carla_map, ego_vehicle, resolution=1.0, min_pts=70, max_pts=130):
    ego_transform = ego_vehicle.get_transform()
    ego_waypoint = carla_map.get_waypoint(ego_transform.location)
    end_transform = random.choice(world.get_map().get_spawn_points())
    start_waypoint = carla_map.get_waypoint(ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    end_waypoint = carla_map.get_waypoint(end_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    grp = GlobalRoutePlanner(carla_map, resolution)  # The second argument is the hop_resolution for the planner
    route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
    if len(route) > min_pts:
        waypoints = [t[0] for t in route]
        return route, waypoints[0:max_pts]
    else:
        route, waypoints = get_route_to_random_location(carla_map, ego_vehicle, resolution=resolution, min_pts=min_pts)
        return route, waypoints[0:max_pts]

def add_lateral_offset(waypoints, offset, sigma=0.2):
    """
    Adds a lateral offset to a set of waypoints.

    Args:
    waypoints (list of carla.Waypoint): The original waypoints.
    offset (float): The lateral offset distance. Positive values offset to the right, negative to the left.

    Returns:
    list of carla.Location: New waypoints with the applied lateral offset.
    """
    new_waypoints = []
    rotations = []

    for waypoint in waypoints:
        # Get the rotation matrix for the waypoint
        rotation_matrix = get_rotation_matrix(waypoint.transform.rotation)

        # Create lateral offset vector (right vector)
        lateral_offset = np.array([0, offset, 0])  # Offset in the Y direction

        # Rotate the lateral offset vector
        rotated_offset = np.dot(rotation_matrix, lateral_offset)

        # Calculate the new location with rotated offset
        new_location = carla.Location(x=waypoint.transform.location.x + rotated_offset[0] + random.gauss(mu=0, sigma=sigma),
                                      y=waypoint.transform.location.y + rotated_offset[1] + random.gauss(mu=0, sigma=sigma),
                                      z=waypoint.transform.location.z + rotated_offset[2]
                                      )

        new_waypoints.append(new_location)
        rotations.append(rotation_matrix)
    return new_waypoints, rotations

# Function to detect a change in segment
def is_new_segment(wp1, wp2):
    # Detecting change in road_id or lane_id
    return wp1.road_id != wp2.road_id or wp1.lane_id != wp2.lane_id

# Function to split and pad segments
def split_and_pad_segment(segment, max_size, min_size):
    subsegments = [segment[i:i + max_size] for i in range(0, len(segment), max_size)]
    padded_subsegments = []
    for subsegment in subsegments:
        subsegment_array = np.array([(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z, wp.lane_width, 0, 1) for wp in subsegment])
        if len(subsegment) >= min_size:
            if len(subsegment) < max_size:
                padding = np.zeros((max_size - len(subsegment), 6))
                subsegment_array = np.vstack((subsegment_array, padding))
            padded_subsegments.append(subsegment_array)
    return padded_subsegments

def substract_ego_location_2d(points, ego_loc):# Iterate through the array and subtract the point
    points_cp = points.copy()
    for point in points_cp:
        if not np.all(point == [0, 0]):  # Check if the point is not zeros
            point -= ego_loc
    return points_cp

def pad(roads_arr, center_location, thresh=100):
    n = roads_arr.shape[0]
    if n < thresh:
        # Pad the array with zeros
        padded_arr = np.pad(roads_arr, ((0, thresh - n), (0, 0), (0, 0)), mode='constant')
        return padded_arr
    else:
        #print("PAD: Filter by distance")
        closest_points = []

        for segment in roads_arr:
            # Calculate distances of all points in the segment from the center_location
            segment = segment[~np.all(segment == 0, axis=1)]
            distances = np.linalg.norm(segment[:, :3] - center_location[:3], axis=1)
            
            # Find the index of the closest point in this segment
            closest_point_idx = np.argmin(distances)

            # Append the closest point of this segment
            closest_points.append(segment[closest_point_idx, :3])

        # Convert list to array
        closest_points_arr = np.array(closest_points)

        # Sort the closest points by their distance to the center location
        sorted_indices = np.argsort(np.linalg.norm(closest_points_arr - center_location[:3], axis=1))

        # Select the closest segments based on the threshold
        filtered_arr = roads_arr[sorted_indices[:thresh]]
        return filtered_arr

def process_roads(road_pts_np, center_location):
    road_pts_np = road_pts_np.reshape(-1,40,road_pts_np.shape[-1])
    road_pts_np = pad(road_pts_np, center_location, thresh=100)
    return road_pts_np

def process_trajectory_map(traj_map, ego_locations):
    process_traj_map = []
    for traj, center_location in zip(traj_map, ego_locations):
        process_traj_map.append(process_roads(traj, center_location))
    process_traj_map = np.array(process_traj_map)
    return process_traj_map

def generate_map(carla_map, current_x, current_y, area_range=300, distance_spread=1.0, min_points_per_segment=3, max_points_per_segment=100):
    ego_location = carla.Location(x=current_x, y=current_y, z=0)
    ego_waypoint = carla_map.get_waypoint(ego_location)
    ego_points_loc = np.array([ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, ego_waypoint.transform.location.z])
    
    # Define the area range around the ego vehicle
    # 80m x 80m area, so 40m in each direction
    min_x, max_x = ego_location.x - area_range, ego_location.x + area_range
    min_y, max_y = ego_location.y - area_range, ego_location.y + area_range
    
    all_waypoints = carla_map.generate_waypoints(distance=distance_spread)  # Generate waypoints every 1 meter
    
    # Add z parameter
    accurate_waypoints = []
    for waypoint in all_waypoints:
        # Create a new location with an arbitrary non-zero z value
        elevated_location = carla.Location(waypoint.transform.location.x, waypoint.transform.location.y, 0)

        # Project this location onto the road
        road_waypoint = carla_map.get_waypoint(elevated_location, project_to_road=True)

        if road_waypoint:
            accurate_waypoints.append(road_waypoint)
    
    # Filter waypoints by area and sort
    relevant_waypoints = [wp for wp in accurate_waypoints if min_x <= wp.transform.location.x <= max_x and min_y <= wp.transform.location.y <= max_y]
    relevant_waypoints.sort(key=lambda wp: (wp.road_id, wp.lane_id))
    
    # Get segments
    segmented_waypoints = []
    current_segment = []
    prev_wp = None

    for wp in relevant_waypoints:
        if prev_wp is not None and is_new_segment(prev_wp, wp):
            if len(current_segment) >= min_points_per_segment:
                segmented_waypoints += split_and_pad_segment(current_segment, max_points_per_segment, min_points_per_segment)
                current_segment = []
            else:
                current_segment = []
        current_segment.append(wp)
        prev_wp = wp

    # Add and process the last segment if not empty
    if current_segment:
        segmented_waypoints += split_and_pad_segment(current_segment, max_points_per_segment, min_points_per_segment)

    # Convert the list of segments into a NumPy array
    padded_waypoints = np.array(segmented_waypoints)
    # Flaten 3D arrays
    road_segments_global_np = padded_waypoints.reshape(-1, padded_waypoints.shape[-1])
    return road_segments_global_np

####################################################
#    Save Utils
####################################################

def create_simulation_folder(base_path):
    """
    Creates a simulation folder with a random ID.

    Args:
    base_path (str): The base path where the folder will be created.

    Returns:
    str: The path to the created simulation folder.
    """
    # Generate a time-stamped ID for the folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(base_path, f"sim_{timestamp}")
    image_path = os.path.join(folder_path, 'images')
    bev_path = os.path.join(folder_path, 'bev')

    # Create the folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        os.makedirs(image_path)
        os.makedirs(bev_path)

    return folder_path, image_path, bev_path

def create_gif(input_folder, output_file, frame_duration):
    """
    Creates a GIF from a sequence of images in a folder.

    Args:
    input_folder (str): The folder containing input images.
    output_file (str): The path to the output GIF file.
    frame_duration (float): Duration of each frame in the GIF in seconds.
    """
    images = []
    # Retrieve file paths
    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            images.append(imageio.imread(file_path))

    # Create a GIF
    imageio.mimsave(output_file, images, duration=frame_duration)

def substract_ego_location_3d(points, ego_loc):# Iterate through the array and subtract the point
    points_cp = points.copy()
    if  len(points.shape) == 3:
        for segment in points_cp:
            for point in segment:
                #print("3d:{}".format(point.shape))
                if not np.all(point == [0, 0, 0]):  # Check if the point is not zeros
                    point -= ego_loc
    else:
        for point in points_cp:
            if not np.all(point == [0, 0, 0]):  # Check if the point is not zeros
                point -= ego_loc
    return points_cp

def global2local2d(global_array, ego_vector):
    #global_array = global_array.reshape(-1, 2)
    local_array = substract_ego_location_2d(global_array[:,:2], ego_vector[:2])
    local_array = rotation_yaw(local_array, radian2degree(ego_vector[5])-90)
    return local_array

def global2local3d(global_array, ego_vector):
    #global_array = global_array.reshape(-1, 2)
    local_array = substract_ego_location_3d(global_array[:,:3], ego_vector[:3])
    local_array = rotation_yaw_3d(local_array, radian2degree(ego_vector[5])-90)
    return local_array

def substract_ego_location_2d(points, ego_loc):# Iterate through the array and subtract the point
    points_cp = points.copy()
    if  len(points.shape) == 3:
        for segment in points_cp:
            for point in segment:
                if not np.all(point == [0, 0]):  # Check if the point is not zeros
                    point -= ego_loc
    else:
        for point in points_cp:
            if not np.all(point == [0, 0]):  # Check if the point is not zeros
                point -= ego_loc
    return points_cp

"""def substract_ego_location(points, ego_loc):# Iterate through the array and subtract the point
    points_cp = points.copy()
    if  len(points.shape) == 3:
        for segment in points_cp:
            for point in segment:
                if not np.all(point == [0, 0]):  # Check if the point is not zeros
                    point -= ego_loc
    else:
        for point in points_cp:
            if not np.all(point == [0, 0]):  # Check if the point is not zeros
                point -= ego_loc
    return points_cp

def global2local(global_array, ego_vector):
    #global_array = global_array.reshape(-1, 2)
    local_array = substract_ego_location(global_array, ego_vector[:2])
    local_array = rotation_yaw(local_array, radian2degree(ego_vector[2])-90)
    return local_array"""


def global2localall(global_coord, ego_coord, dim=2, yaw_info=True):
    if dim == 2:
        local_coord = global2local2d(global_coord[:,:2], ego_coord)
        if yaw_info:
            local_coord = np.concatenate((local_coord, (global_coord[:,5] - ego_coord[5]).reshape(global_coord.shape[0],1)), axis=1)
        else:
            local_coord = np.concatenate((local_coord, np.full((global_coord.shape[0], 1), 0)), axis=1)
    elif dim == 3:
        global_coord = global_coord.reshape(-1, global_coord.shape[-1])
        local_coord = global2local3d(global_coord[:,:3], ego_coord)
        if yaw_info:
            local_coord = np.concatenate((local_coord, global_coord[:,3:5], (global_coord[:,5] - ego_coord[5]).reshape(global_coord.shape[0],1), global_coord[:,6:]), axis=1)
        else:
            #print(local_coord.shape)
            #print(np.full((global_coord.shape[0], 1), -ego_coord[5]).shape)
            #print("-----")
            local_coord = np.concatenate((local_coord, global_coord[:,3:5], np.full((global_coord.shape[0], 1), 0), global_coord[:,5:]), axis=1)
    else:
        raise NotImplementedError("This function hasn't been implemented yet.")
    return local_coord