import carla
import math
import numpy as np
import cv2

#from modules.general import substract_ego_location

def rotate_actor(actor, degrees):
    current_transform = actor.get_transform()
    current_rotation = current_transform.rotation

    # Modify the yaw angle (rotation around the Z-axis) by adding 90 degrees
    new_yaw = (current_rotation.yaw + degrees) % 360  # Use modulo to keep it within [0, 360)

    # Create a new rotation with the modified yaw angle
    new_rotation = carla.Rotation(pitch=current_rotation.pitch, yaw=new_yaw, roll=current_rotation.roll)

    # Create a new transform with the new rotation
    new_transform = carla.Transform(location=current_transform.location, rotation=new_rotation)

    # Apply the new transform to the actor
    actor.set_transform(new_transform)
    return actor

def get_location_in_front_ego(ego_vehicle, dist, yaw):
    ego_vehicle_transform = ego_vehicle.get_transform()
    
    # Calculate a position 10 meters in front of the spectator based on its orientation
    x = ego_vehicle_transform.location.x + dist * math.cos(ego_vehicle_transform.rotation.yaw * math.pi / 180.0)
    y = ego_vehicle_transform.location.y + dist * math.sin(ego_vehicle_transform.rotation.yaw * math.pi / 180.0)

    # Create a carla.Location object for the calculated position
    calculated_location = carla.Location(x, y, ego_vehicle_transform.location.z)
    return calculated_location

def transform_from_local_to_global(local_point, yaw, ego_vehicle):
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation

    # Define a point in local coordinate system of ego vehicle
    # (2 meters ahead and 1 meter to the left)
    local_x = local_point[0]
    local_y = local_point[1]
    local_z = 0.0  # Assuming the point is at the same height as the vehicle

    # Transform this point to world coordinate system

    # Calculate transformation matrix elements
    cos_yaw = math.cos(ego_rotation.yaw * math.pi / 180)
    sin_yaw = math.sin(ego_rotation.yaw * math.pi / 180)

    # Apply transformation
    world_x = ego_location.x + (local_x * cos_yaw - local_y * sin_yaw)
    world_y = ego_location.y + (local_x * sin_yaw + local_y * cos_yaw)
    world_z = ego_location.z + local_z  # Assuming no pitch or roll
    
    # To get the resulting global rotation
    resulting_yaw = (ego_rotation.yaw + yaw) % 360
    resulting_pitch = ego_rotation.pitch
    resulting_roll = ego_rotation.roll
    return carla.Location(world_x, world_y, world_z), carla.Rotation(pitch=resulting_pitch, yaw=resulting_yaw, roll=resulting_roll)

def calculate_distance(location1, location2):
    """
    Calculate the distance between two carla.Location objects.
    """
    dx = location1.x - location2.x
    dy = location1.y - location2.y
    dz = location1.z - location2.z
    return (dx**2 + dy**2 + dz**2)**0.5

def add_locations(loc1, loc2):
    """
    Adds two carla.Location objects.

    Args:
    loc1 (carla.Location): The first location.
    loc2 (carla.Location): The second location.

    Returns:
    carla.Location: The sum of the two locations.
    """
    x = loc1.x + loc2.x
    y = loc1.y + loc2.y
    z = loc1.z + loc2.z

    return carla.Location(x, y, z)

def get_reverse_rotation(actor):
    """
    Returns the inverse rotation of an actor's current rotation.

    Args:
    actor (carla.Actor): The actor.

    Returns:
    carla.Rotation: The inverse rotation.
    """
    # Get the actor's current rotation
    current_rotation = actor.get_transform().rotation

    # Negate each angle to reverse the rotation
    inverse_rotation = carla.Rotation(-current_rotation.pitch, -current_rotation.yaw, -current_rotation.roll)

    return inverse_rotation

def get_rotated_bounding_box_corners(actor, reverse_rotation):
    bounding_box = actor.bounding_box
    corners = [
        carla.Vector3D(-bounding_box.extent.x, -bounding_box.extent.y, -bounding_box.extent.z),
        carla.Vector3D(bounding_box.extent.x, -bounding_box.extent.y, -bounding_box.extent.z),
        carla.Vector3D(-bounding_box.extent.x, bounding_box.extent.y, -bounding_box.extent.z),
        carla.Vector3D(bounding_box.extent.x, bounding_box.extent.y, -bounding_box.extent.z),
        carla.Vector3D(-bounding_box.extent.x, -bounding_box.extent.y, bounding_box.extent.z),
        carla.Vector3D(bounding_box.extent.x, -bounding_box.extent.y, bounding_box.extent.z),
        carla.Vector3D(-bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z),
        carla.Vector3D(bounding_box.extent.x, bounding_box.extent.y, bounding_box.extent.z)
    ]

    # Apply reverse rotation to each corner
    rotated_corners = [apply_rotation(corner, reverse_rotation) for corner in corners]

    # Transform to world coordinates
    actor_transform = actor.get_transform()
    world_corners = [actor_transform.transform(corner) for corner in rotated_corners]

    return world_corners

def apply_rotation(point, rotation):
    # Convert rotation to radians
    pitch, yaw, roll = map(math.radians, (rotation.pitch, rotation.yaw, rotation.roll))

    # Calculate rotation matrices
    Rz_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

    Ry_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                         [0, 1, 0],
                         [-math.sin(pitch), 0, math.cos(pitch)]])

    Rx_roll = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])

    # Combined rotation matrix
    R = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))

    # Apply rotation
    rotated_point = np.dot(R, np.array([point.x, point.y, point.z]))

    return carla.Vector3D(rotated_point[0], rotated_point[1], rotated_point[2])

def calculate_aabb_dimensions(corners):
    # Extract X, Y, Z coordinates
    xs = [corner.x for corner in corners]
    ys = [corner.y for corner in corners]
    zs = [corner.z for corner in corners]

    # Calculate dimensions
    width = max(xs) - min(xs)
    length = max(ys) - min(ys)
    height = max(zs) - min(zs)

    return width, length, height

def get_rotation_matrix(rotation):
    """
    Creates a 3D rotation matrix from pitch, yaw, and roll.

    Args:
    rotation (carla.Rotation): The rotation.

    Returns:
    np.array: A 3D rotation matrix.
    """
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    # Create rotation matrix for each axis
    Rz_yaw = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry_pitch = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rx_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # Combine rotation matrices
    R = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))

    return R

def rotation_yaw(points, yaw):
    # Convert rotation to radians
    yaw = math.radians(yaw)

    # Calculate rotation matrices
    R = np.array([[math.cos(yaw), -math.sin(yaw)],
                  [math.sin(yaw), math.cos(yaw)]])

    # Apply rotation
    return np.dot(points,R)

def rotation_yaw_3d(points, yaw):
    # Convert rotation to radians
    yaw = math.radians(yaw)

    # Calculate rotation matrices
    R = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                  [math.sin(yaw), math.cos(yaw), 0],
                  [0,0,1]])

    # Apply rotation
    return np.dot(points,R)

####################################################
#    Bounding boxes
####################################################

def get_close_objects(actors, ego_vehicle, radius=1):
    # Define a reference location (e.g., your ego vehicle location)
    reference_location = ego_vehicle.get_location()
    vehicle_list = [actor for actor in actors.filter('vehicle.*')]
    static_list = [actor for actor in actors.filter('static.prop.*')]
    # Filter actors based on distance and exclude ego vehicle
    close_vehicles = [
        actor for actor in vehicle_list 
        if actor.id != ego_vehicle.id and calculate_distance(actor.get_location(), reference_location) <= radius]
    close_static = [
        actor for actor in static_list 
        if actor.id != ego_vehicle.id and calculate_distance(actor.get_location(), reference_location) <= radius]
    return close_vehicles, close_static

def get_static_objects_vertices(world, ego_vehicle, radius=100):
    obj_vertices = []
    for npc in world.get_actors().filter('static.prop.*'):
        bb = npc.bounding_box
        dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)

        # Filter for the vehicles within 50m
        if dist < radius:

        # Calculate the dot product between the forward vector
        # of the vehicle and the vector between the vehicle
        # and the other vehicle. We threshold this dot product
        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = ego_vehicle.get_transform().get_forward_vector()
            ray = npc.get_transform().location - ego_vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                obj_vertices.append(verts)
    return obj_vertices

def get_actor_vertices(actor):
    bb = actor.bounding_box
    return [v for v in bb.get_world_vertices(actor.get_transform())]

def get_points_from_vertices(vertices, edges):
    points = []
    for edge in edges:
        loc = vertices[edge[0]]
        points.append([loc.x, loc.y])
    return points

def get_object_points(vertices, edges, cost=1, centered=True):
    vertices_list = []
    for vertice in vertices:
        obj_list = []
        extents = []
        for edge in edges:
            point1 = vertice[edge[0]]
            point2 = vertice[edge[1]]
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            dz = point2.z - point1.z
            dim = math.sqrt(dx**2 + dy**2 + dz**2)
            obj_list.append([point1.x, point1.y, point1.z])
            extents.append(dim)
        if centered:    
            center = np.mean(obj_list, axis=0)
            vertices_list.append([center[0], center[1], center[2], extents[0], extents[1], cost, 1])
        else:
            vertices_list.append([x + [y, cost, 1] for x, y in zip(obj_list, extents)])
    return vertices_list

def radian2degree(rad_angle):
    return rad_angle*180/math.pi
