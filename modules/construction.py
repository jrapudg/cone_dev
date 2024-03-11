import carla
import numpy as np
import os
import math
import random

from modules.general import pad

def transform_local_point_to_global(waypoint_transform, local_point):
    """
    Transform a local point relative to a waypoint's transform to global coordinates.

    Parameters:
    - waypoint_transform: The transform of the waypoint (carla.Transform).
    - local_point: The local coordinate relative to the waypoint's transform (carla.Location).

    Returns:
    - A carla.Location object representing the point in global coordinates.
    """
    # Decompose the waypoint's transform
    loc = waypoint_transform.location
    rot = waypoint_transform.rotation

    # Convert rotation to radians
    yaw = math.radians(rot.yaw)
    pitch = math.radians(rot.pitch)
    roll = math.radians(rot.roll)

    # Calculate cosines and sines for rotation angles
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)

    # Rotation matrix for yaw (ignoring pitch and roll for simplicity)
    x_world = local_point.x * cos_yaw - local_point.y * sin_yaw
    y_world = local_point.x * sin_yaw + local_point.y * cos_yaw
    z_world = local_point.z  # This simplistic approach ignores pitch and roll effects

    # Apply translation
    world_point = carla.Location(x=x_world + loc.x, y=y_world + loc.y, z=z_world + loc.z)

    return world_point

def random_true_false(probability_of_true=0.5):
    """
    Returns a random True or False, where True has a specified probability of occurring.

    Parameters:
    - probability_of_true: The probability of returning True, between 0 and 1.

    Returns:
    - True or False, randomly chosen based on the specified probability.
    """
    return random.random() < probability_of_true

def spawn_construction_zone_2(bp_lib, world, object_description, waypoint, 
                            lane_change=False, block_other_lane=False, 
                            rand=False, prob=0.5, signs=False, lateral=False, block=False):
    current_waypoint = waypoint
    
    if lateral:
        delta_lateral = random.uniform(-0.3, 0.3)
    else:
        delta_lateral = 0
    
    """
    dir_factor = -1
    if not block:
        if abs(current_waypoint.lane_id) > 1:
            if current_waypoint.lane_change == carla.libcarla.LaneChange.Left or current_waypoint.is_intersection:
                dir_factor = 1

        else:
            if current_waypoint.lane_change == carla.libcarla.LaneChange.NONE:
                dir_factor = 1
            elif current_waypoint.is_intersection and (current_waypoint.lane_change == carla.libcarla.LaneChange.Both or 
                                                       current_waypoint.lane_change == carla.libcarla.LaneChange.Left):
                dir_factor = 1
    """

    dir_factor = -1
    if not block:
        if abs(current_waypoint.lane_id) > 1:
            if current_waypoint.lane_change == carla.libcarla.LaneChange.Left or current_waypoint.is_intersection:
                dir_factor = 1

        else:
            if current_waypoint.lane_change == carla.libcarla.LaneChange.NONE:
                dir_factor = 1
            elif current_waypoint.is_intersection and (current_waypoint.lane_change == carla.libcarla.LaneChange.Both or 
                                                       current_waypoint.lane_change == carla.libcarla.LaneChange.Left):
                dir_factor = 1

    prev_y_coord = 0
    for index, (elem, (dx, dy), deg) in enumerate(zip(object_description['elements'], object_description['offsets'], 
                                                      object_description['rotations'])):                                                
        if rand: 
            if not random_true_false(prob): # % probability of spawn
                continue
        
        bp = bp_lib.filter(elem)[0]
        
        if index == 0:
            prev_y_coord = dy
            if dy > 0:
                current_waypoint = current_waypoint.next(dy)[-1]
                
            pass
        else:
            if lane_change and index == 16 and current_waypoint.get_left_lane() != carla.libcarla.LaneChange.NONE:
                prev_waypoint = current_waypoint
                #print(current_waypoint.lane_change)
                if carla.libcarla.LaneChange.Left == current_waypoint.lane_change or carla.libcarla.LaneChange.Both == current_waypoint.lane_change:
                    #print("Left")
                    current_waypoint = current_waypoint.get_left_lane()
                elif carla.libcarla.LaneChange.Right == current_waypoint.lane_change:
                    #print("Right")
                    current_waypoint = current_waypoint.get_right_lane()
                else:
                    pass
                current_waypoint = current_waypoint.next(14)[-1]
                
                location_1 = prev_waypoint.transform.location
                location_2 = current_waypoint.transform.location
                #alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                alpha = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
                for i in range(len(alpha)):
                    delta_inter_x = abs(dx)
                    new_x = (1-alpha[i])*location_1.x + (alpha[i])*location_2.x
                    new_y = (1-alpha[i])*location_1.y + (alpha[i])*location_2.y
                    new_location = carla.Location(x=new_x, y=new_y, z=0)
                    local_location = carla.Location(x=0, y=delta_inter_x+0.2, z=0)
                    
                    transform = carla.Transform(new_location, prev_waypoint.transform.rotation)
                    global_point = transform_local_point_to_global(transform, local_location)
                    spawn_transform = carla.Transform(global_point, prev_waypoint.transform.rotation)
                    obj = world.spawn_actor(bp, spawn_transform)
                    
                    local_location = carla.Location(x=0, y=-delta_inter_x+delta_lateral, z=0)
                    
                    transform = carla.Transform(new_location, prev_waypoint.transform.rotation)
                    global_point = transform_local_point_to_global(transform, local_location)
                    spawn_transform = carla.Transform(global_point, prev_waypoint.transform.rotation)
                    obj = world.spawn_actor(bp, spawn_transform) 
            
            if dy > prev_y_coord:
                current_waypoint = current_waypoint.next(dy-prev_y_coord)[-1]
                #print(current_waypoint.transform)
                prev_y_coord = dy
                
        sigma = random.uniform(0.01, 0.06)
        # Get the z-coordinate from the waypoint to adjust to the road surface
        local_location = carla.Location(x=0+random.gauss(mu=0, sigma=sigma), y=dir_factor*dx+random.gauss(mu=0, sigma=sigma)+delta_lateral, z=0)
        #print("Deg= {}".format(deg))
        global_point = transform_local_point_to_global(current_waypoint.transform, local_location)
        new_rotation = carla.Rotation(pitch=current_waypoint.transform.rotation.pitch, 
                                      yaw=current_waypoint.transform.rotation.yaw + deg, 
                                      roll=current_waypoint.transform.rotation.roll)       
        # Set the new transform for spawning the actor
        spawn_transform = carla.Transform(global_point, new_rotation)

        obj = world.spawn_actor(bp, spawn_transform)
    
    if signs:
        n_signs = len(object_description['elements'])//2
        cons_json = get_config_json('construction_signs', waypoint.lane_width, n_signs)
        spawn_construction_zone_2(bp_lib, world, cons_json, waypoint, 
                                  lane_change=False, block_other_lane=False, rand=True, prob=0.8, signs=False)
        
        
    if block_other_lane:
        lane_w = np.max([waypoint.lane_width, 3])
        spacing_lat = lane_w/1.8
        spacing_for = 1.0

        try:
            if carla.libcarla.LaneChange.Left == waypoint.lane_change:
                print("Left")
                waypoint = waypoint.get_left_lane()
                change_factor = 1
                object_description = {'elements':['trafficcone01']*4,
                                    'offsets':[(0.85*spacing_lat*change_factor,0), (0.4*spacing_lat*change_factor, spacing_for),
                                                (-0.15*spacing_lat*change_factor,2*spacing_for), (-0.4*spacing_lat*change_factor,3*spacing_for)],
                                    'rotations':([0]*4)}
                spawn_construction_zone_2(bp_lib, world, object_description, waypoint, lane_change=False, 
                                          block_other_lane=False, rand=False, prob=1.0, signs=False, lateral=False,
                                          block=True)
            elif carla.libcarla.LaneChange.Right == waypoint.lane_change:
                print("Right")
                waypoint = waypoint.get_right_lane()
                #waypoint.lane_change = carla.libcarla.LaneChange.Left
                change_factor = -1
                object_description = {'elements':['trafficcone01']*4,
                                    'offsets':[(0.85*spacing_lat*change_factor,0), (0.4*spacing_lat*change_factor, spacing_for),
                                                (-0.15*spacing_lat*change_factor,2*spacing_for), (-0.4*spacing_lat*change_factor,3*spacing_for)],
                                    'rotations':([0]*4)}
                spawn_construction_zone_2(bp_lib, world, object_description, waypoint, lane_change=False, 
                                          block_other_lane=False, rand=False, prob=1.0, signs=False, lateral=False,
                                          block=True)
            elif carla.libcarla.LaneChange.Both == waypoint.lane_change:  
                print("Both")         
                waypoint_right = waypoint.get_right_lane()
                #print("ID: {}".format(waypoint_right.lane_id))
                change_factor = -1
                object_description = {'elements':['trafficcone01']*4,
                                    'offsets':[(0.85*spacing_lat*change_factor,0), (0.4*spacing_lat*change_factor, spacing_for),
                                                (-0.1*spacing_lat*change_factor,2*spacing_for), (-0.4*spacing_lat*change_factor,3*spacing_for)],
                                    'rotations':([0]*4)}
                spawn_construction_zone_2(bp_lib, world, object_description, waypoint_right, lane_change=False, 
                                        block_other_lane=False, rand=False, prob=1.0, signs=False, lateral=False,
                                        block=True)
                waypoint_left = waypoint.get_left_lane()
                change_factor = 1 
                object_description = {'elements':['trafficcone01']*4,
                                    'offsets':[(0.85*spacing_lat*change_factor,0), (0.4*spacing_lat*change_factor, spacing_for),
                                            (-0.15*spacing_lat*change_factor,2*spacing_for), (-0.4*spacing_lat*change_factor,3*spacing_for)],
                                    'rotations':([0]*4)}
                spawn_construction_zone_2(bp_lib, world, object_description, waypoint_left, lane_change=False, 
                                        block_other_lane=False, rand=False, prob=1.0, signs=False, lateral=False,
                                        block=True)
            else:
                return
        except:
            pass
        
def spawn_construction_zone(bp_lib, world, ego_vehicle, object_description, waypoints, n_waypoint=45):
    for index, (elem, (dx, dy), deg, t) in enumerate(zip(object_description['elements'], object_description['offsets'], 
                                                      object_description['rotations'], object_description['type'])):
        carla_map = world.get_map()
        if t == 0:
            bp = bp_lib.filter(elem)[0]

            calculated_location = waypoints[n_waypoint + index*2]
    
            waypoint = carla_map.get_waypoint(calculated_location, project_to_road=True, lane_type=carla.LaneType.Driving)

            # Get the z-coordinate from the waypoint to adjust to the road surface
            new_location = carla.Location(calculated_location.x, calculated_location.y, waypoint.transform.location.z)
            new_rotation = carla.Rotation(pitch=waypoint.transform.rotation.pitch, 
                                          yaw=ego_vehicle.transform.rotation.yaw + deg, 
                                          roll=waypoint.transform.rotation.roll)
            # Set the new transform for spawning the actor
            spawn_transform = carla.Transform(new_location, new_rotation)

            obj = world.spawn_actor(bp, spawn_transform)
        else:
            bp = bp_lib.filter(elem)[0]  
            calculated_location = waypoints[n_waypoint]
            waypoint = carla_map.get_waypoint(calculated_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            
            #for i in range(len(object_description['elements'])):
            # Get the z-coordinate from the waypoint to adjust to the road surface
            new_location = carla.Location(calculated_location.x+dx, calculated_location.y+dy, waypoint.transform.location.z)
            new_rotation = carla.Rotation(pitch=waypoint.transform.rotation.pitch, 
                                          yaw=waypoint.transform.rotation.yaw + deg, 
                                          roll=waypoint.transform.rotation.roll)
            # Set the new transform for spawning the actor
            spawn_transform = carla.Transform(new_location, new_rotation)

            obj = world.spawn_actor(bp, spawn_transform)


############################################################################################################
            # Construction zone definitions
############################################################################################################
            
def construction_0(d_lateral, d_forward):
    cone_type = random.choice(['constructioncone', 'trafficcone01'])
    construction = {'elements':([cone_type]*3 + ['warningconstruction'] + ['trafficwarning'] + [cone_type, cone_type, 'warningconstruction'] + 
                                                           [cone_type, cone_type, cone_type, 'warningconstruction']*2),
                    'offsets':[(0,-0.5), (d_lateral, -0.25), (-d_lateral,0), (1.3*d_lateral, 0),
                             (0, 2*d_forward),
                             (d_lateral, 4*d_forward), (-d_lateral,4*d_forward), (1.3*d_lateral,4*d_forward),
                             (0,8*d_forward), (d_lateral, 8*d_forward), (-d_lateral,8*d_forward),  (1.3*d_lateral,8*d_forward),
                             (0,12*d_forward), (d_lateral, 12*d_forward), (-d_lateral,12*d_forward),  (1.3*d_lateral,12*d_forward)],
                    'rotations':([0]*3+[90]*2+[0, 0, 90]+[0, 0, 0, 90]*2),
                    'lane_change':False,
                    'block_other_lane':False,
                    'rand':False,
                    'prob':0.98,
                    'signs':False,
                    'lateral':False}
    return construction

def construction_1(d_lateral, d_forward, n_lines):
    lateral_index = [1,-1]
    spacing_lines = 4
    construction = {'elements':(['trafficcone01']*5 + ['trafficwarning']
                                + ['trafficcone01']*(3+len(lateral_index)*n_lines)),
                    'offsets':[(d_lateral,0), (0.4*d_lateral, d_forward),
                               (-0.4*d_lateral,2*d_forward), (-d_lateral,3*d_forward),
                               (d_lateral,3*d_forward), (0, 5*d_forward)]
                               +[(lat_multiplier * d_lateral, (spacing_lines*spacing_multiplier+7) * d_forward)
                                      for spacing_multiplier in range(n_lines)
                                      for lat_multiplier in lateral_index] +
                                  [(d_lateral, ((spacing_lines)*n_lines+7) * d_forward),
                                   (0, ((spacing_lines)*n_lines+7) * d_forward),
                                   (-d_lateral, ((spacing_lines)*n_lines+7) * d_forward)],
                    'rotations':([0]*5+[90]+[0]*(3+len(lateral_index)*n_lines)),
                    'lane_change':False,
                    'block_other_lane':False,
                    'rand':False,
                    'prob':0.98,
                    'signs':False,
                    'lateral':False}
    return construction

def construction_2(d_lateral, d_forward, n_lines):
    cone_type = random.choice(['constructioncone', 'trafficcone01'])
    spacing_lines = 4
    lateral_index = [1,-1] 
    construction = {'elements':([cone_type]*5 + [cone_type]*(3+len(lateral_index)*n_lines)),
                    'offsets':[(d_lateral,0), (0.4*d_lateral, d_forward), 
                                 (-0.4*d_lateral,2*d_forward), (-d_lateral,3*d_forward), (d_lateral,3*d_forward)]\
                                 +[(lat_multiplier * d_lateral, (spacing_lines*spacing_multiplier+7) * d_forward)
                                      for spacing_multiplier in range(n_lines)
                                      for lat_multiplier in lateral_index] +\
                                  [(d_lateral, ((spacing_lines)*n_lines+7) * d_forward),
                                   (0, ((spacing_lines)*n_lines+7) * d_forward),
                                   (-d_lateral, ((spacing_lines)*n_lines+7) * d_forward)],
                    'rotations':([0]*5+[0]*(3+len(lateral_index)*n_lines)),
                    'lane_change':False,
                    'block_other_lane':False,
                    'rand':False,
                    'prob':0.98,
                    'signs':True,
                    'lateral':False}
    return construction

def construction_3(d_lateral, d_forward, n_lines):
    spacing_lines = 2
    lateral_index = [1,-1] 
    construction = {'elements':['trafficcone01']*(2*n_lines),
                    'offsets':[(lat_multiplier * d_lateral, (spacing_lines*spacing_multiplier+4) * d_forward)
                                    for spacing_multiplier in range(n_lines)
                                    for lat_multiplier in lateral_index],
                    'rotations':([0]*(len(lateral_index)*n_lines)),
                    'lane_change':True,
                    'block_other_lane':True,
                    'rand':False,
                    'prob':0.98,
                    'signs':False,
                    'lateral':False}
    return construction

def construction_signs(d_lateral, d_forward, n_lines):
    spacing_lines = 4
    construction = {'elements':['warningconstruction']*(n_lines),
                    'offsets':[(1.4*d_lateral, (spacing_lines*spacing_multiplier) * d_forward)
                                      for spacing_multiplier in range(n_lines)],
                    'rotations':([90]*n_lines),
                    'lane_change':False,
                    'block_other_lane':False,
                    'rand':True,
                    'prob':0.7,
                    'signs':False,
                    'lateral':False}
    return construction

def construction_diamond(d_lateral, d_forward):
    bias = 0.2
    cone_diamond = {'elements':['constructioncone']*4,
                    'offsets':[(bias,0), (-d_lateral+bias,d_forward), (d_lateral+bias,d_forward), (bias,2*d_forward)],
                    'rotations':[0]*4,
                    'lane_change':False,
                    'block_other_lane':False,
                    'rand':True,
                    'prob':1.0,
                    'signs':False,
                    'lateral':True}
    return cone_diamond

def construction_barrier(d_lateral):
    bias = 0.2
    barriers = {'elements':['streetbarrier']*3,
                'offsets':[(bias,0), (-d_lateral+bias,0), (d_lateral+bias,0)],
                'rotations':[90]*3,
                'lane_change':False,
                'block_other_lane':False,
                'rand':True,
                'prob':0.8,
                'signs':False,
                'lateral':True}
    return barriers
############################################################################################################

def get_spacing(construction_type, lane_width):
    if construction_type == "construction_0":
        d_lateral = lane_width/2.1
        d_forward = 0.8
    elif construction_type == "construction_1":
        d_lateral = lane_width/2.1
        d_forward = 0.8
    elif construction_type == "construction_2":
        d_lateral = lane_width/2.1
        d_forward = 0.8
    elif construction_type == "construction_3":
        d_lateral = lane_width/1.4
        d_forward = 0.8
    elif construction_type == "construction_signs":
        d_lateral = lane_width/1.6
        d_forward = 0.8
    elif construction_type == "construction_diamond":
        d_lateral = lane_width/5.2
        d_forward = 0.8
    elif construction_type == "construction_barrier":
        d_lateral = lane_width/3
        d_forward = 0.8
    else:
        raise NotImplementedError
    return d_lateral, d_forward

def get_config_json(construction_type, lane_width, n_lines):
    d_lateral, d_forward = get_spacing(construction_type, lane_width)

    if construction_type == "construction_0":
        config_json = construction_0(d_lateral, d_forward)
    elif construction_type == "construction_1":
        config_json = construction_1(d_lateral, d_forward, n_lines)
        config_json['signs'] = random_true_false()
    elif construction_type == "construction_2":
        config_json = construction_2(d_lateral, d_forward, n_lines)
        config_json['signs'] = random_true_false()
    elif construction_type == "construction_3":
        config_json = construction_3(d_lateral, d_forward, n_lines)
        config_json['signs'] = False #random_true_false()
        config_json['lane_change'] = True #random_true_false()
    elif construction_type == "construction_signs":
        config_json = construction_signs(d_lateral, d_forward, n_lines)
    elif construction_type == "construction_diamond":
        config_json = construction_diamond(d_lateral, d_forward)
    elif construction_type == "construction_barrier":
        config_json = construction_barrier(d_lateral)
    else:
        raise NotImplementedError
    return config_json
        
def process_construction(static_pts_np, center_location, cost=1):
    #obj_pts_np = np.concatenate((obj_pts_np, np.ones((obj_pts_np.shape[0], 1, 1))), axis=-1)
    #obj_pts_np = np.concatenate((obj_pts_np, cost*np.ones((obj_pts_np.shape[0], 1, 1))), axis=-1)
    obj_pts_np = pad(static_pts_np, center_location, thresh=50)
    #obj_pts_np[:, :, [2, 3]] = obj_pts_np[:, :, [3, 2]]

    return obj_pts_np

def create_trajectories(sequence, past_length=4, future_length=12):
    n = sequence.shape[0]
    past_trajectories = []
    future_trajectories = []

    # Ensure that the sliding window does not go out of bounds
    for i in range(n - past_length - future_length + 1):
        past = sequence[i:i + past_length]
        future = sequence[i + past_length:i + past_length + future_length]
        past_trajectories.append(past)
        future_trajectories.append(future)

    return np.array(past_trajectories), np.array(future_trajectories)

def load_ego_trajectories(filename, data_path):
    trajectory_path = os.path.join(data_path, filename, "trajectory_control_np.txt")
    trajectory_control_np = np.loadtxt(trajectory_path)
    #trajectory_control_np = trajectory_control_np[15:]
    trajectory_control_np = trajectory_control_np
    #print("Traj: ", trajectory_control_np.shape)
    past_traj, future_traj = create_trajectories(trajectory_control_np)
    return past_traj, future_traj

def load_road_points(filename, data_path):
    road_path = os.path.join(data_path, filename, "road_pts_np.txt")
    road_pts_np = np.loadtxt(road_path)
    road_pts_np_pro = road_pts_np.reshape(-1,100,40,road_pts_np.shape[-1])
    road_pts_np_pro = road_pts_np_pro[3:-12]
    road_pts_np_pro = road_pts_np_pro
    return road_pts_np_pro

def load_constr_points(filename, data_path):
    static_points_path = os.path.join(data_path, filename, "static_points_np.txt")
    static_pts_np = np.loadtxt(static_points_path)
    return static_pts_np

def load_constr_corner_points(filename, data_path):
    static_points_path = os.path.join(data_path, filename, "static_points_corners_np.txt")
    static_pts_np = np.loadtxt(static_points_path)
    return static_pts_np

def insert_length(array):
    new_shape = (array.shape[0], array.shape[1], array.shape[2] + 1)  # Incrementing the last dimension
    new_array = np.zeros(new_shape) 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Check if all elements are zero in the point's last dimension
            if np.all(array[i, j, :] == 0):
                insert_value = 0
            else:
                insert_value = 1
            # Copy elements before the insertion point
            new_array[i, j, :4] = array[i, j, :4]
            # Insert the conditional value
            new_array[i, j, 4] = insert_value
            # Copy the remaining elements
            new_array[i, j, 5:] = array[i, j, 4:]    
    return new_array