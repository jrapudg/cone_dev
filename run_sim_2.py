import carla
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import queue
import os
from PIL import Image

from modules.custom_planner import local_planner

from modules.custom_planner import behavioural_planner

from modules.general import initialize_random_town, destroy_all_vehicle_objects, destroy_all_static_objects,\
                            get_route_to_random_location, add_lateral_offset, get_vehicle_measurement,\
                            destroy_all_actors, get_vehicle_speed, send_control_command,\
                            create_simulation_folder, generate_map, create_gif, process_semantic_image,\
                            process_trajectory_map, create_gif

from modules.construction import spawn_construction_zone_2, get_config_json, random_true_false

from modules.transformations import get_static_objects_vertices, get_actor_vertices, get_points_from_vertices,\
                                    get_object_points

from modules.camera import build_projection_matrix, project_vertices_to_image, process_semantic_image,\
                           visualize_custom_mask, process_semantic_labels, get_drivable_surface, check_endpoint_in_mask

from modules.custom_planner import controller2d

def save_image_semantic(image):
    """
    Saves an image in the specified folder with a specific image number.

    Args:
    folder_path (str): The path to the folder where the image will be saved.
    image (ImageType): The image to be saved.
    image_number (int): The number to be appended to the image file name.
    """
    global image_folder_semantic
    global id_semantic
    global should_save_image_semantic
    
    image = process_semantic_image(image)   
    if should_save_image_semantic:
        should_save_image_semantic = False
        image_file_path = os.path.join(image_folder_semantic, f"bev_{id_semantic}.png")
        # Assuming 'image' is of a type that has a 'save_to_disk' method
        image = Image.fromarray(image)
        image.save(image_file_path)
        print("BEV Saved!")
        print(image_file_path)
        print()
        id_semantic +=1

def save_image(image):
    """
    Saves an image in the specified folder with a specific image number.

    Args:
    folder_path (str): The path to the folder where the image will be saved.
    image (ImageType): The image to be saved.
    image_number (int): The number to be appended to the image file name.
    """
    global image_folder
    global frame
    global should_save_image
    
    if should_save_image:
        should_save_image = False
        image_file_path = os.path.join(image_folder, f"image_{frame:06d}.png")
        # Assuming 'image' is of a type that has a 'save_to_disk' method
        image.save_to_disk(image_file_path)

def simulate(world, save_gift=False):
    global should_save_image
    global image_folder
    global frame
    global should_save_image_semantic
    global image_folder_semantic
    global frame
    global id_semantic
    global collision_detected
    global TARGET_SPEED
    global CIRCLE_RADII

    ## Settings
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1/ITER_FOR_SIM_TIMESTEP
    world.apply_settings(settings)
    world.tick()

    ## Initialize map and lib
    destroy_all_static_objects(world)
    destroy_all_vehicle_objects(world)

    carla_map = world.get_map()
    bp_lib = world.get_blueprint_library()
    world.tick()

    ## Spawn Ego Vehicle
    available_vehicles = ['vehicle.audi.a2', 'vehicle.audi.etron', "vehicle.audi.tt",'vehicle.citroen.c3',
                          'vehicle.citroen.c3', 'vehicle.tesla.model3', 'vehicle.mini.cooper_s', 'vehicle.nissan.micra']
    
    spawn_points = carla_map.get_spawn_points()
    vehicle_type = random.choice(available_vehicles)
    vehicle_bp = bp_lib.find(vehicle_type) #random.choice(bp_lib.filter('vehicle.car.*')) #bp_lib.find('vehicle.tesla.model3')
    print("Vehicle: {}".format(vehicle_type))
    ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    world.tick()

    ## Get spectator
    spectator = world.get_spectator()
    transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), ego_vehicle.get_transform().rotation)
    spectator.set_transform(transform)
    world.tick()

    # Collision Sensor
    collision_sensor_bp = bp_lib.find('sensor.other.collision')
    # Spawn the collision sensor and attach it to the vehicle
    sensor_transform = carla.Transform(carla.Location(x=0.0, z=2.0)) 
    collision_sensor = world.spawn_actor(collision_sensor_bp, sensor_transform, attach_to=ego_vehicle)
    world.tick()

    ## Initialize cameras
    # Front RGB camera
    camera_front_bp = bp_lib.find('sensor.camera.rgb')
    camera_front_trans = carla.Transform(carla.Location(z=2))
    camera_front = world.spawn_actor(camera_front_bp, camera_front_trans, attach_to=ego_vehicle)

    # Semantic segmentation camera
    #camera_sem_bev_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    #camera_sem_bev_bp.set_attribute('image_size_x', '600')
    #camera_sem_bev_bp.set_attribute('image_size_y', '600')
    #camera_sem_bev_bp.set_attribute('fov', '110')  # Adjust the field of view if needed       
    ## Adjust the camera location and rotation to simulate a BEV perspective
    #camera_sem_bev_trans = carla.Transform(carla.Location(x=0, z=28), carla.Rotation(pitch=-90))
    #camera_sem_bev = world.spawn_actor(camera_sem_bev_bp, camera_sem_bev_trans, attach_to=ego_vehicle)

    # Semantic segmentation camera
    #camera_sem_front_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    #camera_sem_front_bp.set_attribute('fov', '120')
    ## Adjust the camera location and rotation to simulate a BEV perspective
    #camera_sem_front_trans = carla.Transform(carla.Location(z=2))
    #camera_sem_front = world.spawn_actor(camera_sem_front_bp, camera_sem_front_trans, attach_to=ego_vehicle)
    world.tick()

    """
    # Get the world to camera matrix
    world_2_camera_front = np.array(camera_front.get_transform().get_inverse_matrix())
    # Get the attributes from the camera
    image_w = camera_front_bp.get_attribute("image_size_x").as_int()
    image_h = camera_front_bp.get_attribute("image_size_y").as_int()
    fov = camera_front_bp.get_attribute("fov").as_float()
    # Calculate the camera projection matrix to project from 3D -> 2D
    K_front = build_projection_matrix(image_w, image_h, fov)
    """

    # Get the world to camera matrix
    #world_2_camera_sem_front = np.array(camera_sem_front.get_transform().get_inverse_matrix())
    # Get the attributes from the camera
    #image_w = camera_sem_front_bp.get_attribute("image_size_x").as_int()
    #image_h = camera_sem_front_bp.get_attribute("image_size_y").as_int()
    #fov = camera_sem_front_bp.get_attribute("fov").as_float()
    # Calculate the camera projection matrix to project from 3D -> 2D
    #K_sem_front = build_projection_matrix(image_w, image_h, fov)

    """
    # Get the world to camera matrix
    world_2_camera_sem_bev = np.array(camera_sem_bev.get_transform().get_inverse_matrix())
    # Get the attributes from the camera
    image_w = camera_sem_bev_bp.get_attribute("image_size_x").as_int()
    image_h = camera_sem_bev_bp.get_attribute("image_size_y").as_int()
    fov = camera_sem_bev_bp.get_attribute("fov").as_float()
    # Calculate the camera projection matrix to project from 3D -> 2D
    K_sem_bev = build_projection_matrix(image_w, image_h, fov)
    """

    #front_queue = queue.Queue()
    #sem_front_queue = queue.Queue()
    #sem_bev_queue = queue.Queue()

    #camera_front.listen(front_queue.put)
    #camera_sem_front.listen(sem_front_queue.put)
    #camera_sem_bev.listen(sem_bev_queue.put)

    ## Define construction scenarios
    scenarios = ['construction_0',
                'construction_1',
                'construction_2',
                'construction_3',
                'construction_diamond',
                'construction_barrier',
                'no_work_zone']

    scenarios_no_interaction = ['construction_0',
                                'construction_1',
                                'construction_2',
                                'construction_diamond',
                                'construction_barrier']

    ## Get route
    route, waypoints = get_route_to_random_location(world, carla_map, ego_vehicle, resolution=1.0, max_pts=MAX_NUM_WAYPOINTS)

    ## Define construction scenarios
    n_elements = random.randint(6,12)
    edges = [[2,0], [0,4], [6,2], [4,6]]
    lane_width = np.max([waypoints[DIST_TO_FIRST_OBJ].lane_width, 2.5])

    if random_true_false(0.9):
        print("SCENARIO: Interaction")
        scenario_index = random.randint(0, 5)
        #scenario_index = 3
        if waypoints[DIST_TO_FIRST_OBJ].lane_change == carla.libcarla.LaneChange.NONE and\
        waypoints[DIST_TO_FIRST_OBJ].get_right_lane() == None and (not waypoints[DIST_TO_FIRST_OBJ].is_intersection):
            scenario_index = 3
            cons_json = get_config_json(scenarios[scenario_index], lane_width, n_elements)
            TARGET_SPEED = random.uniform(2.0, 3.0)
        else:
            scenario = scenarios[scenario_index]
            if scenario == "construction_3":
                TARGET_SPEED = random.uniform(2.0, 3.0)
            elif scenario == "construction_barrier":
                CIRCLE_RADII = [1.6]*6
            cons_json = get_config_json(scenario, lane_width, n_elements)
        try:
            spawn_construction_zone_2(bp_lib, world, cons_json, 
                                     waypoints[DIST_TO_FIRST_OBJ], 
                                     lane_change=cons_json['lane_change'], 
                                     block_other_lane=cons_json['block_other_lane'], 
                                     rand=cons_json['rand'], 
                                     prob=cons_json['prob'],
                                     signs=cons_json['signs'],
                                     lateral=cons_json['lateral'])
        except Exception as e:
            print(e)
            pass
            
    elif random_true_false(0.8):
        print("SCENARIO: No Interaction")
        scenario_index = random.randint(0, 4)
        scenario = scenarios_no_interaction[scenario_index]
        lane_change_flag = waypoints[DIST_TO_FIRST_OBJ].lane_change
        if lane_change_flag != carla.libcarla.LaneChange.NONE:
            if lane_change_flag == carla.libcarla.LaneChange.Right:
                new_waypoint = waypoints[DIST_TO_FIRST_OBJ].get_right_lane()
            else:
                new_waypoint = waypoints[DIST_TO_FIRST_OBJ].get_left_lane()
            cons_json = get_config_json(scenario, lane_width, n_elements)
            
            try:
                spawn_construction_zone_2(bp_lib, world, cons_json, 
                                          new_waypoint, 
                                          lane_change=cons_json['lane_change'], 
                                          block_other_lane=cons_json['block_other_lane'], 
                                          rand=cons_json['rand'], 
                                          prob=cons_json['prob'],
                                          signs=cons_json['signs'],
                                          lateral=cons_json['lateral'])  
            except Exception as e:
                print(e)
                pass
        else:
            pass
        
    else:
        print("SCENARIO: No Work Zone")
        scenario_index = 6
        pass
    
    print("Scenario: {} \nNumber of lines: {}".format(scenarios[scenario_index], n_elements))

    waypoints_pts = [[t.transform.location.x, t.transform.location.y, TARGET_SPEED] for t in waypoints]

    for i in range(10):
        world.tick()

    constr_vertices = get_static_objects_vertices(world, ego_vehicle, radius=100)
    ego_vertices = get_actor_vertices(ego_vehicle)
    ego_points = get_object_points([ego_vertices], edges)

    static_points_all = get_object_points(constr_vertices, edges)
    static_points_x_y = [obj[:2] for obj in static_points_all]

    start_x, start_y, start_z, ego_yaw = get_vehicle_measurement(ego_vehicle)
    send_control_command(world, ego_vehicle, throttle=0.0, steer=0, brake=1.0, reverse=False)

    lead_car_pos = [[0,0], [0,0]]
    lead_car_speed  = [0,0]

    local_waypoints = None

    controller = controller2d.Controller2D(waypoints)
    lp = local_planner.LocalPlanner(NUM_PATHS,
                                    PATH_OFFSET,
                                    CIRCLE_OFFSETS,
                                    CIRCLE_RADII,
                                    PATH_SELECT_WEIGHT,
                                    TIME_GAP,
                                    A_MAX,
                                    SLOW_SPEED,
                                    STOP_LINE_BUFFER,
                                    carla_map)

    bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE,
                                                [],
                                                LEAD_VEHICLE_LOOKAHEAD)

    snapshot = world.get_snapshot()
    timestamp = snapshot.timestamp
    current_timestamp = timestamp.elapsed_seconds

    for _ in range(20):
        world.tick()
    time.sleep(0.1)

    base_path = './simulations'  # Set your base path here
    simulation_folder, image_folder, image_folder_semantic = create_simulation_folder(base_path)
    id_semantic = 0
    frame = 0
    reached_the_end = False
    collision_detected = False
    trajectory_control = []
    trajectory_map = []

    #camera_sem_front.listen(sem_front_queue.put)
    camera_front.listen(lambda image: save_image(image))
    #camera_sem_bev.listen(lambda image: save_image_semantic(image))
    collision_sensor.listen(on_collision)

    world.tick()

    car_width = ego_points[0][3]
    car_length = ego_points[0][4]

    for frame in range(TOTAL_EPISODE_FRAMES):
        prev_timestamp = current_timestamp
        current_x, current_y, current_z, current_yaw = get_vehicle_measurement(ego_vehicle)
        current_speed = get_vehicle_speed(ego_vehicle)
        print("Frame: {} Current speed: {} m/s".format(frame, current_speed))

        snapshot = world.get_snapshot()
        current_timestamp = snapshot.timestamp.elapsed_seconds

        if frame % LP_FREQUENCY_DIVISOR == 0:
            #  # Compute open loop speed estimate.
            open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

            #  # Calculate the goal state set in the local frame for the local planner.
            #  # Current speed should be open loop for the velocity profile generation.
            ego_state = [current_x, current_y, current_yaw, open_loop_speed]

            #  # Set lookahead based on current speed.
            bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

            #  # Perform a state transition in the behavioural planner.
            bp.transition_state(waypoints_pts, ego_state, current_speed)

            #  # Check to see if we need to follow the lead vehicle.
            bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

            #  # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints_pts, ego_state)

            #  # Calculate planned paths in the local frame.
            paths, path_validity = lp.plan_paths(goal_state_set)

            #  # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)

            if len(static_points_x_y):
                #  # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, [static_points_x_y])
            else:
                collision_check_array = [True]*NUM_PATHS

            best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
            
            best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
           
            # If no path was feasible, continue to follow the previous best path.
            if best_index == None:
                try:
                    best_path = lp._prev_best_path
                except Exception as e:
                    print(e)
                    break
            else:
                best_path = paths[best_index]
                lp._prev_best_path = best_path

            #  # Compute the velocity profile for the path, and compute the waypoints.
            #  # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
            #  # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
            desired_speed = bp._goal_state[2]
            lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
            decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
            local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state, current_speed, decelerate_to_stop, lead_car_state, False)
            # --------------------------------------------------------------

            if local_waypoints != None:
                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.  Linear interpolation computation on the waypoints
                # is also used to ensure a fine resolution between points.
                wp_distance = []   # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                            np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                    (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                                    # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp      = []    # interpolated values
                                    # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))

                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                float(INTERP_DISTANCE_RES)) - 1)
                    wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))

                # Update the other controller values and controls
                controller.update_waypoints(wp_interp)
                pass

        ###
        # Controller Update
        ###
        if local_waypoints != None and local_waypoints != []:
            controller.update_values(current_x, current_y, current_yaw,
                                    current_speed,
                                    current_timestamp, frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
        else:
            cmd_throttle = 0.0
            cmd_steer = 0.0
            cmd_brake = 0.0

        # Skip the first frame or if there exists no local paths
        if local_waypoints == None:
            pass

        # Output controller command to CARLA server
        send_control_command(world,
                            ego_vehicle,
                            throttle=cmd_throttle,
                            steer=cmd_steer,
                            brake=cmd_brake)

        # Find if reached the end of waypoint. If the car is within
        # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
        # the simulation will end.
        dist_to_last_waypoint = np.linalg.norm(np.array([
            waypoints_pts[-1][0] - current_x,
            waypoints_pts[-1][1] - current_y]))

        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), ego_vehicle.get_transform().rotation)

        # Set the spectator's transform
        spectator.set_transform(transform)
        print("Dist_to_target: ", dist_to_last_waypoint)
        if frame % EGO_TRAJ_RATE == 0:
            trajectory_control.append([current_x, current_y, current_z, car_width, car_length, current_yaw, current_speed, 1])
            trajectory_map.append(generate_map(carla_map, current_x, current_y, area_range=80, 
                                            distance_spread=1.0, min_points_per_segment=MIN_POINTS_PER_SEGMENT, 
                                            max_points_per_segment=MAX_POINTS_PER_SEGMENT))
            # Save image
            should_save_image_semantic = True

        if frame % 10 == 0:
            should_save_image = True
            #should_save_image = True

        if  dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
            reached_the_end = True

        if reached_the_end:
            print("End reached")
            break
        
        if collision_detected:
            print("Collision!")
            break
            
    ego_points_loc = np.array(ego_points) 
    ego_location = carla.Location(x=ego_points_loc[0,0], y=ego_points_loc[0,1], z=ego_points_loc[0,2])
    ego_waypoint = carla_map.get_waypoint(ego_location)
    ego_points_loc = np.array([ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, ego_waypoint.transform.location.z])

    lane_centers_pts_np = generate_map(carla_map, ego_waypoint.transform.location.x, ego_waypoint.transform.location.y, area_range=80, distance_spread=1.0)

    ego_points_np = np.array(ego_points) #- ego_points_loc[:2] #2D
    waypoints_pts_np = np.array(waypoints_pts) #- ego_points_loc #3D

    static_points_np = np.array(static_points_all) #- ego_points_loc #3D
    if static_points_np.shape[0] == 0:
        static_points_np = np.array([[0]])

    trajectory_control_np = np.array(trajectory_control) #- ego_points_loc[:2] #2d
    road_pts_np = process_trajectory_map(trajectory_map, trajectory_control)
   
    # Flaten 3D arrays
    lane_centers_pts_2d_np = lane_centers_pts_np.reshape(-1, lane_centers_pts_np.shape[-1])
    waypoints_pts_2d_np = waypoints_pts_np.reshape(-1, waypoints_pts_np.shape[-1])
    static_points_2d_np = static_points_np.reshape(-1, static_points_np.shape[-1])
    road_pts_2d_np = road_pts_np.reshape(-1, road_pts_np.shape[-1])

    static_points_corners_np = np.array(get_object_points(constr_vertices, edges, centered=False))
    if static_points_corners_np.shape[0] == 0:
        static_points_corners_np = np.array([[0]])
    static_points_corners_2d_np = static_points_corners_np.reshape(-1, static_points_corners_np.shape[-1])

    np.savetxt(os.path.join(simulation_folder, "lane_centers_pts_np.txt"), lane_centers_pts_2d_np)
    np.savetxt(os.path.join(simulation_folder, "ego_points_np.txt"), ego_points_np)
    np.savetxt(os.path.join(simulation_folder, "waypoints_pts_np.txt"), waypoints_pts_2d_np)
    np.savetxt(os.path.join(simulation_folder, "static_points_np.txt"), static_points_2d_np)
    np.savetxt(os.path.join(simulation_folder, "trajectory_control_np.txt"), trajectory_control_np)
    np.savetxt(os.path.join(simulation_folder, "road_pts_np.txt"), road_pts_2d_np)
    np.savetxt(os.path.join(simulation_folder, "static_points_corners_np.txt"), static_points_corners_2d_np)
    np.savetxt(os.path.join(simulation_folder, "success.txt"), np.array([reached_the_end]))

    plt.figure(figsize=(20, 16))

    plt.scatter(lane_centers_pts_2d_np[:,0], lane_centers_pts_2d_np[:,1], s=5, color='gray', label="road points")
    plt.plot(waypoints_pts_np[:,0], waypoints_pts_np[:,1],color='c', label="plan points")
    plt.plot(trajectory_control_np[:,0], trajectory_control_np[:,1], marker="o",color='red', label="ego trajectory")
    try:
        plt.scatter(static_points_2d_np[:,0], static_points_2d_np[:,1], s=20,color='orange', label="construction points")
    except:
        pass
    plt.scatter(ego_points_np[:,0], ego_points_np[:,1], s=100, color='b', label="ego points")

    ax = plt.gca()
    ax.set_aspect('equal')
    plt.legend()

    output_plot = os.path.join(simulation_folder, 'scene.png') 
    plt.savefig(output_plot)
    camera_front.destroy()
    #camera_sem_bev.destroy()
    #camera_sem_front.destroy()
    world.tick()
    destroy_all_static_objects(world)
    destroy_all_vehicle_objects(world)
    world.tick()
    
    if save_gift:
        # Example usage
        output_gif = os.path.join(simulation_folder, 'animation.gif')        # Replace with your desired output file path
        frame_duration = 1.5                      # Duration of each frame in seconds
        create_gif(image_folder, output_gif, frame_duration)
        time.sleep(1)

def on_collision(event):
    """
    Callback function that is called whenever the attached vehicle collides with something.
    """
    global collision_detected 
    collision_detected = True
    # The event object contains information about the collision, such as the other actor involved and the impulse.
    other_actor = event.other_actor
    impulse = event.normal_impulse
    intensity = sum([impulse.x, impulse.y, impulse.z])
    
    print(f"Collision detected! Other actor: {other_actor.type_id}, Impulse intensity: {intensity}")

# Simulation Constants
ITER_FOR_SIM_TIMESTEP  = 40     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
DIST_THRESHOLD_TO_LAST_WAYPOINT = 10.0  # some distance from last position before simulation ends

# Planning Constants
TARGET_SPEED           = 4.0
BP_LOOKAHEAD_BASE      = 8.0              # m
BP_LOOKAHEAD_TIME      = 2.0              # s
PATH_OFFSET            = 1.3              # m
CIRCLE_OFFSETS         = [-4.2, -2.5, -1.0, 0.5, 1.5, 3.2] # m
CIRCLE_RADII           = [1.3]*6
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.6              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 10.0             # m
LP_FREQUENCY_DIVISOR   = 4                # Frequency divisor to make the
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.
# Path interpolation parameters
NUM_PATHS = 7
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points
TOTAL_EPISODE_FRAMES      = 5000

# Segment the waypoints
MAX_POINTS_PER_SEGMENT = 40
MIN_POINTS_PER_SEGMENT = 3

# Trajectory
EGO_TRAJ_RATE = 20
MAX_NUM_WAYPOINTS = 150
DIST_TO_FIRST_OBJ = 40

save_gift = False
should_save_image = False
should_save_image_semantic = False
collision_detected =  False

if __name__ == '__main__':
    ### Initialize simulator
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    destroy_all_actors(world)
    town = initialize_random_town(client)
    time.sleep(2)
    world = client.get_world()



    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1/ITER_FOR_SIM_TIMESTEP
    world.apply_settings(settings)
    
    for _ in range(5):
        if town == "/Game/Carla/Maps/Town01":
            TARGET_SPEED = random.uniform(1.5, 2.2)
        elif town == "/Game/Carla/Maps/Town06":
            TARGET_SPEED = random.uniform(3.0, 8.0)
        else:
            TARGET_SPEED = random.uniform(3.0, 5.0)
        simulate(world, save_gift=save_gift)
    destroy_all_actors(world)