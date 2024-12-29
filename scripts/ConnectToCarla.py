import carla
import cv2
import numpy as np
import random
import threading
import time
import math
import signal
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback

last_frame_time = None

import gymnasium as gym
from gymnasium import spaces

distance_in_front = 5.0 # meters

# def process_image(image):
#     array = np.frombuffer(image.raw_data, dtype=np.uint8)
#     array = array.reshape((image.height, image.width, 4))  # RGBA format
#     array = array[:, :, :3]  # Drop alpha channel
#     cv2.imshow("Camera Feed", array)
#     # cv2.waitKey(1)

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, n_steps, total_timesteps, env, verbose=0):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.n_steps = n_steps
        self.current_step = 0
        self.total_timesteps = total_timesteps
        self.env = env  # Reference to CarlaEnv instance

    def _on_step(self) -> bool:
        self.current_step += 1
        # Print information at each training step
        # print(f"Callback step {self.current_step}, Timestep: {self.num_timesteps}")
        
        # Print every iteration
        if self.current_step % self.n_steps == 0:
            iteration = self.current_step // self.n_steps
            print(f"Iteration: {iteration}, Total Timesteps: {self.num_timesteps}")
        # Set record_video to True in CarlaEnv for the last iteration
        # if self.num_timesteps >= self.total_timesteps - self.n_steps:
        #     self.env.record_video = True
        # else:
        #     self.env.record_video = False
        
        return True  # Continue training

class CarlaEnv(gym.Env):
    def __init__(self, carla_world, vehicle, camera, target_vehicle, collision_sensor):
        super(CarlaEnv, self).__init__()
        self.world = carla_world
        self.vehicle = vehicle
        self.camera = camera
        self.target_vehicle = target_vehicle
        self.collision_sensor = collision_sensor
        self.latest_image = None  # Variable to store the most recent camera image
        self.camera_running = False  # Initialize camera_running flag
        self.stop_flag = False # Flag to control when to stop the camera feed
        self.camera_thread = None  # Initialize camera_thread
        self.collision_occurred = False # Initialize collision flag
        self.previous_location = None # Intialize the previous location variable
        self.max_steps = 2048  # Maximum steps per episode
        self.current_step = 0  # Step counter for the current episode
        self.record_video = False  # Initialize video recording flag
        # Define action space: throttle and steering
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # Define observation space: for simplicity, using the camera image dimensions
        # self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8) # for pretrained CNN
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8)
        # Create the camera listener
        self.setup_camera_listener()
        # Create collision listener
        self.setup_collision_listener()

    def setup_camera_listener(self):
        # Set up the camera listener
        # self.camera.listen(lambda image: self.process_camera_image(image))
        if not self.camera_running:
            self.camera.listen(self.process_camera_image)
            self.camera_running = True

    def setup_collision_listener(self):
        # Set up the listener for collision events
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def on_collision(self, event):
        # This function will be called whenever a collision is detected
        self.collision_occurred = True

    def has_collided(self):
        # Return the collision status and reset the flag
        collision_detected = self.collision_occurred
        self.collision_occurred = False  # Reset the flag for the next step
        return collision_detected

    def show_camera_feed(self):
        try:
            while self.camera_running and not self.stop_flag:
                if self.latest_image is not None:
                    cv2.imshow("CARLA Camera Feed", self.latest_image)
                    # Close the feed if 'q' is pressed
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        self.stop_flag = True
                        break
                else:
                    time.sleep(0.01)
        finally:
            self.camera_running = False
            cv2.destroyAllWindows()  # Ensure the window closes when the loop stops

        # while self.camera_running:
        #     if self.latest_image is not None:
        #         cv2.imshow("CARLA Camera Feed", self.latest_image)
        #         cv2.waitKey(10)  # Add a small delay to prevent high CPU usage
        #         # Check if the window was closed
        #         if cv2.getWindowProperty("CARLA Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
        #             self.camera_running = False  # Stop the loop if the window is closed
        #     else:
        #         # Sleep briefly if no image is available yet
        #         time.sleep(0.01)

    def process_camera_image(self, image):
        # Convert the image to a numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha channel

        # Store the latest image
        self.latest_image = array

        # # Write to video if recording is enabled
        # if self.video_out is not None:
        #     self.video_out.write(self.latest_image)

        # # Convert the image to a numpy array
        # array = np.frombuffer(image.raw_data, dtype=np.uint8)
        # array = array.reshape((image.height, image.width, 4))  # RGBA format
        # array = array[:, :, :3]  # Drop the alpha channel

        # # Store the latest image
        # self.latest_image = array

        # # # Display the image in a window
        # # cv2.imshow("CARLA Camera Feed", array)
        # # cv2.waitKey(10)  # Display the image for 1 ms

        # # Start a thread to show the camera feed, if not already running
        # if not self.camera_running:
        #     self.camera_running = True
        #     self.camera_thread = threading.Thread(target=self.show_camera_feed)
        #     self.camera_thread.start()

    def reset(self, **kwargs):
        foo = self.current_step
        self.current_step = 0  # Reset the step counter at the beginning of each episode
        # Optionally handle the 'seed' if needed
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        if self.record_video:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
            self.video_out = cv2.VideoWriter(r"C:\Users\legos\Documents\Post Grad Learning\Projects\DinkleBot/videos/carla_camera_feed.avi", fourcc, 20.0, (self.latest_image.shape[1], self.latest_image.shape[0]))
        else:
            self.video_out = None

        # Clear existing NPCs
        for actor in self.world.get_actors().filter('*vehicle*'):
            if actor.id != self.vehicle.id and actor.id != self.target_vehicle.id:
                actor.destroy()
        # for actor in self.world.get_actors().filter('*vehicle*'):
        #     if actor.is_alive:
        #         actor.destroy()

        # Reset the hero vehicle's position
        hero_spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_transform(hero_spawn_point)

        # Reset the target vehicle's position to be in front of the hero vehicle
        distance_in_front = 10.0  # Distance ahead of the hero vehicle
        yaw = math.radians(hero_spawn_point.rotation.yaw)
        offset_x = distance_in_front * math.cos(yaw)
        offset_y = distance_in_front * math.sin(yaw)

        target_spawn_point = carla.Transform(
            carla.Location(
                x=hero_spawn_point.location.x + offset_x,
                y=hero_spawn_point.location.y + offset_y,
                z=hero_spawn_point.location.z
            ),
            hero_spawn_point.rotation
        )
        self.target_vehicle.set_transform(target_spawn_point)

        # Filter out the hero's spawn point from the NPC spawn points
        spawn_points = [sp for sp in self.world.get_map().get_spawn_points() if sp.location != hero_spawn_point.location]

        # Spawn NPC vehicles randomly throughout the remaining spawn points
        num_npcs = 20  # Adjust as needed
        for _ in range(num_npcs):
            npc_spawn_point = random.choice(spawn_points)
            npc_bp = random.choice(self.world.get_blueprint_library().filter('*vehicle*'))
            self.world.try_spawn_actor(npc_bp, npc_spawn_point)

        # Disable auto-pilot for our hero vehicle
        self.vehicle.set_autopilot(False)
        # Enable autopilot for all other NPC vehicles
        for NPC_vehicle in self.world.get_actors().filter('*vehicle*'):
            if NPC_vehicle.id != self.vehicle.id:  # Make sure not to enable autopilot for your vehicle
                NPC_vehicle.set_autopilot(True)

        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        # Set up or reset the collision sensor
        if self.collision_sensor is None or not self.collision_sensor.is_alive:
            # Create a new collision sensor if it doesn't exist or is destroyed
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            # Use on_collision instead of handle_collision
            self.collision_sensor.listen(lambda event: self.on_collision(event))
        else:
            # Reattach the existing sensor to the vehicle
            self.collision_sensor.set_transform(carla.Transform())
            self.collision_sensor.listen(lambda event: self.on_collision(event))

        # Reset flags and data for the new episode
        self.collision_occurred = False
        self.latest_image = None
        self.camera_running = False
        self.previous_location = None
        self.stop_flag = False

        # Start the camera feed if not already running
        if not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.show_camera_feed)
            self.camera_thread.start()

        # Get the initial observation and reset any necessary sensors
        observation = self.get_observation()
        info = {}  # Populate with extra info if needed

        if foo % 20 == 0:
            print(f"Hero vehicle: {self.vehicle}, Camera: {self.camera}, Collision Sensor: {self.collision_sensor}")


        return observation, info

    def step(self, action):
        # Increment the step counter
        self.current_step += 1
        # Apply the action (e.g., throttle and steering)
        # throttle, steering = action
        throttle = float(action[0])
        steering = float(action[1])
        control = carla.VehicleControl(throttle=throttle, steer=steering)
        self.vehicle.apply_control(control)

        # Step the simulation
        self.world.tick()

        # Get the observation, reward, done flag, and info
        observation = self.get_observation()
        reward, done = self.compute_reward()
        # done = self.check_done()
        truncated = False  # If you want to use truncation (optional)
        info = {}
        return observation, reward, done, truncated, info

    def get_observation(self):
        if self.latest_image is not None:
            # Resize the image to match the observation space shape
            # resized_image = cv2.resize(self.latest_image, (640, 480))
            resized_image = self.latest_image
            return resized_image
        else:
            # Return a placeholder observation if no image is available yet
            # return np.zeros((224, 224, 3), dtype=np.uint8)
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def compute_reward(self):
        # Start with a base reward
        reward = 0.0
        done = False

        # Reward for staying on the road
        if self.is_on_road():
            off_road_penalty = 1.0  # Positive reward for staying on the road
        else:
            off_road_penalty -= 2.0  # Penalty for going off-road

        # Reward based on speed
        speed = self.vehicle.get_velocity()
        speed_magnitude = np.linalg.norm([speed.x, speed.y, speed.z])
        target_speed = 10.0  # Target speed in m/s
        speed_diff = abs(target_speed - speed_magnitude)
        speed_reward = max(0, 1.0 - speed_diff / target_speed)
        speed_reward = 0.0

        # Penalty for collisions
        if self.has_collided():
            print("Collision occurred. done=True")
            collision_penalty = 5.0
            done = True
            time.sleep(0.2)
        else:
            collision_penalty = 0.0

        if self.current_step >= self.max_steps:  # End episode if max steps reached
            print("Max steps reached. done=True")
            done = True

        # Reward for maintaining sight on the target vehicle
        visual_maintained_reward = self.compute_visual_maintained()

        # Reward for maintaining stability while tracking the target vehicle
        stability_reward = self.compute_tracking_stability()

        # Penalty for going off the road
        off_road_penalty = self.is_on_road()

        # Reward for driving close to the target speed
        # reward += max(0, 1.0 - speed_diff / target_speed)
        #reward = 0.1 * speed_reward + 1.0 * visual_maintained_reward - 1.0 * collision_penalty + - 0.5 * off_road_penalty + 1.0 * stability_reward
        reward = 2.0 * visual_maintained_reward - 1.0 * collision_penalty + - 0.5 * off_road_penalty + 1.0 * stability_reward
        
        if self.current_step % 20 == 0:
            print(f"Reward: {reward}, Done: {done}, Step: {self.current_step}")
        return reward, done        # Calculate the reward based on the agent's performance
   
    def compute_visual_maintained(self):
        # Get the 2D screen position of the target vehicle
        screen_position = self.get_target_screen_location(self.target_vehicle)

        # Check if the screen position is valid and within screen boundaries
        if screen_position is not None:
            screen_x, screen_y = screen_position
            screen_width, screen_height = 800, 600  # Example screen dimensions

            # Check if the target is within the screen boundaries
            if 0 <= screen_x < screen_width and 0 <= screen_y < screen_height:
                return 1.0  # Positive reward for maintaining visual contact

        # Return 0.0 if the target is not visible
        return 0.0

    def is_on_road(self):
        # Get the current location of the vehicle
        vehicle_location = self.vehicle.get_location()

        # Retrieve the nearest waypoint on the road network
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=(carla.LaneType.Driving))

        # Check if the waypoint is valid and belongs to a driving lane
        if waypoint is not None and waypoint.lane_type == carla.LaneType.Driving:
            return True  # The vehicle is on the road

        return False  # The vehicle is off the road

    def get_target_screen_location(self, target_vehicle):
        # Get the target vehicle's 3D world position
        target_location = target_vehicle.get_location()

        # Project the 3D location to 2D screen coordinates using the camera
        camera_transform = self.camera.get_transform()
        camera_location = camera_transform.location
        camera_rotation = camera_transform.rotation

        # Use CARLA's utility to project the 3D location to the 2D screen
        screen_pos = self.world_to_screen(target_location, camera_location, camera_rotation)
        return screen_pos

    def world_to_screen(self, location, camera_location, camera_rotation):
        # Simplified projection from 3D to 2D screen coordinates
        relative_position = location - camera_location
        forward_vector = rotate_vector(carla.Vector3D(1, 0, 0), camera_rotation)
        distance = relative_position.dot(forward_vector)

        if distance > 0:  # Check if the target is in front of the camera
            # Calculate screen position (normalize relative to screen dimensions)
            screen_x = relative_position.x / distance
            screen_y = relative_position.y / distance
            return (int(screen_x * 800), int(screen_y * 600))  # Example screen size
        else:
            return None

    def compute_stability_reward(self, current_location, previous_location):
            # Ensure both locations are valid (not None)
        if current_location is None or previous_location is None:
            return 0.0  # No reward if we don't have valid locations

        # Calculate the distance between the current and previous screen positions
        position_change = np.linalg.norm(np.array(current_location) - np.array(previous_location))

        # Define a maximum change for normalization (to scale the reward)
        max_allowed_change = 75.0  # You can adjust this based on the screen dimensions and desired sensitivity

        # Reward stability: smaller changes get higher rewards (invert the distance)
        stability_reward = max(0, 1.0 - (position_change / max_allowed_change))

        return stability_reward

    def compute_tracking_stability(self):
        # Get current screen location of the target vehicle
        current_location = self.get_target_screen_location(self.target_vehicle)

        # If the target is not visible, return zero reward
        if current_location is None or self.previous_location is None:
            self.previous_location = current_location
            return 0.0

        # Calculate stability reward based on location change
        stability_reward = self.compute_stability_reward(current_location, self.previous_location)

        # Update the previous location for the next step
        self.previous_location = current_location

        return stability_reward
    

    def check_done(self):
        return 0
        # Check if the episode is done

    # def close(self):
    #     # Destroy our hero vehicle
    #     vehicle.destroy()
    #     # Clean up resources
    #     if self.camera is not None:
    #         self.camera.stop()
    #         self.camera.destroy()
    #     # Stop the camera feed thread
    #     self.camera_running = False
    #     if self.camera_thread is not None:
    #         self.camera_thread.join()
    #     if self.video_out is not None:
    #         self.video_out.release()  # Finalize and save the video
    #     # Close OpenCV windows
    #     cv2.destroyAllWindows()
    #     print("GOT HERE")
    def close(self):
        # Destroy our hero vehicle
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        # Clean up resources
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        
        # Stop the camera feed thread
        self.camera_running = False
        if self.camera_thread is not None:
            self.camera_thread.join()
        
        # Release video writer if it exists
        if self.video_out is not None:
            self.video_out.release()  # Finalize and save the video
            print("VideoWriter released and video saved.")

        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("Environment closed.")

# Set up signal handler for Ctrl+C to cleanly exit
def signal_handler(sig, frame):
    print("Exiting program...")
    env.stop_flag = True  # Set the stop flag to stop the camera feed
    cv2.destroyAllWindows()  # Ensure all windows are closed
    exit(0)

def rotate_vector(vector, rotation):
    # Convert rotation angles from degrees to radians
    pitch = math.radians(rotation.pitch)
    yaw = math.radians(rotation.yaw)
    roll = math.radians(rotation.roll)

    # Rotation around the Z-axis (yaw)
    x = vector.x * math.cos(yaw) - vector.y * math.sin(yaw)
    y = vector.x * math.sin(yaw) + vector.y * math.cos(yaw)
    z = vector.z

    # Rotation around the Y-axis (pitch)
    z_pitch = z * math.cos(pitch) - x * math.sin(pitch)
    x = z * math.sin(pitch) + x * math.cos(pitch)
    z = z_pitch

    # Rotation around the X-axis (roll)
    y_roll = y * math.cos(roll) - z * math.sin(roll)
    z = y * math.sin(roll) + z * math.cos(roll)
    y = y_roll

    # Return the rotated vector
    return carla.Vector3D(x, y, z)

def control_vehicle(camera_feed):
    control = carla.VehicleControl()
    control.throttle = 2.5  # Example action (throttle)
    control.steer = 3.0     # Example action (steering)
    vehicle.apply_control(control)

def is_location_free(world, location, radius=2.0):
    # Check if there are any actors within a given radius of the specified location
    actors_nearby = world.get_actors().filter('*vehicle*')
    for actor in actors_nearby:
        if actor.get_location().distance(location) < radius:
            return False  # Location is not free
    return True  # Location is free


# def process_image(image):
#     array = np.frombuffer(image.raw_data, dtype=np.uint8)
#     array = array.reshape((image.height, image.width, 4))  # RGBA format
#     array = array[:, :, :3]  # Drop alpha channel
#     control_vehicle(array)
#     cv2.imshow("Camera Feed", array)
#     cv2.waitKey(1)

def process_segmentation_image(image):
    global last_frame_time

    # Calculate the framerate
    current_time = image.timestamp
    if last_frame_time is not None:
        fps = 1.0 / (current_time - last_frame_time)
        # print(f"Framerate: {fps:.2f} FPS")
    last_frame_time = current_time

    # Convert the segmentation image to a format suitable for visualization
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # RGBA format
    array = array[:, :, :3]  # Drop alpha channel
    cv2.imshow("Segmentation Camera Feed", array)
    cv2.waitKey(1)

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world (simulation environment)
world = client.get_world()
print("Connected to CARLA!")
world = client.reload_world()

# world_settings = world.get_settings()
# world_settings.fixed_delta_seconds = 1.0 / 60.0  # 60 FPS
# world.apply_settings(world_settings)

blueprint_library = world.get_blueprint_library()

target_model_keyword = "CarlaCola" #"mini"
target_vehicle_bp = blueprint_library.filter(f'vehicle.*{target_model_keyword}*')[0]

# Get hero vehicle from library
hero_model_keyword = "mini"
hero_vehicle_bp = blueprint_library.filter(f'vehicle.*{hero_model_keyword}*')[0]
# Get vehicle library to build NPC vehicles
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
filtered_blueprints = [bp for bp in vehicle_blueprints if bp != target_vehicle_bp and bp != hero_vehicle_bp]
# Get the map's spawn points
spawn_points = world.get_map().get_spawn_points()[2:]
# Spawn 50 vehicles randomly distributed throughout the map 
# for each spawn point, we choose a random vehicle from the blueprint library
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
# Spawn our hero vehicle
hero_spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(hero_vehicle_bp, hero_spawn_point)
# Attach a camera sensor to our hero vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("fov", "110")  # Increase the field of view (default is usually 90)
# camera_bp.set_attribute("image_size_x", "400")
# camera_bp.set_attribute("image_size_y", "300")
# camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
# camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Position the camera on top of the vehicle
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))  # Tilt down slightly
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
# Attach collision sensor to our hero car
collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

# Create target vehicle
#-----------------------------------------------------------
# # Calculate the forward offset based on the agent's heading (yaw)
# yaw = math.radians(hero_spawn_point.rotation.yaw)
# offset_x = distance_in_front * math.cos(yaw)
# offset_y = distance_in_front * math.sin(yaw)
# # Create a new spawn point for the target vehicle
# target_spawn_point = carla.Transform(
#     carla.Location(
#         x=hero_spawn_point.location.x + offset_x,
#         y=hero_spawn_point.location.y + offset_y,
#         z=hero_spawn_point.location.z  # Keep the same height
#     ),
#     hero_spawn_point.rotation  # Keep the same orientation as the agent
# )
# target_vehicle_bp = vehicle_blueprints[1]
# target_vehicle = world.spawn_actor(target_vehicle_bp, target_spawn_point)
# Attempt to find a free spawn location

# # Choose a nearby spawn point for the target vehicle
# target_spawn_point = world.get_map().get_spawn_points()[1]  # Choose an adjacent spawn point or one in front

# # Spawn the target vehicle at this existing spawn point
# target_vehicle = world.spawn_actor(target_vehicle_bp, target_spawn_point)

max_attempts = 10
for attempt in range(max_attempts):
    # Calculate the target vehicle's spawn location
    yaw = math.radians(hero_spawn_point.rotation.yaw)
    offset_x = distance_in_front * math.cos(yaw)
    offset_y = distance_in_front * math.sin(yaw)
    target_spawn_point = carla.Transform(
        carla.Location(
            x=hero_spawn_point.location.x + 5.0, #offset_x,
            y=hero_spawn_point.location.y,# + offset_y,
            z=hero_spawn_point.location.z  # Keep the same height
        ),
        hero_spawn_point.rotation  # Keep the same orientation as the agent
    )

    # Check if the location is free
    if is_location_free(world, target_spawn_point.location):
        # Spawn the target vehicle
        # target_vehicle_bp = blueprint_library.filter('vehicle.*')[1]  # Choose a different vehicle blueprint
        target_vehicle = world.spawn_actor(target_vehicle_bp, target_spawn_point)
        break
    else:
        # Increase the distance and try again
        distance_in_front += 2.0  # Increment distance by 2 meters for each attempt
else:
    raise RuntimeError("Failed to find a free spawn location for the target vehicle.")

if target_vehicle is not None:
    print("Target vehicle successfully spawned:", target_vehicle.id)
    print("Target vehicle location:", target_vehicle.get_location())
else:
    print("Failed to spawn target vehicle.")

# world.debug.draw_line(
#     hero_spawn_point.location,
#     target_spawn_point.location,
#     thickness=8.0,
#     color=carla.Color(0, 0, 255),
#     life_time=100.0
# )

# Disable auto-pilot for our hero vehicle
vehicle.set_autopilot(False)
# Enable autopilot for all other NPC vehicles
for NPC_vehicle in world.get_actors().filter('*vehicle*'):
    if NPC_vehicle.id != vehicle.id:  # Make sure not to enable autopilot for your vehicle
        NPC_vehicle.set_autopilot(True)

# Create an instance of your custom CARLA environment
carla_env = CarlaEnv(world, vehicle, camera, target_vehicle, collision_sensor)
# Wrap the CARLA environment with DummyVecEnv
env = DummyVecEnv([lambda: carla_env])
# Transpose the image channels for using image-based observations
env = VecTransposeImage(env)

################################################################################################
############### TRAINED MODEL TEST #############################################################
# Specify the path to your saved model
model_path = "C:/Users/legos/Documents/Post Grad Learning/Projects/DinkleBot/models/ppo_carla_model.zip"

# Load the trained model
model = PPO.load(model_path, env=env)

# Reset the environment
obs = env.reset()

# Run one iteration
action, _states = model.predict(obs, deterministic=True)
obs, reward, done, truncated, info = env.step(action)

# Print results
print("Action:", action)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

if done:
    print("Episode finished. Resetting environment.")
    obs = env.reset()

exit()

####################################################################################################

# Reset environemnt to start new "episode"
obs = env.reset()

# Initialize the PPO model
model = PPO(
    'CnnPolicy', 
    env, 
    learning_rate = 0.0003,
    n_steps = 2048,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    verbose=1, 
    tensorboard_log="./ppo_carla_tensorboard/",
    device='cuda'
    )

print("Using device:", model.device)

# for NPC_vehicle in world.get_actors().filter('*vehicle*'):
#     # NPC_vehicle.set_autopilot(True)
#     if NPC_vehicle == vehicle:
#         vehicle.set_autopilot(False)
#         # continue
#     else:
#         NPC_vehicle.set_autopilot(True)

# Start camera streaming
# camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))
#camera.listen(lambda image: process_image(image))
# camera.listen(lambda image: process_segmentation_image(image))

# Running the program - testing process
# try:
#     while True:
#         world.tick()  # Advances the simulation by one step
# except KeyboardInterrupt:
#     print("Script stopped.")
# finally:
#     camera.stop()
#     vehicle.destroy()
#     for actor in world.get_actors().filter('*vehicle*'):
#         actor.destroy()
#     cv2.destroyAllWindows()

# try:
#     while True:
#         world.tick()  # Advances the simulation by one frame
# except KeyboardInterrupt:
#     print("Script stopped.")
# finally:
#     # Reset settings to asynchronous mode
#     world_settings.synchronous_mode = False
#     world.apply_settings(world_settings)
#     camera.stop()
#     vehicle.destroy()
#     for actor in world.get_actors().filter('*vehicle*'):
#         actor.destroy()
#     cv2.destroyAllWindows()

# # Take a few steps in the environment
# for _ in range(10):
#     # Randomly sample an action from the action space
#     action = env.action_space.sample()
#     # Apply the action and get the results
#     obs, rewards, dones, infos = env.step([action])  # Note the list around action
#     # Print out some information
#     print(f"Action taken: {action}, Reward: {rewards}, Done: {dones}")
#     # If the episode is done, reset the environment
#     if dones[0]:
#         obs = env.reset()
#         print("Environment reset for a new episode.")

# # Check if the target vehicle's rendering components are enabled
# if target_vehicle is not None:
#     is_simulating = target_vehicle.get_physics_control()
#     print("Target vehicle is simulating physics:", is_simulating)

# signal.signal(signal.SIGINT, signal_handler)

# save_path = os.path.abspath("../models/ppo_carla_model")
# print("Model will be saved at:", save_path)
# exit()

# Train the model for a given number of timesteps
total_timesteps = 200000  # Example: 100,000 timesteps

# Initialize the callback
callback = TrainingMonitorCallback(n_steps=2048, total_timesteps=200000, env=carla_env, verbose=1)

try:
    start_time = time.time()
    # Start the training loop
    model.learn(total_timesteps=total_timesteps, callback = callback)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Optionally save the trained model
    # os.makedirs("../models", exist_ok=True)
    # model.save("../models/ppo_carla_model")
    save_path = "C:/Users/legos/Documents/Post Grad Learning/Projects/DinkleBot/models/ppo_carla_model.zip"
    model.save(save_path)
except Exception as e:
    print(f"Training terminated due to: {e}")
finally:
    # Cleanup any resources
    env.close()
    print("GOT PASSED CALL TO CLOSE")
# For later:
# Train the PPO model
# model.learn(total_timesteps=10000)

# Optionally, save the model
# model.save("ppo_carla_model")
