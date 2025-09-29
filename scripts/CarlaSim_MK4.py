# TODO: reward seems to often be -0.25, even though it is moving and visual is maintained:
# TODO: Visual Maintained is turning up 0, need to find why!!!

# C:\Users\legos\Documents\Miscellaneous\Programming\WindowsNoEditor
# C:\Users\legos\Documents\Post Grad Learning\Projects\DinkleBot\scripts
import carla
import cv2
import numpy as np
import random
import threading
import time
import math
import signal
import os
import logging
import os
from datetime import datetime

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

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Create log file, named with timestamp
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# Log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)

# We can use for custom logging, monitoring, or control logic during training
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
            # print(f"Iteration: {iteration}, Total Timesteps: {self.num_timesteps}")
            logger.info(f"Iteration {iteration}, Total Timesteps={self.num_timesteps}")

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
        self.vehicle = vehicle # hero vehicle
        self.camera = camera # camera on top of the hero vehicle
        self.collision_sensor = collision_sensor # collison sensor attached to our hero vehicle
        self.target_vehicle = target_vehicle # target vehicle
        self.latest_image = None  # Variable to store the most recent camera image
        self.camera_running = False  # Initialize a flag to indicate if the camera is running
        self.stop_flag = False # Flag to control when to stop the camera feed
        self.camera_thread = None  # Initialize camera_thread, seperate background thread to keep camera display running, without blocking simulation or learning 
        self.collision_occurred = False # Initialize a flag to indicate if a collision has occurred
        self.previous_location = None # Intialize the previous location variable, last recorded screen position of target vehicle
        self.max_steps = 2048  # Maximum steps per episode, may be edited as needed for training
        self.current_step = 0  # Step counter for the currently running episode
        self.record_video = False  # Initialize video recording flag, has video recording been enabled
        
        # Define action space: throttle and steering
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32) # actions available to take within CARLA env
        
        # Define observation space: for simplicity, using the camera image dimensions
        # self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8) # for pretrained CNN
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8) # the camera frame the hero sees
        
        # Create the camera listener, the listener "recieves" the image that can then be sent to wherever for processing
        self.setup_camera_listener()
        # Create collision listener, the listener "listens" in order to create an event when a collision is detected
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
        # Return the collision status (collision has happened or not) and reset the flag
        collision_detected = self.collision_occurred
        # self.collision_occurred = False  # Reset the flag for the next step
        return collision_detected

    def show_camera_feed(self):
        # Create and display the camera feed window so long as the 'q' key has not been pushed, 
        # the camera is actually running, and the stop camera flag has not been set elsewhere in execution
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
        # Convert the image to a numpy array, a format that our model can learn on
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # Drop alpha channel

        # Store this latest image
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
        # Resetting the entirety of the environment between each episode
        # ** ALMOST CERTAINLY NEED TO REARRANGE THE ORDER OF OPERATIONS IN THIS FUNCTION - ALL DELETING COMPLETED BEFORE ANY NEW SPAWNING **
        foo = self.current_step
        self.current_step = 0  # Reset the step counter at the beginning of each episode
        # Optionally handle the 'seed' if needed
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        self.collision_occurred = False

        if self.record_video: # if the record video flag is set, do the following
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
            self.video_out = cv2.VideoWriter(r"C:\Users\legos\Documents\Post Grad Learning\Projects\DinkleBot/videos/carla_camera_feed.avi", fourcc, 20.0, (self.latest_image.shape[1], self.latest_image.shape[0]))
        else:
            self.video_out = None

        # Clear existing NPCs, not the hero vehicle or target vehicle
        for actor in self.world.get_actors().filter('*vehicle*'):
            if actor.id != self.vehicle.id and actor.id != self.target_vehicle.id:
                actor.destroy()
        # for actor in self.world.get_actors().filter('*vehicle*'):
        #     actor.destroy()

        # Re-select the hero vehicle's position and "move" the existing hero vehicle to that position
        # TODO: REVIEW BENEFITS OF MOVING HERO RATHER THAN RESPAWING
        hero_spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle.set_transform(hero_spawn_point)

        # Re-select the target vehicle's position to be in front of the hero vehicle again
        # May need to adopt the 'try' behavior from the initial spawning at the bottom of this prgm
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

        # Remove the hero spawn point to make a pool of possible NPC spawn points
        spawn_points = [sp for sp in self.world.get_map().get_spawn_points() if sp.location != hero_spawn_point.location]

        # Spawn NPC vehicles randomly throughout the remaining spawn points
        num_npcs = 20  # Not important until our model is consistently "working", adjust as needed
        for _ in range(num_npcs):
            npc_spawn_point = random.choice(spawn_points)
            npc_bp = random.choice(self.world.get_blueprint_library().filter('*vehicle*'))
            self.world.try_spawn_actor(npc_bp, npc_spawn_point)
        
        # Enable autopilot for all newly spawned NPC vehicles
        for NPC_vehicle in self.world.get_actors().filter('*vehicle*'):
            if NPC_vehicle.id != self.vehicle.id:  # Make sure not to enable autopilot for your vehicle
                NPC_vehicle.set_autopilot(True)

        # Disable auto-pilot for our hero vehicle, since we only moved it, likely already turned off (good to check in case?)
        self.vehicle.set_autopilot(False)

        # Destroy any existing collision sensor in order to spawn new one in next iteration
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None

        # Set up or reset the collision sensor
        if self.collision_sensor is None or not self.collision_sensor.is_alive:
            # Create a new collision sensor if it doesn't exist or was destroyed
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            # Use on_collision instead of handle_collision
            self.collision_sensor.listen(lambda event: self.on_collision(event))
        else:
            # Reattach the existing sensor to the vehicle, might be unnecessary since I destroy it right before this?
            self.collision_sensor.set_transform(carla.Transform())
            self.collision_sensor.listen(lambda event: self.on_collision(event))

        # Reset flags and data for the new episode
        self.collision_occurred = False
        self.latest_image = None
        self.camera_running = False
        self.previous_location = None
        self.stop_flag = False

        # Start the camera feed if not already running, including a thread to "watch" it
        if not self.camera_running:
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.show_camera_feed)
            self.camera_thread.start()

        # Get the initial observation and reset any necessary sensors
        observation = self.get_observation()
        info = {}  # Populate with extra info if needed

        # For debugging, every 50 steps we print the (status of) our hero vehicle and its two sensors
        if foo % 50 == 0:
            print(f"Hero vehicle: {self.vehicle}, Camera: {self.camera}, Collision Sensor: {self.collision_sensor}")

        # Returns the observations returned by self.get_observation(), we don't currently use info
        return observation, info

    def step(self, action):
        # Increment the step counter (2048 steps per iteration, step ~= instance of a possible action)
        self.current_step += 1

        # Apply the action (e.g., throttle and steering) (drone project will likely include verticle "throttle" as well as horizontal)
        # throttle, steering = action
        throttle = float(action[0])
        steering = float(action[1])
        control = carla.VehicleControl(throttle=throttle, steer=steering)
        self.vehicle.apply_control(control)

        # Step the CARLA simulation (2048 steps per iteration, step ~= instance of a possible action)
        self.world.tick()

        # Get the observation, reward, done flag, and info
        observation = self.get_observation() # observation, at this stage, is just the camera image after a step/action
        reward, done = self.compute_reward()
        # done = self.check_done()      # Commented out because we have nothing in the check done yet, seem to do it all elsewhere
        truncated = False  # episode only ends when max steps is reached or the collision occurs
        info = {}
        # if self.current_step % 100 == 0:
            # print("observation: ", observation)
            # print("reward: ", reward)
            # print("done: ", done)
            # print("truncated: ", truncated)
            # print("info: ", info)
            # time.sleep(5)
        return observation, reward, done, truncated, info
        # return observation, reward, done, info

    def get_observation(self):
        # observation, at this stage, is just the camera image after a step/action
        if self.latest_image is not None:
            # Resize the image to match the observation space shape, stopped resizing as part of debugging, may need to reactivate
            # resized_image = cv2.resize(self.latest_image, (640, 480))
            resized_image = self.latest_image
            return resized_image
        else:
            # Return a placeholder observation if no image is available yet (first step)
            # return np.zeros((224, 224, 3), dtype=np.uint8)
            return np.zeros((600, 800, 3), dtype=np.uint8)

    def compute_reward(self):
        # Establish a base reward of zero
        reward = 0.0
        done = False

        # End episode if max steps reached
        if self.current_step >= self.max_steps:
            print("Max steps reached. done=True")
            done = True

        # # Reward for staying on the road
        # TODO: Consider reintroducing off-road penalty reward when stability is improved
        # if self.is_on_road():
        #     off_road_penalty = 1.0  # Positive reward for staying on the road
        # else:
        #     off_road_penalty -= 2.0  # Penalty for going off-road

        # Reward based on speed - NOT CURRENLTY USING - TECHNICALLY IRRELEVENT FOR OUR EVENTUAL PURPOSES
        speed = self.vehicle.get_velocity()
        speed_magnitude = int(np.linalg.norm([speed.x, speed.y, speed.z]))
        if speed_magnitude > 0:
            speed_reward = 0.1  # small reward for being in motion
        else:
            speed_reward = -0.25  # tiny penalty for standing still
        # target_speed = 10.0  # Target speed in m/s
        # speed_diff = abs(target_speed - speed_magnitude)
        # speed_reward = max(0, 1.0 - speed_diff / target_speed)
        # speed_reward = 0.0

        # Penalty for collisions, if collided, we weight the punishment for the eventual reward calculation
        if self.has_collided():
            # print("Collision occurred. done=True")
            logger.warning("Collision occurred. Setting done=True")
            collision_penalty = 5.0     # large penalty for collisions
            done = True     # the current iteration should end if there was a collision, can't recover from that
            time.sleep(0.2)     # was added as part of debugging, will likely be removed for final draft
        else:
            collision_penalty = 0.0     # no penalty should be applied if no collision occurs

        # Reward for maintaining sight on the target vehicle, for now our most important goal - largest reward potential
        visual_maintained_reward = self.compute_visual_maintained()

        # Reward for maintaining stability (~stable video feed) while tracking the target vehicle, a stable tracking is better for future application
        stability_reward = self.compute_tracking_stability()

        # Penalty for going off the road, HAD THIS BACKWARDS - WAS REWARDING LEAVING THE ROAD
        off_road_penalty = not self.is_on_road()

        # Reward for driving close to the target speed - SEE ABOVE
        # reward += max(0, 1.0 - speed_diff / target_speed)

        # Calculate total reward based on weighted sub rewards
        #reward = 0.1 * speed_reward + 1.0 * visual_maintained_reward - 1.0 * collision_penalty + - 0.5 * off_road_penalty + 1.0 * stability_reward
        reward = 2.5 * visual_maintained_reward - 1.5 * collision_penalty + - 1.0 * off_road_penalty + 1.5 * stability_reward + speed_reward

        # Log reward results
        logger.info(
            f"Step {self.current_step}: "
            f"Reward={reward:.2f}, Visual={visual_maintained_reward:.2f}, "
            f"Stability={stability_reward:.2f}, Collision={collision_penalty}, "
            f"OnRoad={not off_road_penalty}"
        )


        return reward, done        # Return the reward reflecting the hero's performance
   
    # determine "how well" the hero vehicle has maintained visual on the target vehicle
    def compute_visual_maintained(self):
        # retrieve the screen position of the target vehicle
        screen_position = self.get_target_screen_location(self.target_vehicle)
        if screen_position is not None:
            screen_x, screen_y = screen_position
            screen_width, screen_height = 800, 600  # Screen dimensions

            # "Normalize" position to the screen center, to ensure reward is consistent despite changes in screen resolution
            x_center, y_center = screen_width / 2, screen_height / 2
            distance_from_center = np.linalg.norm([screen_x - x_center, screen_y - y_center])
            max_distance = np.linalg.norm([x_center, y_center])

            # Reward inversely proportional to distance from the center - closer to center, higher reward
            return max(0, 1.0 - (distance_from_center / max_distance))
        return 0.0

    def get_target_screen_location(self, target_vehicle):
        # Get the world position of the target
        target_location = target_vehicle.get_location()

        # Use the camera's transform to compute the relative position
        camera_transform = self.camera.get_transform()
        camera_location = camera_transform.location
        camera_rotation = camera_transform.rotation

        # Compute relative position (target - camera location)
        relative_location = target_location - camera_location

        # Apply camera rotation to align with the camera's view
        relative_location = self.rotate_to_camera(relative_location, camera_rotation)

        # Convert relative location to a NumPy array for matrix operations
        relative_coords = np.array([relative_location.x, relative_location.y, relative_location.z])

        # The intrinsic matrix (example values), 3D projection into 2D coordinates
        intrinsic_matrix = np.array([
            [800, 0, 400],  # Focal length in pixels, principal point (cx)
            [0, 800, 300],  # Focal length in pixels, principal point (cy)
            [0, 0, 1]
        ])

        # Project to 2D screen space
        screen_coords = intrinsic_matrix @ relative_coords

        # Normalize the screen coordinates
        if screen_coords[2] > 0:  # Ensure the target is in front of the camera
            screen_x = screen_coords[0] / screen_coords[2]
            screen_y = screen_coords[1] / screen_coords[2]
            return screen_x, screen_y

        # Return None if the target is not in front of the camera
        return None

    def is_on_road(self):
        # Get the current location of the vehicle
        vehicle_location = self.vehicle.get_location()

        # Retrieve the nearest waypoint on the road network
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=(carla.LaneType.Driving))

        # Check if the waypoint is valid and belongs to the actual driving lane
        if waypoint is not None and waypoint.lane_type == carla.LaneType.Driving:
            return True  # The vehicle is on the road

        return False  # The vehicle is off the road

    # translate target env location into camera coordinates
    def rotate_to_camera(self, relative_location, camera_rotation):  
        # Convert CARLA rotation to radians
        pitch, yaw, roll = np.radians([camera_rotation.pitch, camera_rotation.yaw, camera_rotation.roll])

        # Rotation matrices for pitch, yaw, roll
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Apply rotation to relative location
        rotated_coords = R @ np.array([relative_location.x, relative_location.y, relative_location.z])
        return carla.Location(x=rotated_coords[0], y=rotated_coords[1], z=rotated_coords[2])


    def world_to_screen(self, location, camera_location, camera_rotation):
        # Simplified projection from 3D to 2D screen coordinates
        relative_position = location - camera_location
        forward_vector = rotate_vector(carla.Vector3D(1, 0, 0), camera_rotation)
        distance = relative_position.dot(forward_vector)

        if distance > 0:  # Check if the target is in front of the camera
            # Calculate screen position (normalize relative to screen dimensions)
            screen_x = relative_position.x / distance
            screen_y = relative_position.y / distance
            return (int(screen_x * 800), int(screen_y * 600))  # Screen size
        else:
            return None

    def compute_stability_reward(self, current_location, previous_location):
        # Ensure both locations are valid (not None)
        if current_location is None or previous_location is None:
            return 0.0  # No reward if we don't have valid locations

        # Calculate the distance between the current and previous screen positions
        position_change = np.linalg.norm(np.array(current_location) - np.array(previous_location))

        # Define a maximum change for normalization (to scale the reward)
        max_allowed_change = 75.0  # We'll be adjusting this based on the screen dimensions and desired sensitivity

        # Reward stability: smaller changes get higher rewards (invert the distance)
        stability_reward = max(0, 1.0 - (position_change / max_allowed_change))

        # We'll have to confirm that this isn't discouraging turning in general
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

        # We'll have to confirm that his is even helping our model, may just be better to reward a visual maintained rather than a "stable" one
        return stability_reward
    

    def check_done(self):
        return 0
        # Check if the episode is done

    def close(self):
        # Destroy our hero vehicle
        if self.vehicle is not None:
            self.vehicle.destroy()
        
        # Destroy the camera
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


# Connect to the CARLA server - CARLA runs on port 2000 by default
client = carla.Client('localhost', 2000)
# Set a timeout so that our connection attempt doesn't just hang forever
client.set_timeout(10.0)

# Get the world (simulation environment) - necessary to "interact" with the CARLA environment
world = client.get_world()
print("Connected to CARLA!")
# Added this reload because on restart issues were being reported about the world already existing
world = client.reload_world()

# Limit the CARLA world to 60 FPS, for both logic and rendering (I believe)
# world_settings = world.get_settings()
# world_settings.fixed_delta_seconds = 1.0 / 60.0  # 60 FPS
# world.apply_settings(world_settings)

# Fetch the "library" from which we can pull "models" to populate the world, e.g. people, specific cars, etc.
blueprint_library = world.get_blueprint_library()

# We define our target vehicle to be the Cola truck (because its easy to see in our cammera window)
target_model_keyword = "CarlaCola" #"mini"
# We find this Cola truck model within the library defined above
target_vehicle_bp = blueprint_library.filter(f'vehicle.*{target_model_keyword}*')[0]

# Define and retrieve our hero vehicle from library, model is less important here
hero_model_keyword = "mini"
hero_vehicle_bp = blueprint_library.filter(f'vehicle.*{hero_model_keyword}*')[0]

# Get vehicle library to build NPC vehicles, these can be anytype of vehicle
vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
# Any type of vehicle, EXCEPT our target and hero vehicles - namely target to ensure an NPC doesn't confuse our camera
filtered_blueprints = [bp for bp in vehicle_blueprints if bp != target_vehicle_bp and bp != hero_vehicle_bp]

# Get the map's possible spawn points
spawn_points = world.get_map().get_spawn_points()[2:]

# Spawn 50 vehicles randomly distributed throughout the map 
# for each spawn point, we choose a random vehicle from the blueprint library
# 50 is an arbitrary number, more could be deployed later to improve or test our model
for i in range(0,50):
    world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Spawn our hero vehicle
hero_spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(hero_vehicle_bp, hero_spawn_point)

# Attach a camera sensor to our hero vehicle, and set a 110 FOV
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("fov", "110")  # Increase the field of view (default is usually 90)
# Position the camera on top of the vehicle, tilted down slightly (tilt added during debugging)
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)) 
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Camera tweaks created during debugging, not currently in use
# camera_bp.set_attribute("image_size_x", "400")
# camera_bp.set_attribute("image_size_y", "300")
# camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
# camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  

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

# We spawn the target vehicle in front of the hero vehicle's existing spawn point, currently 5 (feet?) in front
# It is assumed, for now, that in eventual deployment, the target vehicle will always start directly in front of the hero
# We must 'attempt' to spawn the target because there could already be another car or object in front of our hero
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

# Disable auto-pilot for our hero vehicle, as it will be driven by our model
vehicle.set_autopilot(False)
# Enable autopilot for all other NPC vehicles
for NPC_vehicle in world.get_actors().filter('*vehicle*'):
    # Make sure not to enable autopilot for the hero vehicle
    if NPC_vehicle.id != vehicle.id: 
        NPC_vehicle.set_autopilot(True)

# Create an instance of our custom CARLA environment
carla_env = CarlaEnv(world, vehicle, camera, target_vehicle, collision_sensor)

# Wrap the CARLA environment with DummyVecEnv, for custom CARLA <--> PPO compatibility
env = DummyVecEnv([lambda: carla_env])
# Transpose the image channels for using image-based observations
env = VecTransposeImage(env)
#########################################################

################################################################################################
############### TRAINED MODEL TEST #############################################################
################################################################################################
# # Specify the path to our saved model
# model_path = "C:/Users/legos/Documents/Post Grad Learning/Projects/DinkleBot/models/ppo_carla_model.zip"

# # Load the trained model
# model = PPO.load(model_path, env=env)

# # Reset the environment
# obs = env.reset()

# # Run one iteration
# action, _states = model.predict(obs, deterministic=True)
# obs, reward, done, truncated, info = env.step(action)

# # Print results
# print("Action:", action)
# print("Reward:", reward)
# print("Done:", done)
# print("Info:", info)

# if done:
#     print("Episode finished. Resetting environment.")
#     obs = env.reset()

# exit()

####################################################################################################

# Reset environemnt to start new "episode", not sure if this is required, but I believe I added it during debugging
obs = env.reset()

# Initialize the PPO model
# Settings are somewhat arbitrary at this point, will be edited as needed
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

# Print confirmation that the model is running on our GPU
print("Using device:", model.device)

# Start camera streaming, currently disabled on the ~initialization-episode
# camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))
# camera.listen(lambda image: process_image(image))
# camera.listen(lambda image: process_segmentation_image(image))

# Train the model for a given number of timesteps
total_timesteps_ = 1000000  # Example: 100,000 timesteps

# Initialize the callback, monitors training progress, can print at fixed intervals for debug
callback = TrainingMonitorCallback(n_steps=2048, total_timesteps=total_timesteps_, env=carla_env, verbose=1)

# We begin training the model, one CARLA env iteration at a time, 2048 steps per iteration
# 100,000 timesteps / 2048 n_steps = ~48 iterations
# Likely need more iterations to truly train the model, but we need to get this much working first
# Currently using "try" as part of the debugging process, need to identify what is causing early termination
try:
    start_time = time.time()
    # Start the training loop
    model.learn(total_timesteps=total_timesteps_, callback = callback)
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
# model.learn(total_timesteps=total_timesteps_, callback = callback)
#nprint("--- %s seconds ---" % (time.time() - start_time))

# Optionally, save the model
# model.save("ppo_carla_model")
