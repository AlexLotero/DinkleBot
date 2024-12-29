import os
# save_path = os.path.abspath("../models/ppo_carla_model")
# print("Model will be saved at:", save_path)
# exit()

# Optionally save the trained model
# os.makedirs("../models", exist_ok=True)
# model.save("../models/ppo_carla_model")
# model.save(save_path)

# save_path = os.path.abspath("../models/ppo_carla_model")
# print("Model will be saved at:", save_path)
# model.save(save_path)

save_path = "C:/Users/legos/Documents/Post Grad Learning/Projects/DinkleBot/models/ppo_carla_model"
# completeName = os.path.join(save_path, name_of_file+".txt")         

file1 = open(save_path, "w")

toFile = "Write what you want into the field"

file1.write(toFile)

file1.close()





# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world (simulation environment)
world = client.get_world()
print("Connected to CARLA!")
world = client.reload_world()

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
# camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
# camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Position the camera on top of the vehicle
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))  # Tilt down slightly
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
# Attach collision sensor to our hero car
collision_bp = blueprint_library.find('sensor.other.collision')
collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

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

# Reset environemnt to start new "episode"
obs = env.reset()