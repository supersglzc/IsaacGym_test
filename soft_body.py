"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Soft Body
---------
Simple import of a URDF with a soft body link and rigid body press mechanism
"""

import math
import random
from isaacgym import gymapi
from isaacgym import gymutil
import os
import shutil
import open3d as o3d
import numpy as np
from PIL import Image as im
import h5py
import point_cloud_utils as pcd_utils


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="FEM Soft Body Example")
if args.physics_engine != gymapi.SIM_FLEX:
    print("*** Soft body example only supports FleX")
    print("*** Run example with --flex flag")
    quit()

random.seed(7)

# simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.75
sim_params.flex.shape_collision_margin = 0.1

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

# set up the env grid
num_envs = 1
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# cache some common handles for later use
envs = []
soft_actors = []
actor_shadows = []

sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.segmentation_id = 1
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load urdf for sphere asset used to create softbody
asset_root = "../../assets"
soft_asset_file = "urdf/icosphere.urdf"

soft_thickness = 0.1    # important to add some thickness to the soft body to avoid interpenetrations

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)

# Print asset soft material properties
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

asset_root_shadow = "../../assets"
asset_file_shadow = "mjcf/open_ai_assets/hand/shadow_hand.xml"

asset_options_shadow = gymapi.AssetOptions()
asset_options_shadow.fix_base_link = True
asset_options_shadow.armature = 0.01

asset_shadow = gym.load_asset(sim, asset_root_shadow, asset_file_shadow, asset_options_shadow)
print("Load shadow hand")

dof_names = gym.get_asset_dof_names(asset_shadow)
dof_props = gym.get_asset_dof_properties(asset_shadow)
num_dofs = gym.get_asset_dof_count(asset_shadow)
dof_states = np.zeros((num_envs, num_dofs), dtype=gymapi.DofState.dtype)
dof_types = [gym.get_asset_dof_type(asset_shadow, i) for i in range(num_dofs)]
dof_positions = dof_states[:]['pos']
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
# defaults = np.zeros((num_envs, num_dofs))
speeds = np.zeros(num_dofs)
speed_scale = 1.0
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        # make sure our default position is in range
        # if lower_limits[i] > 0.0:
        #     defaults[:][i] = lower_limits[i]
        # elif upper_limits[i] < 0.0:
        #     defaults[:][i] = upper_limits[i]
    else:
        # set reasonable animation limits for unlimited joints
        if dof_types[i] == gymapi.DOF_ROTATION:
            # unlimited revolute joint
            lower_limits[i] = -math.pi
            upper_limits[i] = math.pi
        elif dof_types[i] == gymapi.DOF_TRANSLATION:
            # unlimited prismatic joint
            lower_limits[i] = -1.0
            upper_limits[i] = 1.0
    # set DOF position to default

    # dof_positions[:][i] = defaults[:][i]
    # set speed depending on DOF type and range of motion
    if dof_types[i] == gymapi.DOF_ROTATION:
        speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])
    

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))
for i in range(num_envs):

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.1, 0.0)

    # add soft body + rail actor
    soft_actor = gym.create_actor(env, soft_asset, pose, "soft", i, 1, segmentationId=200)
    soft_actors.append(soft_actor)

    # set soft material within a range of default
    actor_default_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    actor_soft_materials = gym.get_actor_soft_materials(env, soft_actor)
    for j in range(asset_soft_body_count):
        youngs = actor_soft_materials[j].youngs
        actor_soft_materials[j].youngs = random.uniform(youngs * 0.2, youngs * 2.4)

        poissons = actor_soft_materials[j].poissons
        actor_soft_materials[j].poissons = random.uniform(poissons * 0.8, poissons * 1.2)

        damping = actor_soft_materials[j].damping
        # damping is 0, instead we just randomize from scratch
        # actor_soft_materials[j].damping = random.uniform(0.0, 0.08)**2

        gym.set_actor_soft_materials(env, soft_actor, actor_soft_materials)

    # enable pd-control on rail joint to allow
    # control of the press using the GUI

    # shadow hand
    pose_shadow = gymapi.Transform()
    pose_shadow.r = gymapi.Quat(0.7071068, 0, 0, 0.7071068)
    pose_shadow.p = gymapi.Vec3(0.0, 0.18, 0.32)

    actor_shadow = gym.create_actor(env, asset_shadow, pose_shadow, "shadow", i, 1, segmentationId=11)
    actor_shadows.append(actor_shadow)
    gym.set_actor_dof_states(env, actor_shadow, dof_states[i], gymapi.STATE_ALL)


# Create 2 cameras in each environment,
camera_handles = [[]]
for i in range(num_envs):
    camera_handles.append([])
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 360
    camera_properties.height = 360

    h1 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(-0.5, 0.5, 0.1)
    camera_target = gymapi.Vec3(0, 0.1, 0)
    gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    camera_handles[i].append(h1)

    h2 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(0.5, 0.35, -0.1)
    camera_target = gymapi.Vec3(0, 0.1, 0)
    gym.set_camera_location(h2, envs[i], camera_position, camera_target)
    camera_handles[i].append(h2)

    h3 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(0.5, 0.4, -0.3)
    camera_target = gymapi.Vec3(0, 0.1, 0)
    gym.set_camera_location(h3, envs[i], camera_position, camera_target)
    camera_handles[i].append(h3)

    h4 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(0.5, 0, 0)
    camera_target = gymapi.Vec3(0, 0.1, 0)
    gym.set_camera_location(h4, envs[i], camera_position, camera_target)
    camera_handles[i].append(h4)
num_cam = len(camera_handles[0])

if not os.path.exists("graphics_images"):
    os.mkdir("graphics_images")
else:
    shutil.rmtree("graphics_images")
    os.mkdir("graphics_images")

frame_count = 0

# Point camera at environments
cam_pos = gymapi.Vec3(-4.0, 2.8, -1.2)
cam_target = gymapi.Vec3(0.0, 1.4, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
   
    if np.mod(frame_count, 5) == 0:
        dof_actions = np.random.uniform(low=-1, high=1, size=(num_envs, num_dofs))
    dof_positions += speeds * sim_params.dt * dof_actions

    # clone actor state in all of the environments
    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_shadows[i], dof_states[i], gymapi.STATE_POS)

    # update the viewer
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    if np.mod(frame_count, 5) == 0 and frame_count != 0:
        for i in range(num_envs):
            point_cloud = []
            for j in range(num_cam):
                pcd = pcd_utils.get_point_cloud(sim, envs[i], camera_handles[i][j], gym, mode='open3d')
                point_cloud.append(pcd)

                rgb_filename = "graphics_images/rgb_env%d_cam%d_frame%d.png" % (i, j, frame_count)
                gym.write_camera_image_to_file(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_COLOR, rgb_filename)

                depth_image = gym.get_camera_image(sim, envs[i], camera_handles[i][j], gymapi.IMAGE_DEPTH)

                # -inf implies no depth value, set it to zero. output will be black.
                depth_image[depth_image == -np.inf] = 0

                # clamp depth image to 10 meters to make output image human friendly
                depth_image[depth_image < -10] = -10

                # flip the direction so near-objects are light and far objects are dark
                normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))

                # Convert to a pillow image and write it to disk
                normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8), mode="L")
                normalized_depth_image.save("graphics_images/depth_env%d_cam%d_frame%d.jpg" % (i, j, frame_count))

            np.save("graphics_images/action_env%d_frame%d.npy" % (i, frame_count), dof_actions)
            # This line will merge points from different point cloud
            # Not necessary for visualization but useful for processing data into inputs of networks 
            merged_pcd = pcd_utils.merge_point_cloud(point_cloud)
            o3d.visualization.draw_geometries([merged_pcd])

    gym.draw_viewer(viewer, sim, False)
    frame_count = frame_count + 1
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    # camera = gym.get_viewer_camera_handle(viewer)
    # gym.get_camera_image(sim, camera, "IMAGE_COLOR")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
