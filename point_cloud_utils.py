from isaacgym import gymapi
import open3d as o3d
import numpy as np
from PIL import Image as im


def get_point_cloud(sim, env, camera_handle, gym, cam_width=360, cam_height=360, mode='open3d'):
    """
    Create a o3d point cloud object from an issac gym environment with a specific camera
    Args:
        sim: issac gym simulation
        env: a specific environment in simulation
        camera_handle: the camera that we want to use
        gym: the gym object created by acquire_gym()
        cam_width: width of image
        cam_height: height of image
        mode: return mode of this function, either 'open3d' or 'numpy'. If using 'numpy', two arrays will be returned (points and colors).
    """
    assert mode in ['open3d', 'numpy'], 'Only two modes are allowed: open3d and numpy.'

    depth_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)

    color_image = gym.get_camera_image(
        sim, env, camera_handle, gymapi.IMAGE_COLOR).reshape([cam_width, cam_width, -1])  # RGBA
    rgb_image = im.fromarray(color_image).convert('RGB')
    rgb_image = np.asarray(rgb_image)

    # -inf implies no depth value, set it to zero. output will be black.
    depth_image[depth_image == -np.inf] = 0

    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, camera_handle)))
    
    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, env, camera_handle)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]
    points = []
    colors = []

    # Ignore any points which originate from ground plane or empty space
    depth_image[seg_buffer == 1] = -10001

    u = -(np.arange(0, cam_width) - cam_width/2) / cam_width
    v = (np.arange(0, cam_height) - cam_height/2) / cam_height
    idx = np.argwhere(seg_buffer != 1)

    d = depth_image[idx[:, 0], idx[:, 1]]
    l1 = np.expand_dims(d * u[idx[:, 1]] * fu, axis=1)
    l2 = np.expand_dims(d * v[idx[:, 0]] * fv, axis=1)
    points = np.concatenate((l1, l2, np.expand_dims(d, axis=1), np.ones_like(l1)), axis=1)
    points = np.asarray(points) * vinv
    points = points[:, [2, 0, 1]]

    colors = rgb_image[idx[:, 0], idx[:, 1]] / 255.

    # im_h = depth_image.shape[0]
    # im_w = depth_image.shape[1]

    # pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    # cam_pts_x = np.multiply(pix_x - centerU, depth_image / fu)
    # cam_pts_y = np.multiply(pix_y - centerV, depth_image / fv)
    # cam_pts_z = depth_image.copy()
    # cam_pts_x.shape = (im_h * im_w, 1)
    # cam_pts_y.shape = (im_h * im_w, 1)
    # cam_pts_z.shape = (im_h * im_w, 1)

    # idx = np.where(np.all(cam_pts_z >= -10000, axis=1))
 
    # rgb_pts_r = rgb_image[:, :, 0]
    # rgb_pts_g = rgb_image[:, :, 1]
    # rgb_pts_b = rgb_image[:, :, 2]
    # rgb_pts_r.shape = (im_h * im_w, 1)
    # rgb_pts_g.shape = (im_h * im_w, 1)
    # rgb_pts_b.shape = (im_h * im_w, 1)

    # cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    # print(cam_pts.shape)
    # cam_pts = cam_pts[idx]
    # rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)
    # rgb_pts = rgb_pts[idx]
    # print(cam_pts.shape)
    # print(rgb_pts.shape)

    if mode == 'open3d':
        scene_mesh = o3d.geometry.PointCloud()
        scene_mesh.points = o3d.utility.Vector3dVector(points)
        scene_mesh.colors = o3d.utility.Vector3dVector(colors)
        return scene_mesh
    else:
        return points, colors


def merge_point_cloud(pcds):
    points = np.concatenate([pcd.points for pcd in pcds])
    colors = np.concatenate([pcd.colors for pcd in pcds])

    scene_mesh = o3d.geometry.PointCloud()
    scene_mesh.points = o3d.utility.Vector3dVector(points)
    scene_mesh.colors = o3d.utility.Vector3dVector(colors)
    return scene_mesh

