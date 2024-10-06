import numpy as np
import open3d as o3d
import os
import ast
import time
import csv
import threading
import argparse

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to a 4x4 rotation matrix."""
    w, x, y, z = quat
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
 
    return np.array([[1.0 - (tyy + tzz), txy - twz, txz + twy, 0],
                     [txy + twz, 1.0 - (txx + tzz), tyz - twx, 0],
                     [txz - twy, tyz + twx, 1.0 - (txx + tyy), 0],
                     [0, 0, 0, 1]])

def transform_point_cloud(pcd, translation, quaternion):
    """Apply transformation defined by a translation and quaternion rotation to a point cloud."""
    # Create the transformation matrix
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    rotation_matrix_inversed = np.linalg.inv(rotation_matrix)
    #print(rotation_matrix)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
    transformation_matrix[:3, 3] = translation

    # transformation_matrix = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] @ transformation_matrix
 
    # Transform the point cloud
    transformed_pcd = pcd.transform(transformation_matrix) # left multiplication
    #print(np.asarray(transformed_pcd.points))

    return transformed_pcd
 
def read_tracker_data(file_path):
    """Read the tracker data from a file and return the position and quaternion."""
    positions = []
    quaternions = []
    grippers = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # if i % 20 == 0:  
            pos = ast.literal_eval(row['position'])
            ori = ast.literal_eval(row['orientation'])
            gripper = int(row['gripper'])
            positions.append(np.array([pos['x'], pos['y'], pos['z']]))
            quaternions.append([
                round(ori['w'],3),
                round(ori['x'],3), 
                round(ori['y'],3),
                round(ori['z'],3)])
            grippers.append(gripper)
    return positions, quaternions, grippers

def animate_trajectory(dataset, vis, positions, quaternions, grippers, color_map):
    spheres = []
    frames = []

    for position, quaternion, gripper in zip(positions, quaternions,grippers):
        color = color_map['open'] if gripper == 1 else color_map['closed']
        if dataset == "zed":
            set_camera_view(vis)
        elif dataset == "zed_top": 
            set_camera_view_top(vis)
        elif dataset == "zed_side":
            set_camera_view_side(vis)
        # set_camera_view_top(vis)
        # set_camera_view(vis)
        # Create sphere at the current position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
        sphere.translate(position)  # Move the sphere to the correct position
        sphere.paint_uniform_color(color)  # Color the sphere
        spheres.append(sphere)  # Store the sphereZ

        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        frame.rotate(rotation_matrix, center=[0, 0, 0])
        frame.translate(position)
        frames.append(frame)

    return spheres, frames

def set_camera_view(vis):
    control = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1070.1000, fy=939.8100, cx=960.0, cy=549.6520)
    camera_parameters.extrinsic = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 5],  # Moves the camera back to ensure a good initial view
        [0, 0, 0, 1]
    ])
    control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

def set_camera_view_top(vis):
    control = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1060.4000, fy=1060.0100, cx=946.0600, cy=496.6630)
    camera_parameters.extrinsic = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 5],  # Moves the camera back to ensure a good initial view
        [0, 0, 0, 1]
    ])
    control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

def set_camera_view_side(vis):
    control = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1060.4000, fy=1065.9900, cx=970.8000, cy=547.1950)
    camera_parameters.extrinsic = np.array([
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [-1, 0, 0, 5],  # Moves the camera back to ensure a good initial view
        [0, 0, 0, 1]
    ])
    control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)
def set_camera_view_top_c(vis):
    control = vis.get_view_control()
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic.set_intrinsics(width=1920, height=1080, fx=1060.4000, fy=1060.0100, cx=946.0600, cy=496.6630)
    camera_parameters.extrinsic = np.array([
        [0, 0, -1, 0],   # New X is the original Z
        [0, -1, 0, 0],  # Y remains flipped
        [1, 0, 0, 5],  # New Z is the negative original X, maintain the translation
        [0, 0, 0, 1]
    ])
    control.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)

# def run_visualization_close(vis, duration):
#     """
#     Runs the visualization for a specified amount of time then automatically closes the window.
#     """
#     start_time = time.time()
#     while True:
#         vis.poll_events()  # Update the visualizer with any events
#         vis.update_renderer()  # Redraw the visualizer
#         if time.time() - start_time > duration:  # Check if the duration has elapsed
#             break
#     vis.destroy_window()

def run_visualization(vis, duration):
    """Runs the visualizer for a specified amount of time."""
    start_time = time.time()
    while time.time() - start_time < duration:
        vis.poll_events()
        vis.update_renderer()
    vis.close()

def calculate_extended_point(position, quaternion, extensin_length_x, extension_length_y, extension_length_z):
    R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    extension_vector = np.array([extensin_length_x, extension_length_y, extension_length_z])
    extended_vector = np.dot(R, extension_vector)
    extended_point = np.array(position) + extended_vector
    return extended_point



pcd_directory_path_zed = "/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/pointcloud"
pcd_directory_path_zedtop = "/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/pointcloud_top"
pcd_directory_path_zedside ="/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/pointcloud_side"


# # Read tracker data
# tracker3_position, tracker3_quaternion = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/691427/bag/tracker_3/pose_converted_times_zed.csv")
# tracker4_position, tracker4_quaternion = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/691427/bag/tracker_4/pose_converted_times_zed.csv")
tracker3_position, tracker3_quaternion, tracker3_gripper = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_3/tracker3_zed_modified.csv")
tracker4_position, tracker4_quaternion, tracker4_gripper = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_4/tracker4_zed_modified.csv")
tracker3_positiontop, tracker3_quaterniontop, tracker3_grippertop = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_3/tracker3_zedtop_modified.csv")
tracker4_positiontop, tracker4_quaterniontop, tracker4_grippertop = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_4/tracker4_zedtop_modified.csv")
tracker3_positionside, tracker3_quaternionside, tracker3_gripperside = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_3/tracker3_zedside_modified.csv")
tracker4_positionside, tracker4_quaternionside, tracker4_gripperside = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/193426/bag/tracker_4/tracker4_zedside_modified.csv")

color_map_tracker3 = {'open': [128/255, 0, 128/255], 'closed': [1, 0, 0]}  # Purple when open, red otherwise
color_map_tracker4 = {'open': [128/255, 0, 128/255], 'closed': [0, 1, 0]}

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)

def main(dataset):
    if dataset == "zed":
        directory_path = pcd_directory_path_zed
    elif dataset == "zed_top":
        directory_path = pcd_directory_path_zedtop
    elif dataset == "zed_side":
        directory_path = pcd_directory_path_zedside
    else:
        raise ValueError("Invalid dataset specified")
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if filename.endswith(".pcd"):
            file_path = os.path.join(directory_path, filename)
            pcd_zed = o3d.io.read_point_cloud(file_path)
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            # Clear previous geometries and add new ones
            vis.clear_geometries()
            if dataset == "zed":
                translation_zed = [0.125, -0.71, -1.525]
                quaternion_zed =  [ 0.6837931,   0.12103661,  0.08102451, -0.71521635]  # [w, x, y, z]
                transformed_pcd_zed = transform_point_cloud(pcd_zed, translation_zed, quaternion_zed)
                vis.add_geometry(transformed_pcd_zed)
            elif dataset == "zed_top":
                translation_zedtop = [-0.508, -0.32, -0.84]
                quaternion_zedtop = [0.65430574, 0.61371309, 0.32184954, 0.30185888]
                transformed_pcd_zedtop = transform_point_cloud(pcd_zed, translation_zedtop, quaternion_zedtop)
                vis.add_geometry(transformed_pcd_zedtop)
            elif dataset == "zed_side":
                translation_zedside = [0.98, -1.045, -0.0075]
                quaternion_zedside =  [0.4121696,  -0.6735454, -0.5830692,  0.1910054]
                transformed_pcd_zed_side = transform_point_cloud(pcd_zed, translation_zedside, quaternion_zedside)
                vis.add_geometry(transformed_pcd_zed_side)
            # Add trackers
            if dataset == "zed":
                if i < len(tracker3_position):
                    tracker3_sphere, tracker3_frames = animate_trajectory(dataset, vis, [tracker3_position[i]], [tracker3_quaternion[i]], [tracker3_gripper[i]], color_map_tracker3)
                    tracker4_sphere, tracker4_frames = animate_trajectory(dataset, vis, [tracker4_position[i]], [tracker4_quaternion[i]], [tracker4_gripper[i]], color_map_tracker4)
                    for sphere in tracker3_sphere + tracker4_sphere:
                        vis.add_geometry(sphere)
                    for frame in tracker3_frames + tracker4_frames:
                        vis.add_geometry(frame)
            elif dataset == "zed_top":
                if i < len(tracker3_position):
                    tracker3_spheretop, tracker3_framestop = animate_trajectory(dataset, vis, [tracker3_positiontop[i]], [tracker3_quaterniontop[i]], [tracker3_grippertop[i]], color_map_tracker3)
                    tracker4_spheretop, tracker4_framestop = animate_trajectory(dataset, vis, [tracker4_positiontop[i]], [tracker4_quaterniontop[i]], [tracker4_grippertop[i]], color_map_tracker4)
                    for sphere in tracker3_spheretop + tracker4_spheretop:
                        vis.add_geometry(sphere)
                    for frame in tracker3_framestop + tracker4_framestop:
                        vis.add_geometry(frame)
            elif dataset == "zed_side":
                if i < len(tracker3_position):
                    tracker3_sphereside, tracker3_framesside = animate_trajectory(dataset, vis, [tracker3_positionside[i]], [tracker3_quaternionside[i]], [tracker3_gripperside[i]], color_map_tracker3)
                    tracker4_sphereside, tracker4_framesside = animate_trajectory(dataset, vis, [tracker4_positionside[i]], [tracker4_quaternionside[i]], [tracker4_gripperside[i]], color_map_tracker4)
                    for sphere in tracker3_sphereside + tracker4_sphereside:
                        vis.add_geometry(sphere)
                    for frame in tracker3_framesside + tracker4_framesside:
                        vis.add_geometry(frame)
            # Add highlight point
            if dataset == "zed" and i < len(tracker3_gripper):
                if tracker3_gripper[i] == 1:
                    extended_point = calculate_extended_point(tracker3_position[i], tracker3_quaternion[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zed.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index] 
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])  # Yellow
                    vis.add_geometry(highlight_sphere)               

                if tracker4_gripper[i] == 1:
                    extended_point = calculate_extended_point(tracker4_position[i], tracker4_quaternion[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zed.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index]
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])  # Yellow
                    vis.add_geometry(highlight_sphere)

            if dataset == "zed_top" and i < len(tracker3_grippertop):
                if tracker3_grippertop[i] == 1:
                    extended_point = calculate_extended_point(tracker3_positiontop[i], tracker3_quaterniontop[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zedtop.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index]
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])
                    vis.add_geometry(highlight_sphere)               

                if tracker4_grippertop[i] == 1:
                    extended_point = calculate_extended_point(tracker4_positiontop[i], tracker4_quaterniontop[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zedtop.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index]
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])
                    vis.add_geometry(highlight_sphere)

            if dataset == "zed_side" and i < len(tracker3_gripperside):
                if tracker3_gripperside[i] == 1:
                    extended_point = calculate_extended_point(tracker3_positionside[i], tracker3_quaternionside[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zed_side.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index] 
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])
                    vis.add_geometry(highlight_sphere)               

                if tracker4_gripperside[i] == 1:
                    extended_point = calculate_extended_point(tracker4_positionside[i], tracker4_quaternionside[i], -0.005, 0.015, 0.11)
                    pcd_points = np.asarray(transformed_pcd_zed_side.points)
                    distances = np.linalg.norm(pcd_points - extended_point, axis=1)
                    closest_point_index = np.argmin(distances)
                    closest_point = pcd_points[closest_point_index]
                    highlight_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
                    highlight_sphere.translate(closest_point)
                    highlight_sphere.paint_uniform_color([1, 1, 0])
                    vis.add_geometry(highlight_sphere)

            # Add origin coordinate frame
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
            vis.add_geometry(origin)
            if dataset == "zed":
                set_camera_view(vis)
            elif dataset == "zed_top":
                set_camera_view_top(vis)
            elif dataset == "zed_side":
                set_camera_view_side(vis)
            run_visualization(vis, 0.1)
            # vis.run()
            # time.sleep(0.5)
            # vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run point cloud visualization for zed or zed_top datasets.")
    parser.add_argument("dataset", type=str, choices=['zed', 'zed_top','zed_side'], help="Specify 'zed' or 'zed_top' or 'zed_side' to select the dataset.")
    args = parser.parse_args()
    main(args.dataset)