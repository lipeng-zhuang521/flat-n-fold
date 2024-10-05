from scipy.spatial.transform import Rotation as R
import os
import ast
import csv
import numpy as np
import cv2
import open3d as o3d
import os
import copy

def find_nearest_timestamp(target_timestamp, timestamps):
    timestamps = np.array(timestamps, dtype=np.int64)
    idx = np.abs(timestamps - target_timestamp).argmin()
    return timestamps[idx]

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)
'''
def transform_point_cloud(point_cloud, rotation, translation):
    # Create rotation matrix
    r = R.from_euler('xyz', rotation)
    rotation_matrix = r.as_matrix()

    # Apply transformation
    points = np.asarray(point_cloud.points)
    transformed_points = np.dot(points, rotation_matrix.T) + translation

    # Create a new point cloud with transformed points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return transformed_pcd
'''



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


base_dir ='/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/david_datacollect/baxter/drag_red45/pointcloud_transformed'
pointcloud_folder = "/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/david_datacollect/baxter/drag_red45/pointcloud"
# pointcloud_top_folder = "/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/david_datacollect/baxter/fold_red45/pointcloud_top"
# pointcloud_side_folder = "/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/david2/407814/pointcloud_side"
    
pointcloud_files = sorted(os.listdir(pointcloud_folder))
# pointcloud_top_files = sorted(os.listdir(pointcloud_top_folder))
# pointcloud_side_files = sorted(os.listdir(pointcloud_side_folder))
    
# pointcloud_top_timestamps = [int(f.split('.')[0]) for f in pointcloud_top_files]
# pointcloud_side_timestamps = [int(f.split('.')[0]) for f in pointcloud_side_files]


for index in range(len(pointcloud_files)):
# for pointcloud_file in posintcloud_files:
    pointcloud_file = pointcloud_files[index]
    # pointcloud_top_file = pointcloud_top_files[index]
    # pointcloud_side_file = pointcloud_side_files[index]
# target_timestamp = int(pointcloud_file.split('.')[0])
        
# nearest_top_timestamp = find_nearest_timestamp(target_timestamp, pointcloud_top_timestamps)
# nearest_side_timestamp = find_nearest_timestamp(target_timestamp, pointcloud_side_timestamps)
        
    pointcloud_path = os.path.join(pointcloud_folder, pointcloud_file)
    # top_pointcloud_path = os.path.join(pointcloud_top_folder, f"{nearest_top_timestamp}.pcd")
    # side_pointcloud_path = os.path.join(pointcloud_side_folder, f"{nearest_side_timestamp}.pcd")
    # top_pointcloud_path = os.path.join(pointcloud_top_folder, pointcloud_top_file)
    # side_pointcloud_path = os.path.join(pointcloud_side_folder, pointcloud_side_file)
        
# print(f"Processing {pointcloud_file}:")
# print(f"Nearest top point cloud: {nearest_top_timestamp}.pcd")
# print(f"Nearest side point cloud: {nearest_side_timestamp}.pcd")

# def apply_aabb_cropping(pcd, min_bound, max_bound):
#     # Create an axis-aligned bounding box
#     aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
#     # Crop the point cloud
#     cropped_pcd = pcd.crop(aabb)
#     return cropped_pcd

# def execute_icp(source, target, init_pose=None, icp_type='point_to_point'):
#     """
#     Perform ICP registration.
#     - source: The source point cloud (to be transformed)
#     - target: The target point cloud (reference)
#     - init_pose: Initial guess of the transformation
#     - icp_type: 'point_to_point' or 'point_to_plane'
#     """
#     # Set initial alignment
#     if init_pose is None:
#         init_pose = np.identity(4)  # Identity matrix, no initial alignment
    
#     # Compute point to point ICP
#     if icp_type == 'point_to_point':
#         reg_p2p = o3d.pipelines.registration.registration_icp(
#             source, target, max_correspondence_distance=0.01, init=init_pose,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
#         )
#         transformation = reg_p2p.transformation
#     elif icp_type == 'point_to_plane':
#         # Ensure normals are estimated
#         source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#         target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
#         reg_p2plane = o3d.pipelines.registration.registration_icp(
#             source, target, max_correspondence_distance=0.02, init=init_pose,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
#         )
#         transformation = reg_p2plane.transformation
    
#     return transformation

# def draw_registration_result(source, target, transformation):
#     source_temp = copy.deepcopy(source)
#     target_temp = copy.deepcopy(target)
#     source_temp.paint_uniform_color([1, 0.706, 0])
#     target_temp.paint_uniform_color([0, 0.651, 0.929])
#     source_temp.transform(transformation)
#     o3d.visualization.draw_geometries([source_temp, target_temp])

# def read_tracker_data(file_path):
#     """Read the tracker data from a file and return the position and quaternion."""
#     positions = []
#     quaternions = []
#     grippers = []
#     with open(file_path, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         for i, row in enumerate(reader):
#             # if i % 20 == 0:  
#             pos = ast.literal_eval(row['position'])
#             ori = ast.literal_eval(row['orientation'])
#             gripper = int(row['gripper'])
#             positions.append(np.array([pos['x'], pos['y'], pos['z']]))
#             quaternions.append([
#                 round(ori['w'],3),
#                 round(ori['x'],3), 
#                 round(ori['y'],3),
#                 round(ori['z'],3)])
#             grippers.append(gripper)
#     return positions, quaternions, grippers

# tracker1_position, tracker1_quaternion, tracker1_gripper = read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/lipeng/napkin_pink30_5/bag/tracker_1/tracker_right_zed_modified.csv")
# tracker4_position, tracker4_quaternion, tracker4_gripper= read_tracker_data("/home/ubb/Documents/Baxter_isaac/ROS2/src/experiment_recorder/data/baxter/lipeng/napkin_pink30_5/bag/tracker_4/tracker_left_zed_modified.csv")

# Load point clouds
    front_pcd = load_point_cloud(pointcloud_path)
    # top_pcd = load_point_cloud(top_pointcloud_path)
    # side_pcd = load_point_cloud(side_pointcloud_path)

# Transform top and side point clouds
# top_rotation = [-0.03261978179216385, -1.5211526155471802, 0.04351048171520233]

# top_translation_original = [0.5958163738250732, -0.3476501405239105, 0.6767989993095398]
# top_translation = [-0.5958163738250732, 0.3476501405239105, 0.6767989993095398]
# top_transformed_pcd = transform_point_cloud(top_pcd, top_rotation, top_translation)

# side_rotation = [0.05642933025956154, 2.6661343574523926, 0.057520296424627304]

# side_translation_original = [-0.6605479717254639, 0.2682040333747864, 1.701134443283081]
# side_translation = [0.6605479717254639, -0.2682040333747864, 1.701134443283081]
# side_transformed_pcd = transform_point_cloud(side_pcd, side_rotation, side_translation_original)
# translation_zed = [0.12, -0.70, -1.52] # front original
# quaternion_zed =  [ 0.6837931,   0.12103661,  0.08102451, -0.71521635]  # [w, x, y, z] front original
# quaternion_zed =  [ 0.679,   0.142,  0.103, -0.713]
# translation_zed = [0.125, -0.71, -1.525] # front modified lp

    translation_zed = [0.139, -0.68, -1.525] # front modified leo final
    # quaternion_zed =  [0.689, 0.128, 0.086, -0.708] # front modified leo
    quaternion_zed =  [0.690, 0.126, 0.084, -0.708]  # front modified leo final

    # translation_zedtop = [-0.53, -0.31, -0.84]
    # quaternion_zedtop = [0.65430574, 0.61371309, 0.32184954, 0.30185888]
    # translation_zedtop = [-0.508, -0.32, -0.84] # top modified lp
    # quaternion_zedtop = [0.65430574, 0.61371309, 0.32184954, 0.30185888] # top original

    translation_zedtop = [-0.508, -0.313, -0.84] # top modified leo final
    # quaternion_zedtop = [0.659, 0.608, 0.332, 0.293] # top modified leo
    quaternion_zedtop = [0.652, 0.615, 0.329, 0.297] # top modified leo final

    # translation_zedside = [0.684, -0.54, -0.32]
    # quaternion_zedside =  [-0.211,  0.533,  0.683, -0.452]
    #translation_zedside = [1.00, -0.988, -0.005] # side original
    #quaternion_zedside =  [0.4121696,  -0.6735454, -0.5830692,  0.1910054] # side original
    # translation_zedside = [1.03, -1.035, -0.01]
    # translation_zedside = [0.98, -1.045, -0.0075] # side modified lp
    translation_zedside = [1.0, -1.030, -0.01] # side modified leo final
    # quaternion_zedside =  [0.392,  -0.691, -0.578,  0.195] # side modified leo
    # quaternion_zedside =  [  0.389, -0.683, -0.587,  0.193] 
    quaternion_zedside =  [0.385, -0.687, -0.588,  0.186]  # side modified leo final
    
    translation_zedtop_ryd = [-0.471, -0.308, -0.81] # top modified leo
    quaternion_zedtop_ryd = [0.655,0.611, 0.305, 0.323] # top modified leo

    translation_zedside_ryd = [0.99, -1.03, 0.05]  # side modified leo
    quaternion_zedside_ryd =  [0.408, -0.662, -0.602,  0.182]
#

# x_min = -0.493813
# x_max = 0.527173
# y_min = -0.534935
# y_max = 0.488354
# z_min = 0.523
# z_max = 1.45

# min_bound = np.array([x_min, y_min, z_min])
# max_bound = np.array([x_max, y_max, z_max])

# cropped_pcd_zed = apply_aabb_cropping(front_pcd, min_bound, max_bound)
# cropped_pcd_top = apply_aabb_cropping(top_pcd, min_bound, max_bound)
# cropped_pcd_side = apply_aabb_cropping(side_pcd, min_bound, max_bound)

    transformed_pcd_zed = transform_point_cloud(front_pcd, translation_zed, quaternion_zed)
    # transformed_pcd_zedtop = transform_point_cloud(top_pcd, translation_zedtop_ryd, quaternion_zedtop_ryd)
    # transformed_pcd_zedside = transform_point_cloud(side_pcd, translation_zedside_ryd, quaternion_zedside_ryd)


# transformed_pcd_zed = transform_point_cloud(cropped_pcd_zed, translation_zed, quaternion_zed)
# transformed_pcd_zedtop = transform_point_cloud(cropped_pcd_top, translation_zedtop, quaternion_zedtop)
# transformed_pcd_zedside = transform_point_cloud(cropped_pcd_side, translation_zedside, quaternion_zedside)

# initial_transform = np.eye(4)  # Assuming some initial transformation or use the identity matrix
# aligned_transformation = execute_icp(transformed_pcd_zedtop, transformed_pcd_zed, init_pose=initial_transform, icp_type='point_to_point')

# Apply the transformation to align the side point cloud to the front
# transformed_pcd_zedtop.transform(aligned_transformation)

# draw_registration_result(transformed_pcd_zedtop, transformed_pcd_zed, aligned_transformation)

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=1920, height=1080)
# vis.add_geometry(transformed_pcd_zed)

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=1920, height=1080)

# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
# colors = {0: [1, 0, 0], 1: [128/255, 0, 128/255]} 
# color_tracker3 = colors[tracker1_gripper[index]]
# color_tracker4 = colors[tracker4_gripper[index]]

# sphere3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
# sphere3.translate(tracker1_position[index]) 
# sphere3.paint_uniform_color(color_tracker3) 
# vis.add_geometry(sphere3)
# frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# rotation_matrix3 = o3d.geometry.get_rotation_matrix_from_quaternion(tracker1_quaternion[index])
# frame3.rotate(rotation_matrix3, center=[0, 0, 0])
# frame3.translate(tracker1_position[index])
# vis.add_geometry(frame3)

# sphere4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=20)
# sphere4.translate(tracker4_position[index]) 
# sphere4.paint_uniform_color(color_tracker4)
# vis.add_geometry(sphere4)
# frame4 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
# rotation_matrix4 = o3d.geometry.get_rotation_matrix_from_quaternion(tracker4_quaternion[index])
# frame4.rotate(rotation_matrix4, center=[0, 0, 0])
# frame4.translate(tracker4_position[index])
# vis.add_geometry(frame4)
# vis.add_geometry(origin)
# # set_camera_view_side(vis)
# vis.run()
# vis.destroy_window()
# o3d.visualization.draw_geometries([transformed_pcd_zedside,origin])

# Save or visualize the merged point cloud
# o3d.io.write_point_cloud("merged_point_cloud.pcd", merged_pcd)
# o3d.visualization.draw_geometries([transformed_pcd_zed,transformed_pcd_zedside,origin])
# vis.add_geometry(transformed_pcd_zedside)
# vis.add_geometry(transformed_pcd_zed)
# # vis.add_geometry(transformed_pcd_zedtop)


# vis.run()
# vis.destroy_window()
    # merged_point_cloud2 = transformed_pcd_zed + transformed_pcd_zedtop
    # merged_point_cloud3 = transformed_pcd_zed + transformed_pcd_zedtop + transformed_pcd_zedside
    pointcloud_front_world = transformed_pcd_zed
    # pointcloud_top_world = transformed_pcd_zedtop
    # pointcloud_side_world = transformed_pcd_zedside
    output_path = os.path.join(base_dir, f"{pointcloud_file}")
    o3d.io.write_point_cloud(output_path, pointcloud_front_world)

