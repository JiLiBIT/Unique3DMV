import torch
import numpy as np


def read_colmap_points3D_bin(file_path):
    points = []
    with open(file_path, "rb") as f:
        num_points = np.fromfile(f, dtype=np.uint64, count=1)[0]
        for _ in range(num_points):
            point_id = np.fromfile(f, dtype=np.int64, count=1)[0]
            xyz = np.fromfile(f, dtype=np.float64, count=3)
            rgb = np.fromfile(f, dtype=np.uint8, count=3)
            error = np.fromfile(f, dtype=np.float64, count=1)[0]
            num_visible_images = int(np.fromfile(f, dtype=np.uint64, count=1)[0])
            image_ids_and_points2D = np.fromfile(f, dtype=np.int32, count=num_visible_images*2).reshape((num_visible_images, 2))
            points.append({
                "point_id": point_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "image_ids_and_points2D": image_ids_and_points2D
            })
    # 返回所有点的XYZ坐标
    return [point["xyz"] for point in points]

# 读取点云数据
points = read_colmap_points3D_bin("../data/snow_boy/sparse/0/points3D.bin")
# 假设 points 是一个 N x 3 的张量，代表点云
# transform_matrix 是一个 4 x 4 的张量，代表从COLMAP坐标系到目标坐标系的变换
transform_matrix = torch.tensor([[ 9.9968e-01, -1.4132e-02,  2.1146e-02,  1.4009e-01],
        [ 1.3317e-02,  9.9918e-01,  3.8191e-02,  1.4380e+00],
        [-2.1669e-02, -3.7897e-02,  9.9905e-01, -1.2661e+01],
        [-6.0904e-20, -5.2809e-18,  3.5682e-17,  1.0000e+00]],
       dtype=torch.float64)
# transform_matrix = torch.eye(4).to(torch.float64)
# 将点云转换为齐次坐标
print(len(points))
points = [torch.tensor(point, dtype=torch.float64) for point in points]
ones = torch.ones((len(points), 1))
points = torch.stack(points, dim=0)
points_homogeneous = torch.cat((points, ones), dim=1)

points_transformed_homogeneous = torch.matmul(transform_matrix, points_homogeneous.T).T
points_transformed = points_transformed_homogeneous[:, :3] / points_transformed_homogeneous[:, 3:]

# points_transformed_homogeneous = torch.matmul(points_homogeneous, transform_matrix)
# points_transformed = points_transformed_homogeneous[:, :3]

# 现在 points_transformed 是变换后的点云
points_transformed_np = points_transformed.numpy()
print(points_transformed_np)
output_file_path = "../data/snow_boy/sparse/0/points3D_transformed.ply"
# 创建并写入PLY文件
with open(output_file_path, 'w') as ply_file:
    # 写入PLY文件头部
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write(f"element vertex {len(points_transformed_np)}\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("end_header\n")
    
    # 写入点坐标
    for point in points_transformed_np:
        ply_file.write(f"{point[0]} {point[1]} {point[2]}\n")
