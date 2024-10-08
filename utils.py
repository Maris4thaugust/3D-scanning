import open3d as o3d 
import numpy as np
import copy
import csv
import os

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def GetAxis(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    mid1 = (p1 + p2) / 2
    mid2 = (p2 + p3) / 2

    def perpendicular_bisector_3d(pt1, pt2):
        mid = (pt1 + pt2) / 2
        direction = np.cross(normal, pt2 - pt1)
        return mid, direction

    mid1, dir1 = perpendicular_bisector_3d(p1, p2)
    mid2, dir2 = perpendicular_bisector_3d(p2, p3)

    A = np.vstack([dir1, -dir2, normal]).T
    b = mid2 - mid1
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("The points do not form a valid circle in 3D space.")
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    center = mid1 + t[0] * dir1
    
    return np.array(center), normal

def GetAxisAngleMatrix(Angle,Axis,Center):
    """
    Compute the matrix for the rotation transformation around a fixed axis.
    """
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(Axis*(-Angle))

    InitialTransformationOrigin = np.eye(4)  
    InitialTransformationOrigin[:3, 3] = -Center

    InitialTransformationBack = np.eye(4)
    InitialTransformationBack[:3, 3] = Center

    InitialTransformationRot = np.eye(4)
    InitialTransformationRot[:3, :3] = R

    InitialTransformation = InitialTransformationBack @ InitialTransformationRot @ InitialTransformationOrigin

    return InitialTransformation

def Registrate(Target, Pcd, VoxelSize, InitialTransformation):
    if not Target.has_normals():
        Target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VoxelSize*4, max_nn=30))
    if not Pcd.has_normals():
        Pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=VoxelSize*4, max_nn=30))

    Threshold = VoxelSize * 0.4
    RefineRegis = o3d.pipelines.registration.registration_icp(
        Pcd, Target, Threshold, InitialTransformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))   
    Transformation = RefineRegis.transformation
    Fitness = RefineRegis.fitness
    Rmse = RefineRegis.inlier_rmse
    Correspondence = RefineRegis.correspondence_set
    
    return Fitness, Rmse, Correspondence, Transformation


def NoiseRemoving(Pcd,NbPoints: int =20, Radius: float=15, Iteration : int  =1):
    """
    Points with too few neighbors are considered outliers and removed.
    - NbPoints: Minimum number of neighbor of a point to be considerd inliner
    - Radius: Evaluate range
    - Iteration: Repeat the process after remove the previous outline
    """
    for _ in range(Iteration):
        _ , ind = Pcd.remove_radius_outlier(NbPoints, Radius)
        Pcd = Pcd.select_by_index(ind)   
    return Pcd

def getInfo_1():       
    X_1 = np.array([1.3e+02, -45, 5e+02])
    X_2 = np.array([70, 1.7e+02, 3.9e+02])
    X_3 = np.array([47, -1.2e+02, 3.3e+02])

    Box1 = o3d.io.read_point_cloud("FilterBox1.ply")
    Box1 = Box1.get_oriented_bounding_box()
    Center_1, Z_Axis_1 = GetAxis(X_1,X_2,X_3)
    return Center_1, Z_Axis_1,Box1

def getInfo_2():
    X_1 = np.array([1.4e+02, 99, 4.2e+02])
    X_2 = np.array([38, -1.2e+02, 2.6e+02])
    X_3 = np.array([1.4e+02, -75, 4.1e+02])

    Box2 = o3d.io.read_point_cloud("FilterBox1.ply")
    Box2 = Box2.get_oriented_bounding_box()
    Center_2, Z_Axis_2 = GetAxis(X_1,X_2,X_3)
    return Center_2, Z_Axis_2, Box2

def getInfo_3():
    X_1 = np.array([65, 1.6e+02, 3.1e+02])
    X_2 = np.array([52, -1.3e+02, 2.8e+02])
    X_3 = np.array([1.5e+02, 11, 3.9e+02])

    Box3 = o3d.io.read_point_cloud("FilterBox3.ply")
    Box3 = Box3.get_oriented_bounding_box()
    Center_3, Z_Axis_3 = GetAxis(X_1,X_2,X_3)
    return Center_3, Z_Axis_3, Box3

def AddValuesToCsv(Dir,FileName,Attempt, Fitness, RMSE, CorrespondentSet):
    fieldnames = ["Attempt","Fitness", "RMSE", "Correspondent Set"]
    
    if not os.path.exists(Dir):
        raise FileNotFoundError(f"The folder '{Dir}' does not exist.")
    file_path = os.path.join(Dir, FileName)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({"Attempt": Attempt,
                         "Fitness": Fitness, 
                         "RMSE": RMSE, 
                         "Correspondent Set": CorrespondentSet})

def SaveResult(file, dir, base_filename):
    if not os.path.exists(dir):
        raise FileNotFoundError(f"The folder '{dir}' does not exist.")
    
    # Determine the next available filename
    i = 1
    while True:
        filename = f"{base_filename}_{i}.txt"
        file_path = os.path.join(dir, filename)
        if not os.path.exists(file_path):
            break
        i += 1
    
    # Save the data to the file
    with open(file_path, 'w') as file:
        if isinstance(file, (list, dict)):
            file.write(str(file))
        else:
            file.write(file)


if __name__ == "__main__":
   pass