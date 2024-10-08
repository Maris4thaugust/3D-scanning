from utils import * 
import open3d as o3d 
import sys

def main():
    Resultfolder = "Result"
    FileName = "ExperimentResult.csv"
    fieldnames = ["Attempt","Fitness", "RMSE", "Correspondent Set"]
    with open(FileName, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    
    VoxelSize = 5
    
    # PointCloud taking from camera position 1 (1-12)
    Center_1, Z_Axis_1, Box1 = getInfo_1()
    TargetCloud_1 = o3d.io.read_point_cloud("3005/1.ply")
    TargetCloud_1.crop(Box1)
    TargetCloud_1.voxel_down_sample(VoxelSize)
    TargetCloud_1_Filterd = NoiseRemoving(TargetCloud_1,30,VoxelSize*0.5,10)
    o3d.io.write_point_cloud("TargetCloud_1.ply", TargetCloud_1_Filterd)
    
    # PointCloud taking from camera position 2 (13-24)
    Center_2, Z_Axis_2, Box2 = getInfo_2()
    TargetCloud_2 = o3d.io.read_point_cloud("3005/13.ply")
    TargetCloud_2.crop(Box2)
    TargetCloud_2.voxel_down_sample(VoxelSize)
    TargetCloud_2_Filterd = NoiseRemoving(TargetCloud_2,30,VoxelSize*0.5,10)
    o3d.io.write_point_cloud("TargetCloud_2.ply", TargetCloud_2_Filterd)
    
    # PointCloud taking from camera position 2 (25-36)
    Center_3, Z_Axis_3, Box3 = getInfo_3()
    TargetCloud_3 = o3d.io.read_point_cloud("3005/25.ply")
    TargetCloud_3.crop(Box3)
    TargetCloud_3.voxel_down_sample(VoxelSize)
    TargetCloud_3_Filterd = NoiseRemoving(TargetCloud_3,30,VoxelSize*0.5,10)
    o3d.io.write_point_cloud("TargetCloud_3.ply", TargetCloud_3_Filterd)

    Angle = np.radians(30)
    for i in range(1,35): 
        Pcd = o3d.io.read_point_cloud("3005/{}.ply".format(i+1))
        if i in range(1,12):
            Target = o3d.io.read_point_cloud(f"TargetCloud_1.ply")
            Pcd.crop(Box1)
            InitalMatrix = GetAxisAngleMatrix(Angle,Z_Axis_1,Center_1)
            SavedResult = "TargetCloud_1.ply" 
        elif i in range (13,24):
            Target = o3d.io.read_point_cloud(f"TargetCloud_2.ply")
            Pcd.crop(Box2)
            InitalMatrix = GetAxisAngleMatrix(Angle,Z_Axis_2,Center_2)
            SavedResult = "TargetCloud_2.ply"
        else:   
            Target = o3d.io.read_point_cloud(f"TargetCloud_3.ply")
            Pcd.crop(Box3)
            InitalMatrix = GetAxisAngleMatrix(Angle,Z_Axis_3,Center_3)
            SavedResult = "TargetCloud_3.ply"

        Pcd_Filter = NoiseRemoving(Pcd,30,VoxelSize*0.5,10)
        Fitness, Rmse, Correspondence, Transformation = Registrate(Target,Pcd_Filter,VoxelSize,InitalMatrix)       
        print(f"Attempt registrate {i}")                                                                 
        
        Angle  += np.radians(30)

        GlobalCloud = Target + Pcd_Filter.transform(Transformation)
        GlobalCloud.voxel_down_sample(VoxelSize)
        o3d.io.write_point_cloud(SavedResult, GlobalCloud)
        
        AddValuesToCsv(Dir = Resultfolder,
                       FileName= FileName,
                       Attempt=i,
                       Fitness=round(Fitness,4),
                       RMSE=round(Rmse,4),
                       CorrespondentSet=len(np.asarray(Correspondence)))     

if __name__ == "__main__":
    Target = o3d.io.read_point_cloud(f"TargetCloud_2.ply")
    o3d.visualization.draw_geometries([Target])
