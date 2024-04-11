import numpy as np
import pandas as pd
import tqdm
import utilities as U
import open3d as o3d

pc = pd.read_csv('./annotation/infra.csv', delimiter=' ').values
infra_idx = set(pc[::100, -1].tolist())
print(infra_idx)
infra_idx_list = []
infra_center_list = []
infra_pc_list = []

for idx in tqdm.tqdm(infra_idx):
    infra_pc = pc[np.where(pc[:, -1] == idx)]
    # calculate center
    center = np.mean(infra_pc[:, :3], axis=0)
    print(center)
    infra_idx_list.append(idx)
    infra_center_list.append(center)
    infra_pc_list.append(infra_pc)
    U.pickle_dump((
        infra_idx_list, infra_center_list, infra_pc_list
    ), './data/infrastructure.pkl')

    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=4, resolution=20)
    # sphere.transform(U.create_se3_matrix_from_rpyxyz(0, 0, 0, *center))
    # U.vis(infra_pc, additional_geometry=sphere)
    # U.save_pcd(
    #     U.to_o3d_pointcloud(infra_pc),
    #     './infrastructure/{:06d}.pcd'.format(int(idx))
    # )
    # U.vis(infra_pc)
