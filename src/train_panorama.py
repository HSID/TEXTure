from pathlib import Path


if __name__ == "__main__":
    folderpath = Path("/storage0/data/panorama/room-1/0802-din-512")
    views = [str(i) for i in range(25)]

    rgb_paths = []
    depth_paths = []
    transforms = []
    for view in views:
        camera_poses = np.loadtxt(folderpath / view / "real_pose.txt", delimiter=" ")
        for i in range(6):
            rgb_paths.append(folderpath / view / "rgb_model" / (str(i + 1) + ".png"))
            depth_paths.append(folderpath / view / "depth" / (str(i + 1) + ".png"))
            camera_pose = camera_poses[i]
            T_wc = from_camera_pose_to_T_wc(camera_pose)
            transforms.append(T_wc)

    pcd = rgbd2pcd(rgb_paths, depth_paths, transforms, 512, 512, 256, 256, 256, 256)
    mesh = pcd2mesh(pcd, 0.03)

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
