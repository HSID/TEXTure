import open3d as o3d
mesh_path = '/home/SENSETIME/xuehua/Projects/TEXTurePaper/shapes/BL_01.obj' 

mesh = o3d.io.read_triangle_mesh(mesh_path)

pcd = mesh.sample_points_poisson_disk(1400)

#o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
