import kaolin as kal
import torch
import numpy as np
from src.training.views_dataset import pose_euclidean2sphere
from src.training.views_dataset import from_camera_pose_to_T_wc


px = -1.0
py = 0.0
pz = 1.6
qx = 0.0
qy = 0.707107
qz = 0.0
qw = 0.707107
camera_pos = [px, py, pz, qx, qy, qz, qw]

T_wc = from_camera_pose_to_T_wc(camera_pos)
print(T_wc)

phi, theta, r, look_at, direction = pose_euclidean2sphere(torch.tensor(px), torch.tensor(py), torch.tensor(pz), torch.tensor(qx), torch.tensor(qy), torch.tensor(qz), torch.tensor(qw))


pos = torch.tensor([px, py, pz]).unsqueeze(0)
look_at = torch.tensor(look_at).unsqueeze(0)
direction = torch.tensor(direction).unsqueeze(0)
camera_proj = kal.render.camera.generate_transformation_matrix(pos.float(), look_at.float(), direction.float())

camera_proj = camera_proj[0].T

print(camera_proj.shape)
print(camera_proj)



