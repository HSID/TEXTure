from src.training.views_dataset import FixedViewDataset

dataloader = FixedViewDataset('/storage0/data/panorama/room-1/cam_translations.txt', '/storage0/data/panorama/room-1/cubemap_rotations.txt', "cpu").dataloader()

for data in dataloader:
    print(data)


