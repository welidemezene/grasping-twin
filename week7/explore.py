from lerobot.datasets import LeRobotDataset
import torch
from PIL import Image

ds = LeRobotDataset('lerobot/svla_so101_pickplace')

ep0 = ds.meta.episodes[0]
print(ep0)
start = ep0['dataset_from_index']
end = ep0['dataset_to_index']
print(start, end, end - start)
frame = ds[start]
print(frame['observation.state'])

img = ds[0]['observation.images.up']
print(img.dtype, img.shape, img.min(), img.max())

arr = (img.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
Image.fromarray(arr).save('test.jpg')
