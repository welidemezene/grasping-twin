from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset('lerobot/svla_so101_pickplace')
print(ds.meta.features['action'])
print(ds.meta.features['observation.state'])

