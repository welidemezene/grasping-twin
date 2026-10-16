from lerobot.datasets import LeRobotDataset
import torch

ds = LeRobotDataset('lerobot/svla_so101_pickplace')


ep0 = ds.meta.episodes[0]
start = ep0['dataset_from_index']
end = ep0['dataset_to_index']

command = []
actual = []

for i in range(start,end):
    command.append(ds[i]['action'])
    actual.append(ds[i]['observation.state'])
states = torch.stack(actual)
action = torch.stack(command)

copycat = ((states - action) **2).mean()
print("copy-cat loss:", copycat)

print(states.shape)
print(action.shape)
print(states.dtype)

print(len(command),len(actual))

print(actual[0])
print(command[0])






