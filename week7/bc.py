from lerobot.datasets import LeRobotDataset
import torch
from torch import nn

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


total = 0

model = nn.Linear(6,6)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

print(optimizer)

for p in model.parameters():
    total += p.numel()
print(total)

guesses = model(states)
print(guesses.shape)


untrained = ((guesses - action) ** 2).mean()
print("untrained model loss:" , untrained)

for step in range(3000):
    guesses = model(states)

    loss = ((guesses- action) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())








