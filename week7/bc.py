from lerobot.datasets import LeRobotDataset
import torch
from torch import nn
import matplotlib.pyplot as plt 


ds = LeRobotDataset('lerobot/svla_so101_pickplace')


# ep0 = ds.meta.episodes[0]
# start = ep0['dataset_from_index']
# end = ep0['dataset_to_index']

# command = []
# actual = []

# for i in range(start,end):
#     command.append(ds[i]['action'])
#     actual.append(ds[i]['observation.state'])
# states = torch.stack(actual)
# action = torch.stack(command)

# ep1 = ds.meta.episodes[1]
# start1 = ep1['dataset_from_index']
# end1 = ep1['dataset_to_index']

# test_command = []
# test_actual = []

# for i in range(start1,end1):
#     test_command.append(ds[i]['action'])
#     test_actual.append(ds[i]['observation.state'])
# test_states = torch.stack(test_actual)
# test_action = torch.stack(test_command)


command = []
actual = []
for e in range(0, 40):
    epe = ds.meta.episodes[e]
    start = epe['dataset_from_index']
    end = epe['dataset_to_index']

    for i in range(start, end):
        command.append(ds[i]['action'])
        actual.append(ds[i]['observation.state'])
states = torch.stack(actual)
action = torch.stack(command)


test_command = []
test_actual = []
for e in range(40,50):
    epe = ds.meta.episodes[e]
    test_start = epe['dataset_from_index']
    test_end = epe['dataset_to_index']

    for i in range(test_start, test_end):
        test_command.append(ds[i]['action'])
        test_actual.append(ds[i]['observation.state'])
test_states = torch.stack(test_actual)
test_action = torch.stack(test_command)





copycat = ((states - action) **2).mean()
test_copycat = ((test_states - test_action) **2).mean()
print("copy-cat loss train:", copycat)
print("copy-cat loss test:", test_copycat)





print(states.shape)
print(action.shape)
print(states.dtype)

print(test_states.shape, test_action.shape)


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
final = model(states).detach()


with torch.no_grad():
    test_guesses = model(test_states)
    test_loss = ((test_guesses - test_action) ** 2).mean()
print("model loss ep1:", test_loss)



# plt.plot(final[:, 5], label='model guess')
# plt.plot(action[:, 5], label='human command')
# plt.plot(states[:, 5], label='real gripper (copy-cat)')



# plt.xlabel('frame (30 per second)')
# plt.ylabel('gripper position')
# plt.legend()
# plt.savefig('week7/bc_gripper.png')
# print('saved')

