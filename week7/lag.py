# WHY: a motor needs time to reach the commanded position; the robot AI must plan for that delay
# WHAT: plot the gripper's command (action) vs. its real position (state) for episode 0
# HOW: load data, collect the gripper number from every frame into two lists, plot both


from lerobot.datasets import LeRobotDataset
import matplotlib.pyplot as plt

ds = LeRobotDataset('lerobot/svla_so101_pickplace')

ep0 = ds.meta.episodes[0]
start = ep0['dataset_from_index']
end = ep0['dataset_to_index']
commands = []
actual = []

for i in range(start, end):
    commands.append(ds[i]['action'][5].item())
    actual.append(ds[i]['observation.state'][5].item())

plt.plot(commands, label='action (command)')
plt.plot(actual, label='state (real position)')
plt.xlabel('frame (30 per second)')
plt.ylabel('gripper position')
plt.legend()
plt.savefig('week7/lag.png')
print('saved')


