import numpy as np
import matplotlib.pyplot as plt

# plotting from the actions_and_joint_positions.npz file
data = np.load("actions_and_joint_positions.npz")
actions = data['actions']
joint_positions = data['joint_positions']

data2 = np.load("actions_and_joint_positions2.npz")
actions2 = data2['actions']
joint_positions2 = data2['joint_positions']

actions = np.array(actions)
actions2 = np.array(actions2)
joint_position = np.array(joint_positions)
joint_position2 = np.array(joint_positions2)

leg_names = ['Front Right', 'Front Left', 'Rear Right', 'Rear Left']
joint_indices = {
    'Front Right':  [0, 1, 2],
    'Front Left': [3, 4, 5],
    'Rear Right':   [6, 7, 8],
    'Rear Left':  [9, 10, 11],
}

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for i, leg in enumerate(leg_names):
    ax = axs[i]
    for j, joint_idx in enumerate(joint_indices[leg]):
        # ax.plot(actions[:, joint_idx], label=f'Joint {j+1}')
        ax.plot(joint_position[:, joint_idx], label=f'Joint {j+1}')
        ax.plot(joint_position2[:, joint_idx], label=f'Joint {j+1}')
        # x limit
        # ax.set_xlim(0, 100)

    ax.set_title(f'{leg} Leg')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Action Value')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

    