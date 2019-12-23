import matplotlib.pyplot as plt
import numpy as np

from unityagents import UnityEnvironment


env_name = "env19" # Name of the Unity environment binary to launch
train_mode = True # Whether to run the environment in training or inference mode

env = UnityEnvironment(file_name=env_name)

# Examine environment parameters
print(str(env))

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.states[0]))

# Examine the observation space for the default brain
# for observation in env_info.observations:
#     print("Agent observations look like:")
#     if observation.shape[3] == 3:
#         plt.imshow(observation[0,:,:,:])
#     else:
#         plt.imshow(observation[0,:,:,0])


for episode in range(100):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        # if brain.action_space_type == 'continuous':
        #     env_info = env.step(np.random.randn(len(env_info.agents), 
        #                                         brain.action_space_size))[default_brain]
        # else:
        #     env_info = env.step(np.random.randint(0, brain.action_space_size, 
        #                                           size=(len(env_info.agents))))[default_brain]
        # episode_rewards += env_info.rewards[0]
         done = env_info.local_done[0]
        #done = True
    print("Total reward this episode: {}".format(episode_rewards))

env.close()