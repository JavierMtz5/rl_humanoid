import gymnasium as gym
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["MUJOCO_GL"] = "egl"


# env = gym.make("Humanoid-v5", render_mode='rgb_array', width=640, height=480)
# env = gym.make("Humanoid-v5", render_mode='human')

# _ = env.reset()
# action = np.zeros(shape=(17,))
# for i in range(1000):
#     env.step(action)
#     env.render()
#     # frame = env.render()
#     # plt.imshow(frame)
#     # plt.axis("off")
#     # plt.pause(0.01)

# env.close()

# print('Test finished')

'=============================================================================================='

# def train() -> None:
#     """
#     Training loop
#     """
#     # Create environment and agent
#     environment: gym.Env = gym.make(GAME)
#     policy_kwargs = dict(activation_fn=ACTIVATION_FN, net_arch=NET_ARCH)
#     agent: algorithm.OnPolicyAlgorithm = A2C("MlpPolicy", environment, policy_kwargs=policy_kwargs,
#                                              n_steps=N_STEPS, learning_rate=LEARNING_RATE, gamma=GAMMA, verbose=1)

#     # Train the agent
#     callback_on_best: BaseCallback = StopTrainingOnRewardThreshold(reward_threshold=MAX_EPISODE_DURATION, verbose=1)
#     eval_callback: BaseCallback = EvalCallback(Monitor(environment), callback_on_new_best=callback_on_best,
#                                                eval_freq=EVAL_FREQ, n_eval_episodes=AVERAGING_WINDOW)
#     # Set huge number of steps because termination is based on the callback
#     agent.learn(int(1e10), callback=eval_callback)

#     # Save the agent
#     agent.save(MODEL_FILE)

'=============================================================================================='

env = gym.make('Humanoid-v5')

# Action space noise (for exploration)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=0, sigma=0.1 * torch.ones(n_actions))

# Create TD3 model
model = TD3(
    "MlpPolicy", 
    env, 
    action_noise=action_noise,
    learning_rate=3e-4, 
    buffer_size=100000,  
    batch_size=256,        
    gamma=0.99,            
    tau=0.005,             
    policy_delay=2,        
    train_freq=(1, "episode"),  
    gradient_steps=-1,      
    verbose=1
)

# callback_on_best: BaseCallback = StopTrainingOnRewardThreshold(reward_threshold=MAX_EPISODE_DURATION, verbose=1)
# eval_callback: BaseCallback = EvalCallback(Monitor(environment), callback_on_new_best=callback_on_best,
#                                            eval_freq=EVAL_FREQ, n_eval_episodes=AVERAGING_WINDOW)
model.learn(total_timesteps=10e6, log_interval=1000)

model.save("td3_humanoid_1")
del model

model = TD3.load("td3_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
