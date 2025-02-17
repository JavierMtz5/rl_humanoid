import gymnasium as gym
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os
os.environ["MUJOCO_GL"] = "egl"

class StopTrainingOnEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, save_freq: int, verbose=1):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.save_freq = save_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if the episode has ended
        dones = self.locals.get("dones", [False])
        if any(dones):
            self.episode_count += 1
            # Save model every 'save_freq' episodes
            if self.episode_count % self.save_freq == 0:
                checkpoint_path = f"{os.getcwd()}/checkpoints/td3_checkpoint_ep_{self.episode_count}"
                self.model.save(checkpoint_path)
                if self.verbose:
                    print(f"Checkpoint created: {checkpoint_path}")

        # Stop training if max episodes reached
        return self.episode_count < self.max_episodes

class TD3Agent:
    def __init__(self, gym_env: str):
        self.env = gym.make(gym_env, render_mode='human')

    def train_td3(self, max_episodes: int, max_timesteps: int, save_freq: int, log_interval: int) -> None:
        """Trains the TD3 agent on the given environment"""
        # Initialize noise process
        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=0, sigma=0.1 * torch.ones(n_actions))

        # Create TD3 model using SB3
        model = TD3(
            "MlpPolicy", 
            self.env, 
            action_noise=action_noise,
            learning_rate=3e-4, 
            buffer_size=int(1e6),  
            batch_size=256,        
            gamma=0.99,            
            tau=0.005,             
            policy_delay=2,        
            train_freq=(1, "episode"),  
            gradient_steps=-1,      
            verbose=1
        )

        callback = StopTrainingOnEpisodes(max_episodes=max_episodes, save_freq=save_freq)
        model.learn(total_timesteps=max_timesteps, log_interval=log_interval, callback=callback)

        model.save("td3_humanoid")
        print('Agent was trained and model was saved')

    def evaluate_model(self, checkpoint_path: str) -> None:
        """Evaluates the performance of the agent"""
        model = TD3.load(checkpoint_path)

        obs, _ = self.env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, _, dones, _, _ = self.env.step(action)
            self.env.render()
            if dones:
                obs, _ = self.env.reset()


def main():
    td3_agent = TD3Agent(gym_env='Humanoid-v5')
    td3_agent.train_td3(
        max_episodes=10000,
        max_timesteps=1.5e7,
        save_freq=500,
        log_interval=100
    )
    # model_path = os.path.join(os.getcwd(), 'checkpoints', 'td3_checkpoint_ep_4500.zip')
    # td3_agent.evaluate_model(model_path)


if __name__ == '__main__':
    main()
