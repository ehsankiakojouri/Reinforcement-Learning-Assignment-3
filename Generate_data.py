import gymnasium as gym
import numpy as np
import torch
import cv2


def main(total_episodes=200, time_steps=300, render=False):
    print(f"Generating data for env car_racing")
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    obs_data = torch.zeros(total_episodes*time_steps, 3, 64, 64)
    action_data = torch.zeros(total_episodes*time_steps, *env.action_space.sample().shape)
    for i_episode in range(total_episodes):
        print('-----')
        observation = env.reset()[0]
        terminated = False
        truncated = False
        for t in range(time_steps):
            if terminated or truncated:
                break
            print(np.max(observation))
            action = env.action_space.sample()
            obs_data[t+i_episode*time_steps] = torch.tensor(cv2.resize(observation/255., dsize=(64, 64)).T)
            action_data[t+i_episode*time_steps] = torch.tensor(action)

            observation, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

        print(f"Episode {i_episode} finished after {t + 1} timesteps")
        print(f"Current episode contains {sum(map(len, obs_data))} observations for dataset")

    print("Saving VAE dataset ...")
    torch.save(obs_data, './data/obs_data_car_racing.pth')
    torch.save(action_data, './data/action_data_car_racing.pth')
    print("Done")
    env.close()


if __name__ == "__main__":
    main()