import gymnasium as gym
import torch
import cv2

def pre_process(obs, target_image_shape):
    # resize and reshape for giving to the VAE
    x = cv2.resize(obs, dsize=target_image_shape[1:], interpolation=cv2.INTER_NEAREST)
    x = x.astype('float32') / 255
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x)
    return x

def main(total_episodes=200, time_steps=300, render=False, image_size=(3, 64, 64), action_data_path='./data/action_data_car_racing.pth', obs_data_path='./data/obs_data_car_racing.pth'):
    print(f"Generating data for env car_racing")
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    obs_data = torch.zeros(total_episodes*time_steps, *image_size)
    action_data = torch.zeros(total_episodes*time_steps, *env.action_space.sample().shape)
    for i_episode in range(total_episodes):
        print('-----')
        observation, _ = env.reset()
        terminated = False
        truncated = False
        for t in range(time_steps):
            if terminated or truncated:
                break
            action = env.action_space.sample()
            obs_data[t+i_episode*time_steps] = pre_process(observation, image_size)
            action_data[t+i_episode*time_steps] = torch.tensor(action)
            observation, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

        print(f"Episode {i_episode} finished after {t + 1} timesteps")
        print(f"Current episode contains {sum(map(len, obs_data))} observations for dataset")

    print("Saving VAE dataset ...")
    torch.save(obs_data, obs_data_path)
    torch.save(action_data, action_data_path)
    print("Done")
    env.close()


if __name__ == "__main__":
    main()