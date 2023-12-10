import gymnasium as gym
# from gymnasium import wrappers

def main():
    # Create the CarRacing environment
    env = gym.make('CarRacing-v2', render_mode='human')

    # Wrap the environment to record a video of the user's actions
    # env = wrappers.Monitor(env, "./gym-results", force=True)

    # Reset the environment to get the initial state
    observation = env.reset()

    # Run the main loop
    while True:
        # Render the environment to visualize the car's movement
        env.render()

        # Prompt the user for continuous control actions (steering, acceleration, and brake)
        # action = get_user_action()
        #[steering, acceleration, brake]
        action = [-0.1, 0.5, 0.0]
        # Apply the user's action to the environment
        observation, reward, ter, trun, _ = env.step(action)

        # Check if the episode is done (either goal reached or max steps)
        if ter or trun:
            print("Episode finished.")
            break

    # Close the environment
    env.close()

def get_user_action():
    # Prompt the user for continuous control actions
    steering = float(input("Enter steering (-1.0 to 1.0): "))
    acceleration = float(input("Enter acceleration (0.0 to 1.0): "))
    brake = float(input("Enter brake (0.0 to 1.0): "))

    # Ensure the values are within valid ranges
    steering = max(-1.0, min(steering, 1.0))
    acceleration = max(0.0, min(acceleration, 1.0))
    brake = max(0.0, min(brake, 1.0))

    # Return the continuous control action
    return [steering, acceleration, brake]

if __name__ == "__main__":
    main()
