import concurrent.futures
import cv2
import cma
import gymnasium as gym
import torch
from VAE import VAE
from RNN import RNN
from Controller import CONTROLLER

# utils
def load_parameters(params, controller):
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

def unflatten_parameters(params, example, device):
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened
class RolloutGenerator(object):
    ''' Generate a single rollout '''
    def __init__(self, v_path, m_path, render=False, device=None):
        ''' Initialize a RolloutGenerator object.

        Parameters
        ----------
        v_path : Union[str, pathlib.Path]
            Path to a saved V model (VAE) checkpoint.
        m_path : Union[str, pathlib.Path]
            Path to a saved M model (memory: LSTM) checkpoint.
        device : Optional[str]
            The device to onto which to push tensors, or None to use CUDA where available.
        '''
        self.vae = VAE().to(device)
        checkpoint_v = torch.load(v_path)
        self.vae.load_state_dict(checkpoint_v)

        self.mdnrnn = RNN().to(device)
        checkpoint_m = torch.load(m_path)
        self.mdnrnn.load_state_dict(checkpoint_m)

        self.control = CONTROLLER().to(device)
        render_mode = 'human' if render else 'rgb_array'
        self.env = gym.make('CarRacing-v2', render_mode=render_mode)
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'


    def pre_process(self, obs):
        ''' Preprocess an input observation.

        Parameters
        ----------
        obs : numpy.ndarray, shape=(R, C, 3)
            The observation (image).
        Returns
        -------
        x : torch.Tensor, shape=(1, 3, 64, 64)
            The cropped-and-resized observation.
        '''
        x = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        x = x.astype('float32') / 255
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).view(-1, 3, 64, 64)
        return x

    def rollout(self, params, num_episodes=1000):
        ''' Perform a single rollout.

        Parameters
        ----------
        params : dict
            controller fully connected layer weights and biases
        Returns
        -------
        total_reward * -1 : float
            The negative reward accumulated during the rollout
        '''
        load_parameters(params, self.control)

        obs = self.env.reset()[0]
        self.env.render()
        (h, c) = [torch.zeros(1, 1, 256).to(self.device) for _ in range(2)]
        total_reward = 0

        for step in range(num_episodes):
            obs = self.pre_process(obs).to(self.device)
            # using mean as z vector
            z = self.vae.encoder(obs)
            a = self.control.get_action(z, h)
            obs, reward, terminated, truncated, _ = self.env.step(a)

            a = torch.from_numpy(a).view(1, 1, 3).to(self.device)
            x = torch.cat((z.view(1, 1, 1024), a), 2)
            _, (h, c) = self.mdnrnn.rnn(x, (h, c))

            total_reward += reward
            if terminated or truncated:
                break
        return -1 * total_reward

def fitness(rolloutGenerator, solutions, n_rollouts, num_episodes):
    rewards = [0] * len(solutions)
    for idx, solution in enumerate(solutions):  # loss calculation
        # load params into controller
        avg_cumulated_rewards = 0
        for i in range(n_rollouts):
            avg_cumulated_rewards += rolloutGenerator.rollout(solution, num_episodes) / n_rollouts
        rewards[idx] = avg_cumulated_rewards
    return rewards


def main():
    v_path = 'models/obs_data_car_racing.VAE_model'
    m_path = './models/rnn_weights.pth'
    c_save_path = 'models/cmaes_train_controller.pth'
    num_workers = 6 # number of CPU cores


    obs_index = 2
    population_each_rollout_episodes = 300
    best_each_rollout_episodes = 1000
    interval_rollouts = 1024
    n_rollouts = 16
    pop_size = 64
    target = 930
    log_interval = 1
    weight_save_interval = 5
    sigma = 0.1
    # dummy controller to initialize parameters
    controller = CONTROLLER()
    cur_best = None
    generation= 0
    eval_best_interval = 25 # according to the paper

    parameters = controller.parameters()
    flatten_parameters = torch.cat([p.detach().view(-1) for p in parameters], dim=0).numpy()
    es = cma.CMAEvolutionStrategy(flatten_parameters, sigma, {'popsize': pop_size})

    human_r_gen = RolloutGenerator(v_path, m_path, render=True)
    r_gen = RolloutGenerator(v_path, m_path, render=False)
    while not es.stop():
        generation += 1
        print(f"Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s] -- generation: {generation} -- cur_best: {cur_best}")
        if cur_best is not None and (-cur_best) > target:
            print("Already better than target, breaking...")
            break

        r_list = [0] * pop_size
        solutions = es.ask()
        with torch.no_grad():
            r_list = fitness(r_gen, solutions, n_rollouts, population_each_rollout_episodes)

        es.tell(solutions, r_list)
        es.disp()
        best_solution = es.result.xbest
        best = es.result.fbest
        best_std = es.result.xbest.std
        if generation> 0 and generation%eval_best_interval == 0:
            with torch.no_grad():
                r_list = fitness(human_r_gen, best_solution, interval_rollouts, best_each_rollout_episodes)
            # best comparrison is > because working with negative of rewards
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"Saving new best with value {-cur_best}±{best_std}...")
                load_parameters(best_solution, controller)
                torch.save({'generation': generation,
                            'reward': -cur_best,
                            'state_dict': controller.state_dict()},
                           c_save_path)
        if -best > target:
            print("Terminating controller training with value {}...".format(-best))
            break

    es.results_pretty()
if __name__ == "__main__":
    main()
