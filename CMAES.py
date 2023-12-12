import cv2
import torch
import cma
import gymnasium as gym
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
    # Generate a single rollout
    def __init__(self, vae_latent_size, rnn_hidden_units, action_dim, gaussian_mixtures, v_path, m_path, render=False,
                 device=None):
        self.gaussian_mixtures = gaussian_mixtures
        self.action_dim = action_dim
        self.vae_latent_size = vae_latent_size
        self.rnn_hidden_units = rnn_hidden_units
        self.vae = VAE(vae_latent_size, device).to(device)
        checkpoint_v = torch.load(v_path)
        self.vae.load_state_dict(checkpoint_v)

        self.mdnrnn = RNN(vae_latent_size, self.rnn_hidden_units, self.action_dim, self.gaussian_mixtures).to(device)
        checkpoint_m = torch.load(m_path)
        self.mdnrnn.load_state_dict(checkpoint_m)

        self.control = CONTROLLER(self.rnn_hidden_units, self.vae_latent_size, self.action_dim).to(device)
        render_mode = 'human' if render else 'rgb_array'
        self.env = gym.make('CarRacing-v2', render_mode=render_mode)
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'

    def pre_process(self, obs):
        x = cv2.resize(obs, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        x = x.astype('float32') / 255
        x = x.T
        x = torch.from_numpy(x).view(-1, 3, 64, 64)
        return x

    def rollout(self, params, time_steps=1000):
        # Perform a single rollout.
        # time_steps should be until the end of race, however, due to limited computational horsepower
        # we use (50+generations) time steps for CMAES algorithm

        load_parameters(params, self.control, )

        obs, _ = self.env.reset()
        self.env.render()
        (h, c) = [torch.zeros(1, 1, self.rnn_hidden_units).to(self.device) for _ in range(2)]
        total_reward = 0

        for step in range(time_steps):
            obs = self.pre_process(obs).to(self.device)
            # using mean as z vector
            _, z, _, _ = self.vae(obs)
            a = self.control.get_action(z, h)
            obs, reward, terminated, truncated, _ = self.env.step(a)

            a = torch.from_numpy(a).view(1, 1, self.action_dim).to(self.device)
            x = torch.cat((z.view(1, 1, self.vae_latent_size), a), 2)
            _, (h, c) = self.mdnrnn.rnn(x, (h, c))

            total_reward += reward
            if terminated or truncated:
                break
        return -1 * total_reward


def individual_fitness(rolloutGenerator, solution, n_rollouts, time_steps):
    avg_cumulated_rewards = 0
    for i in range(n_rollouts):
        avg_cumulated_rewards += rolloutGenerator.rollout(solution, time_steps) / n_rollouts
    return avg_cumulated_rewards


def fitness(rolloutGenerator, solutions, n_rollouts, time_steps):
    rewards = [0] * len(solutions)
    for idx, solution in enumerate(solutions):  # loss calculation
        # load params into controller
        rewards[idx] = individual_fitness(rolloutGenerator, solution, n_rollouts, time_steps)
    return rewards


def main(initial_time_steps, vae_latent_size, best_rollouts, n_rollouts, pop_size, target, sigma, rnn_hidden_units, action_dim,
         gaussian_mixtures, eval_best_interval, v_path, m_path, c_path):
    cur_best = None
    generation = 0
    best = 0

    # dummy controller to initialize parameters
    controller = CONTROLLER(rnn_hidden_units, vae_latent_size, action_dim)
    parameters = controller.parameters()
    flatten_parameters = torch.cat([p.detach().view(-1) for p in parameters], dim=0).numpy()
    es = cma.CMAEvolutionStrategy(flatten_parameters, sigma, {'popsize': pop_size})

    r_gen = RolloutGenerator(vae_latent_size, rnn_hidden_units, action_dim, gaussian_mixtures, v_path, m_path,
                             render=False)
    time_steps = initial_time_steps
    while not es.stop():
        generation += 1
        time_steps += 1
        print(
            f"Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s] -- generation: {generation} -- cur_best: {cur_best}")
        if cur_best is not None and (-cur_best) > target:
            print("Already better than target, breaking...")
            break
        solutions = es.ask()
        with torch.no_grad():
            r_list = fitness(r_gen, solutions, n_rollouts, time_steps)

        es.tell(solutions, r_list)
        es.disp()
        if generation > 0 and generation % eval_best_interval == 0:
            with torch.no_grad():
                best = individual_fitness(r_gen, es.result.xbest, best_rollouts, time_steps)
                print(f"cur_best evaluated so far resulting: {best}")
            # best comparison is > because working with negative of rewards
            if not cur_best or cur_best > best:
                cur_best = best
                print(f"Saving new best with value {-cur_best}±{es.result.xbest.std}...")
                load_parameters(es.result.xbest, controller)
                torch.save({'generation': generation,
                            'reward': -cur_best,
                            'state_dict': controller.state_dict()},
                           c_path)
        if -best > target:
            print("Terminating controller training with value {}...".format(-best))
            break

    es.results_pretty()


# parameters
v_path = './models/VAE_weights.pth'
m_path = './models/rnn_weights.pth'
c_path = './models/cmaes_train_controller.pth'
vae_latent_size = 32
rnn_hidden_units = 256
# given by continuous environment is 3
action_dim = 3
gaussian_mixtures = 5
target_reward = 930
initial_time_steps = 50
CMAES_population_size = 64
CMAES_initial_sigma = 0.1
# (best in name is because it is obtained from the best individual in CMAES algorithm)
NO_best_rollouts = 100  # average number of rollouts for agent to have the target reward
eval_best_interval = 25  # the interval that best solution of CMAES is evaluated
# (best in name is because it is calculated for all individuals regardless of their cost in CMAES algorithm)
normal_rollouts = 16  # average number of rollouts for agent to have the target reward

if __name__ == "__main__":
    main(initial_time_steps=initial_time_steps, vae_latent_size=vae_latent_size,
           best_rollouts=NO_best_rollouts, n_rollouts=normal_rollouts, pop_size=CMAES_population_size,
           eval_best_interval=eval_best_interval, target=target_reward, sigma=CMAES_initial_sigma,
           rnn_hidden_units=rnn_hidden_units,
           action_dim=action_dim, gaussian_mixtures=gaussian_mixtures, v_path=v_path, m_path=m_path,
           c_path=c_path)
