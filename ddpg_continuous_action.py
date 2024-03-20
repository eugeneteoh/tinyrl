from dataclasses import dataclass
import os
import random
import time
import gymnasium as gym
import numpy as np
import tyro
from tensorboardX import SummaryWriter
from tinygrad import Tensor, nn, dtypes
# TODO: Remove sb3
from stable_baselines3.common.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class Actor:
    def __init__(self, env):
        self.fc1 = nn.Linear(np.prod(env.single_observation_space.shape).item(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape).item())
        # action rescaling
        self.action_scale = Tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=dtypes.float32)
        self.action_bias = Tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=dtypes.float32)

    def __call__(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc_mu(x).tanh()
        return x * self.action_scale + self.action_bias

class QNetwork:
    def __init__(self, env):
        in_features = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)
        in_features = in_features.item()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def __call__(self, x, a):
        x = Tensor.cat(x, a, dim=1)
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    Tensor.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs)
    target_actor = Actor(envs)
    qf1 = QNetwork(envs)
    qf1_target = QNetwork(envs)
    nn.state.load_state_dict(target_actor, nn.state.get_state_dict(actor))
    nn.state.load_state_dict(qf1_target, nn.state.get_state_dict(qf1))
    q_optimizer = nn.optim.Adam(nn.state.get_parameters(qf1), lr=args.learning_rate)
    actor_optimizer = nn.optim.Adam(nn.state.get_parameters(actor), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with Tensor.inference_mode():
                actions = actor(Tensor(obs, dtype=dtypes.float32))
                actions = actions + Tensor.normal(actions.shape, mean=0, std=actor.action_scale * args.exploration_noise)
                actions = actions.numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            next_observations = Tensor(data.next_observations.numpy(), dtype=dtypes.float32)
            rewards = data.rewards.numpy()
            dones = data.dones.numpy()
            with Tensor.inference_mode():
                next_state_actions = target_actor(next_observations)
                qf1_next_target = qf1_target(next_observations, next_state_actions)
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (qf1_next_target).view(-1)
