import isaacgym
import torch
import sys

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv


def demo_a2c_ppo(gpu_id, env_id):
    env_name = ['Ant',
                'Humanoid', ][env_id]
    agent_class = AgentPPO

    if env_name == 'Ant':
        env_func = IsaacVecEnv
        env_args = {
            'env_num': 1024,
            'env_name': 'Ant',
            'max_step': 1000,
            'state_dim': 60,
            'action_dim': 8,
            'if_discrete': False,
            'target_return': 15000.,

            'sim_device_id': gpu_id,
            'rl_device_id': gpu_id,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.eval_env_func = IsaacOneEnv
        args.eval_env_args = {
            'env_num': 1,
            'env_name': 'Ant',
            'max_step': 1000,
            'state_dim': 60,
            'action_dim': 8,
            'if_discrete': False,
            'target_return': 15000.,

            'device_id': gpu_id,
        }

        args.reward_scale = 2 ** -4
        args.if_cri_target = False

        args.target_step = args.max_step

        args.net_dim = 2 ** 8
        args.batch_size = args.net_dim * 4
        args.repeat_times = 2 ** 4
        args.gamma = 0.985
        args.lambda_gae_adv = 0.9
        args.if_use_gae = True
        args.learning_rate = 2 ** -15

        args.lambda_entropy = 0.001

        args.eval_gap = 2 ** 9
        args.eval_times = 2 ** 1
        args.max_step = int(6e7)
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    demo_a2c_ppo(GPU_ID, ENV_ID)
