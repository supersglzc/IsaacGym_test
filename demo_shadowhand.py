import isaacgym
import torch
import sys

from elegantrl.train.run import train_and_evaluate, train_and_evaluate_mp
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv


def demo(gpu_id):
    env_name = 'ShadowHand'
    agent_class = AgentPPO

    env_func = IsaacVecEnv
    env_args = {
        'env_num': 1024,
        'env_name': env_name,
        'max_step': 600,
        'state_dim': 211,
        'action_dim': 20,
        'if_discrete': False,
        'target_return': 15000.,

        'sim_device_id': gpu_id,
        'rl_device_id': gpu_id,
    }
    args = Arguments(agent_class, env_func=env_func, env_args=env_args)
    args.eval_env_func = IsaacOneEnv
    args.eval_env_args = {
        'env_num': 1,
        'env_name': env_name,
        'max_step': 600,
        'state_dim': 211,
        'action_dim': 20,
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
    args.lambda_gae_adv = 0.85
    args.if_use_gae = True
    args.learning_rate = 2 ** -15

    args.lambda_entropy = 2 ** -6  # 0.001
    args.if_use_old_traj = False

    args.eval_gap = 2 ** 9
    args.eval_times = 2 ** 1
    args.max_step = int(6e7)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    if_check = 0
    if if_check:
        train_and_evaluate(args)
    else:
        train_and_evaluate_mp(args)


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU

    demo(GPU_ID)
