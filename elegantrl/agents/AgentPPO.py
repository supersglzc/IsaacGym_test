import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorDiscretePPO
from elegantrl.agents.net import ActorPPO, CriticPPO
import numpy as np

"""[ElegantRL.2022.03.03](github.com/AI4Fiance-Foundation/ElegantRL)"""


class AgentPPO(AgentBase):
    """
    PPO algorithm. “Proximal Policy Optimization Algorithms”. John Schulman. et al.. 2017.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        if getattr(args, 'if_use_gae', False):
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_gae_adv', 0.98)  # could be 0.95~0.99, GAE (ICLR.2016.)
        self.act_update_gap = getattr(args, 'act_update_gap', 1)

    def explore_one_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = [0, ]
        state = self.states[0]

        step_i = 0
        done = False
        get_action = self.act.get_action
        get_a_to_e = self.act.convert_action_for_env
        while step_i < target_step or not done:
            ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a, ten_n = [ten.cpu() for ten in get_action(ten_s.to(self.device))]  # different
            next_s, reward, done, _ = env.step(get_a_to_e(ten_a)[0].numpy())

            traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different

            step_i += 1
            state = env.reset() if done else next_s
        self.states[0] = state
        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def explore_vec_env(self, env, target_step) -> list:
        traj_list = list()
        last_done = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        ten_s = self.states

        step_i = 0
        ten_dones = torch.zeros(self.env_num, dtype=torch.int, device=self.device)
        get_action = self.act.get_action
        get_a_to_e = self.act.convert_action_for_env
        while step_i < target_step or not any(ten_dones):
            ten_a, ten_n = get_action(ten_s)  # different
            ten_s_next, ten_rewards, ten_dones, _ = env.step(get_a_to_e(ten_a))

            traj_list.append((ten_s.clone(), ten_rewards.clone(), ten_dones.clone(), ten_a, ten_n))  # different

            step_i += 1
            last_done[torch.where(ten_dones)[0]] = step_i  # behind `step_i+=1`
            ten_s = ten_s_next

        self.states = ten_s
        return self.convert_trajectory(traj_list, last_done)  # traj_list

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            del buf_noise

        '''update network'''
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            # if i % self.act_update_gap == 0:
            '''PPO: Surrogate objective of Trust Region'''
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, obj_actor)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item() * 100  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param buf_reward: a list of rewards for the state-action pairs.
        :param buf_mask: a list of masks computed by the product of done signal and discount factor.
        :param buf_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(self, buf_len, ten_reward, ten_mask, ten_value) -> (torch.Tensor, torch.Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.

        :param buf_len: the length of the ``ReplayBuffer``.
        :param ten_reward: a list of rewards for the state-action pairs.
        :param ten_mask: a list of masks computed by the product of done signal and discount factor.
        :param ten_value: a list of state values estimated by the ``Critic`` network.
        :return: the reward-to-go and advantage estimation.
        """
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv_v = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):  # Notice: mask = (1-done) * gamma
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_adv_v[i] = ten_reward[i] + ten_mask[i] * pre_adv_v - ten_value[i]
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
            # ten_mask[i] * pre_adv_v == (1-done) * gamma * pre_adv_v
        return buf_r_sum, buf_adv_v


class AgentDiscretePPO(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorDiscretePPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)


class AgentPPOHterm(AgentPPO):  # HtermPPO2
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term(buf_state, buf_action, buf_r_sum)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            # if i % self.act_update_gap == 0:
            '''PPO: Surrogate objective of Trust Region'''
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + self.get_obj_h_term()  # todo H-term
            self.optimizer_update(self.act_optimizer, obj_actor)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple


class AgentPPOHtermV2(AgentPPO):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        AgentPPO.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

    def update_net(self, buffer):
        with torch.no_grad():
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten.to(self.device) for ten in buffer]
            buf_len = buf_state.shape[0]

            '''get buf_r_sum, buf_logprob'''
            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [self.cri_target(buf_state[i:i + bs]) for i in range(0, buf_len, bs)]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(buf_len, buf_reward, buf_mask, buf_value)  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) / (buf_adv_v.std() + 1e-5)
            # buf_adv_v: buffer data of adv_v value
            self.get_buf_h_term(buf_state, buf_action, buf_r_sum)  # todo H-term
            del buf_noise

        '''update network'''
        obj_critic = None
        obj_actor = None
        assert buf_len >= self.batch_size
        for i in range(int(1 + buf_len * self.repeat_times / self.batch_size)):
            if_update_act = bool(i % self.act_update_gap == 0)
            obj_critic_h_term, obj_hamilton = self.get_obj_c_obj_h_term(if_update_act)  # todo H-term

            indices = torch.randint(buf_len, size=(self.batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, r_sum) + obj_critic_h_term
            self.optimizer_update(self.cri_optimizer, obj_critic)
            if self.if_cri_target:
                self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if if_update_act:
                '''PPO: Surrogate objective of Trust Region'''
                adv_v = buf_adv_v[indices]
                action = buf_action[indices]
                logprob = buf_logprob[indices]

                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy + obj_hamilton
                self.optimizer_update(self.act_optimizer, obj_actor)

        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critic.item(), -obj_actor.item(), a_std_log.item()  # logging_tuple

    def get_obj_c_obj_h_term(self, if_update_act=True):
        if (self.ten_state is None) or (self.ten_state.shape[0] < 1024):
            obj_critic = torch.zeros(1, dtype=torch.float32, device=self.device)
            obj_hamilton = torch.zeros(1, dtype=torch.float32, device=self.device)
            return obj_critic, obj_hamilton

        '''rd sample'''
        ten_size = 0 if self.ten_state is None else self.ten_state.shape[0]
        indices = torch.randint(ten_size, size=(int(ten_size * self.h_term_bs_rate),),
                                requires_grad=False, device=self.device)

        ten_state = self.ten_state[indices]
        ten_r_sum = self.ten_r_sum[indices]
        ten_r_norm = self.ten_r_norm[indices]
        '''critic'''
        ten_value = self.cri(ten_state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state

        obj_critic = (self.criterion1(ten_value, ten_r_sum) * ten_r_norm).mean()  # todo which one is better?
        # obj_critic = self.criterion(ten_value, ten_r_sum)  # todo which one is better?

        if if_update_act:
            ten_action = self.ten_action[indices]

            '''hamilton'''
            ten_logprob = self.act.get_logprob(ten_state, ten_action)
            ten_hamilton = ten_logprob.exp().prod(dim=1)
            obj_hamilton = -(ten_hamilton * ten_r_norm).mean() * self.lambda_h_term
        else:
            obj_hamilton = torch.zeros(1, dtype=torch.float32, device=self.device)
        return obj_critic, obj_hamilton
