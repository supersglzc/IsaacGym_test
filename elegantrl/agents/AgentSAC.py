import numpy as np
import torch

from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.net import ActorSAC, CriticTwin
from elegantrl.agents.net import ActorFixSAC, CriticREDq


class AgentSAC(AgentBase):  # [ElegantRL.2022.03.03]
    """
    Soft Actor-Critic algorithm. “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor”. Tuomas Haarnoja et al.. 2018.

    :param net_dim: the dimension of networks (the width of neural networks)
    :param state_dim: the dimension of state (the number of state vector)
    :param action_dim: the dimension of action (the number of discrete action)
    :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
    :param args: the arguments for agent training. `args = Arguments()`
    """

    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.alpha_log = torch.tensor(
            (-np.log(action_dim),), dtype=torch.float32, requires_grad=True, device=self.device
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = getattr(args, 'target_entropy', np.log(action_dim))

    def update_net(self, buffer):
        buffer.update_now_len()

        obj_critic = obj_actor = None
        update_times = int(1 + buffer.now_len * self.repeat_times / self.batch_size)
        for _ in range(update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            if self.if_cri_target:
                self.optimizer_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

            q_value_pg = self.cri(state, a_noise_pg)
            obj_actor = -(q_value_pg + log_prob * alpha).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            if self.if_act_target:
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentReliableSAC(AgentSAC):  # Using TTUR (Two Time-scale Update Rule) for reliable_lambda
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorFixSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        args.if_act_target = getattr(args, 'if_act_target', True)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

        self.lambda_a_log_std = getattr(args, 'lambda_a_log_std', 2 ** -4)

        # h term
        self.lambda_h_term = getattr(args, 'lambda_h_term', 2 ** -3)
        self.h_term_throw_rate = 6

    def update_net0(self, buffer):
        buffer.update_now_len()
        with torch.no_grad():  # H term
            # buf_state = buffer.sample_batch_r_m_a_s()[3]
            if buffer.prev_p <= buffer.next_p:
                buf_state = buffer.buf_state[buffer.prev_p:buffer.next_p]
            else:
                buf_state = torch.vstack((buffer.buf_state[buffer.prev_p:],
                                          buffer.buf_state[:buffer.next_p],))
            buffer.prev_p = buffer.next_p

            avg_a_log_std = self.act.get_a_log_std(buf_state).mean(dim=0, keepdim=True)
            avg_a_log_std = avg_a_log_std * torch.ones((self.batch_size, 1), device=self.device)
            del buf_state

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = torch.zeros(1)
        update_times = int(buffer.now_len * self.repeat_times / self.batch_size)
        # update_times = int(buffer.cur_capacity * self.repeat_times / self.batch_size)
        for update_c in range(1, 2 + update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                obj_a_std = self.criterion(self.act.get_a_log_std(state), avg_a_log_std) * self.lambda_a_log_std

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean() + obj_a_std

                self.optimizer_update(self.act_optimizer, obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()

    def update_net(self, buffer):
        # buffer.update_now_len()  # todo
        with torch.no_grad():  # H term
            # buf_state = buffer.sample_batch_r_m_a_s()[3]
            if buffer.prev_p <= buffer.next_p:
                buf_state = buffer.buf_state[buffer.prev_p:buffer.next_p]
            else:
                buf_state = torch.vstack((buffer.buf_state[buffer.prev_p:],
                                          buffer.buf_state[:buffer.next_p],))
            buffer.prev_p = buffer.next_p

            avg_a_log_std = self.act.get_a_log_std(buf_state).mean(dim=0, keepdim=True)
            avg_a_log_std = avg_a_log_std * torch.ones((self.batch_size, 1), device=self.device)
            del buf_state

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = torch.zeros(1)
        # update_times = int(buffer.now_len * self.repeat_times / self.batch_size)
        update_times = int(buffer.cur_capacity * self.repeat_times / self.batch_size)
        for update_c in range(1, 2 + update_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                obj_a_std = self.criterion(self.act.get_a_log_std(state), avg_a_log_std) * self.lambda_a_log_std

                q_value_pg = self.cri(state, a_noise_pg)
                obj_actor = -(q_value_pg + logprob * alpha).mean() + obj_a_std

                self.optimizer_update(self.act_optimizer, obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()


class AgentREDqSAC(AgentSAC):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorFixSAC)
        self.cri_class = getattr(self, 'cri_class', CriticREDq)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        obj_critic = self.criterion(qs, q_label * torch.ones_like(qs))
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(next_s)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_s, next_a)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_log_prob * alpha)
        qs = self.cri.get_q_values(state, action)
        td_error = self.criterion(qs, q_label * torch.ones_like(qs)).mean(dim=1)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentHtermSAC(AgentSAC):  # Using TTUR (Two Time-scale Update Rule) for reliable_lambda + Hamilton term
    def __init__(self, net_dim, state_dim, action_dim, gpu_id=0, args=None):
        self.act_class = getattr(self, 'act_class', ActorFixSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.lambda_a_log_std = getattr(args, 'lambda_a_log_std', 2 ** -4)  # for H term

    def update_net(self, buffer):
        buffer.update_now_len()

        with torch.no_grad():  # H term
            # h_term
            # todo add  r_sum
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_batch_r_m_a_s()
            buf_len = buf_reward.shape[0]
            buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=buffer.device)

            pre_r_sum = 0
            for i in range(buf_len - 1, -1, -1):
                buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
                pre_r_sum = buf_r_sum[i]

            self.get_buf_h_term(buf_state, buf_action, buf_r_sum)

            # if buffer.prev_idx <= buffer.next_idx:
            #     buf_state = buffer.buf_state[buffer.prev_idx:buffer.next_idx]
            # else:
            #     buf_state = torch.vstack((buffer.buf_state[buffer.prev_idx:],
            #                               buffer.buf_state[:buffer.next_idx],))
            avg_a_log_std = self.act.get_a_log_std(buf_state).mean(dim=0, keepdim=True)
            avg_a_log_std = avg_a_log_std * torch.ones((self.batch_size, 1), device=self.device)
            del buf_state

            buffer.prev_p = buffer.next_p

        alpha = self.alpha_log.exp().detach()
        update_a = 0
        obj_actor = torch.zeros(1)
        for update_c in range(1, int(2 + buffer.now_len * self.repeat_times / self.batch_size)):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()  # for reliable_lambda

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            '''objective of alpha (temperature parameter automatic adjustment)'''
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
            alpha = self.alpha_log.exp().detach()

            '''objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)'''
            reliable_lambda = np.exp(-self.obj_c ** 2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                obj_q_value = self.cri(state, a_noise_pg)  # policy gradient
                obj_h_term = 0 * self.get_obj_h_term()
                obj_a_std = self.criterion(self.act.get_a_log_std(state), avg_a_log_std) * self.lambda_a_log_std

                obj_actor = -(obj_q_value + logprob * alpha).mean() + obj_h_term + obj_a_std

                self.optimizer_update(self.act_optimizer, obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)
        return self.obj_c, -obj_actor.item(), alpha.item()
