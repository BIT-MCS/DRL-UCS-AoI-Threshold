# Copyright (C) 2018 Heron Systems, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from dis import Instruction
import torch
import numpy as np
import copy
from collections import namedtuple
from adept.utils.util import listd_to_dlist, dlist_to_listd
from ..exp.replay import ObjDict
from .base import LearnerModule
from .base.dm_return_scale import DeepMindReturnScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time

def _flatten(tensor):
    assert tensor is not None
    flat = tensor.reshape(-1, )
    return flat, tensor.shape, flat.shape


class ImpalaLearner(LearnerModule):
    """
    Reference implementation:
    https://github.com/deepmind/scalable_agent/blob/master/vtrace.py
    """

    args = {
        "discount": 0.99,
        "minimum_importance_value": 1.0,
        "minimum_importance_policy": 1.0,
        "entropy_weight": 0.01,
    }

    def __init__(
            self,
            reward_normalizer,
            discount,
            minimum_importance_value,
            minimum_importance_policy,
            entropy_weight,
            nb_agent,
            is_centralize,
            independent_reward,
            rollout_len,
            args
    ):
        self.reward_normalizer = reward_normalizer
        self.discount = discount
        self.minimum_importance_value = minimum_importance_value
        self.minimum_importance_policy = minimum_importance_policy
        self.entropy_weight = entropy_weight
        self.nb_agent = nb_agent
        self.independent_reward = independent_reward
        self.rollout_len = rollout_len
        self.device = torch.device("cuda" if (
            torch.cuda.is_available()) else "cpu")
        self.args = args
       
        self.use_intrinsic = args.use_intrinsic


    def sync_target_network(self, network):
        self.target_network = copy.deepcopy(network)

    @classmethod
    def from_args(cls, args, reward_normalizer):
        return cls(
            reward_normalizer,
            discount=args.discount,
            minimum_importance_value=args.minimum_importance_value,
            minimum_importance_policy=args.minimum_importance_policy,
            entropy_weight=args.entropy_weight,
            nb_agent=args.nb_agent,
            is_centralize=args.is_centralize,
            independent_reward=args.independent_reward,
            rollout_len=args.rollout_len,
            args=args
        )



    def learn_step(self, updater, network, experiences, next_obs, internals):
        value_loss = 0
        policy_loss = 0
        entropy_loss = 0
        with torch.no_grad():
            results, _, _ = network(next_obs, internals)
            b_last_values = results["critic"].squeeze(2).data

        r_obs = torch.stack(listd_to_dlist(experiences.observations)['SmallBox']) if self.args.small_obs_num is not None and self.args.small_obs_num > -1 else  torch.stack(listd_to_dlist(experiences.observations)['Box']) 
        r_state = torch.stack(listd_to_dlist(experiences.observations)['State'])
        r_actions = torch.stack(listd_to_dlist(experiences.actions)['Discrete'])

        
        if self.use_intrinsic:
            rnd_loss = network.rnd.update(r_obs)
            predictor_loss = network.predictor.update(r_obs)
            
        for agent_id in range(self.nb_agent):
            # Gather host log_probs
            r_log_probs = []
            for b_action, b_log_softs in zip(
                    experiences.actions, experiences.log_softmaxes
            ):
                k_log_probs = []
                for act_tensor, log_soft in zip(
                        b_action.values(), b_log_softs[agent_id].unbind(1)
                ):
                    a = act_tensor[:, agent_id]
                    log_prob = log_soft.gather(1, a.unsqueeze(1))
                    k_log_probs.append(log_prob)
                r_log_probs.append(torch.cat(k_log_probs, dim=1))

            r_log_probs_learner = torch.stack(r_log_probs)
            r_log_probs_actor = torch.stack(experiences.log_probs)[
                :, agent_id, :, :]  # TODO
            if self.independent_reward:
                r_rewards = self.reward_normalizer(
                    torch.stack(experiences.rewards)[:, :, agent_id + 1]
                )  # normalize rewards
            else:
                r_rewards = self.reward_normalizer(
                    torch.stack(experiences.rewards)[:, :, 0]
                )  # normalize rewards

            r_values = torch.stack(experiences.values)[:, agent_id, :]
            r_terminals = torch.stack(experiences.terminals)
            r_entropies = torch.stack(experiences.entropies)[
                :, agent_id, :]
            r_dterminal_masks = self.discount * (1.0 - r_terminals.float())
        

            # print('r_values',r_values.shape) # 20 64
            # print('r_terminals',r_terminals.shape) # 20 64
            # print('r_entropies',r_entropies.shape) # 20 64 1
            # print('r_dterminal_masks',r_dterminal_masks.shape) #20 64
            # print('r_rewards',r_rewards.shape) # 20 64
            with torch.no_grad():
                r_log_diffs = r_log_probs_learner - r_log_probs_actor
                # print('r_log_diffs', r_log_diffs.shape) # 20 64
                vtrace_target, pg_advantage, importance = self._vtrace_returns(
                    r_log_diffs,
                    r_dterminal_masks,
                    r_rewards,
                    r_values,
                    b_last_values[agent_id],
                    self.minimum_importance_value,
                    self.minimum_importance_policy
                )

            value_loss += 0.5 * (vtrace_target - r_values).pow(2).mean()
            policy_loss += torch.mean(-r_log_probs_learner * pg_advantage)
            entropy_loss += torch.mean(-r_entropies) * self.entropy_weight
        
      

        updater.step(value_loss + policy_loss + entropy_loss)

        losses = {
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }
        metrics = {"importance": importance.mean()}
    

        if self.use_intrinsic:
            losses ={
                "rnd_loss":rnd_loss,
                "predictor_loss":predictor_loss,
                **losses
            }
        return losses, metrics

    @staticmethod
    def _vtrace_returns(
            log_prob_diffs,
            discount_terminal_mask,
            r_rewards,
            r_values,
            bootstrap_value,
            min_importance_value,
            min_importance_policy
    ):
        rollout_len = log_prob_diffs.shape[0]

        importance = torch.exp(log_prob_diffs)

        clamped_importance_value = importance.clamp(
            max=min_importance_value)
        # if multiple actions take the average, (dim 3 is seq, batch, # actions)
        if clamped_importance_value.dim() == 3:
            clamped_importance_value = clamped_importance_value.mean(-1)

        # create nstep vtrace return
        # first create d_tV of function 1 in the paper
        values_t_plus_1 = torch.cat(
            (r_values[1:], bootstrap_value.unsqueeze(0))
        )
        diff_value_per_step = clamped_importance_value * (
            r_rewards + discount_terminal_mask * values_t_plus_1 - r_values
        )

        # reverse over the values to create the summed importance weighted
        # return everything on the right side of the plus in function 1 of
        # the paper
        vs_minus_v_xs = []
        nstep_v = 0.0
        # TODO: this uses a different clamping if != 1
        if min_importance_policy != 1 or min_importance_value != 1:
            raise NotImplementedError()

        for i in reversed(range(rollout_len)):
            nstep_v = (
                diff_value_per_step[i]
                + discount_terminal_mask[i]
                * clamped_importance_value[i]
                * nstep_v
            )
            vs_minus_v_xs.append(nstep_v)
        # reverse to a forward in time list
        vs_minus_v_xs = torch.stack(list(reversed(vs_minus_v_xs)))

        # Add V(s) to finish computation of v_s
        v_s = r_values + vs_minus_v_xs

        # advantage is pg_importance * (v_s of t+1 - values)
        clamped_importance_pg = importance.clamp(
            max=min_importance_value)*0.99+(1-0.99)*importance

        v_s_tp1 = torch.cat((v_s[1:], bootstrap_value.unsqueeze(0)))
        advantage = r_rewards + discount_terminal_mask * v_s_tp1 - r_values

        # if multiple actions broadcast the advantage to be weighted by the
        # different actions importance
        # (dim 3 is seq, batch, # actions)
        if clamped_importance_pg.dim() == 3:
            advantage = advantage.unsqueeze(-1)

        weighted_advantage = clamped_importance_pg * advantage
        return v_s, weighted_advantage, importance



    def update_predict_network(self,experiences,next_obs):
        #print('Update predictor!')
        exp = listd_to_dlist(experiences.observations)['Box']
        obs_all = torch.cat(
            [torch.stack(exp), next_obs['Box'].unsqueeze(0)], dim=0).clone().detach()
        obs = obs_all[:-1, ...]
        obs_next = obs_all[1:, ...]
        T, B, N, O = obs.shape

        add_id = torch.eye(self.nb_agent).to(obs.device).expand( # 20 64 3 
            [obs.shape[0], obs.shape[1], self.nb_agent,self.nb_agent]).detach()
        actions = torch.stack(listd_to_dlist(experiences.actions)['Discrete']).clone().detach()
        actions_onehot = torch.nn.functional.one_hot(actions, self.n_actions).to(self.device)
        mask =  (1.0 - torch.stack(experiences.terminals).float()).unsqueeze(-1).repeat(1,1,self.nb_agent).clone().detach()
        h_cat = torch.stack(experiences.hx_actor).to(self.device).clone().detach()
        #cx = torch.stack(experiences.cx_actor)

        #print(add_id.shape,actions_onehot.shape,mask.shape,h_cat.shape) torch.Size([20, 64, 3, 3]) torch.Size([20, 64, 3, 9]) torch.Size([20, 64]) torch.Size([20, 64, 3, 256])

        _obs = obs.reshape(-1, obs.shape[-1]).detach()
        _add_id = add_id.reshape(-1,add_id.shape[-1]).detach()
        _mask_reshape = mask.reshape(-1, 1).detach()
        # _obs_next = obs_next.reshape(-1, obs_next.shape[-1]).detach()
        # _h_cat = h_cat.reshape(-1, h_cat.shape[-1]).detach()
        # _actions_onehot = actions_onehot.reshape(
        #     -1, actions_onehot.shape[-1]).detach()

  
        # h_cat_r = torch.cat(
        #     [torch.zeros_like(h_cat[:, 0]).unsqueeze(1), h_cat[:, :-1]], dim=1)
        intrinsic_input_with_obs = torch.cat(
            [h_cat, obs_next, actions_onehot], dim=-1).detach()
        _inputs_obs = intrinsic_input_with_obs.reshape(-1, intrinsic_input_with_obs.shape[-1]).detach()

        intrinsic_input = torch.cat(
            [h_cat, actions_onehot], dim=-1).detach()
        _inputs = intrinsic_input.reshape(-1, intrinsic_input.shape[-1]).detach()

        loss_withobs_list, loss_withoutobs_list= [], []
        # update predict network
        for _ in range(self.predict_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(_obs.shape[0])), 256, False):
                loss_withoutobs = self.predict_id.update(
                    _inputs[index], _add_id[index], _mask_reshape[index])
                loss_withobs = self.predict_id_with_obs.update(
                    _inputs_obs[index], _add_id[index], _mask_reshape[index])

                if loss_withoutobs:
                    loss_withoutobs_list.append(loss_withoutobs)
                if loss_withobs:
                    loss_withobs_list.append(loss_withobs)

        return  torch.stack(loss_withoutobs_list).mean(),torch.stack(loss_withobs_list).mean()
