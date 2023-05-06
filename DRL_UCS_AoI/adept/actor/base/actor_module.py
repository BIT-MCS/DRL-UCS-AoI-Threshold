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
"""
An actor observes the environment and decides actions. It also outputs extra
info necessary for model updates (learning) to occur.
"""
import abc
from collections import defaultdict

from adept.exp.base.spec_builder import ExpSpecBuilder
from adept.utils.requires_args import RequiresArgsMixin


class ActorModule(RequiresArgsMixin, metaclass=abc.ABCMeta):
    def __init__(self, action_space,nb_agent):
        self._action_space = action_space
        self.nb_agent = nb_agent

    @property
    def action_space(self):
        return self._action_space

    @property
    def action_keys(self):
        return sorted(self.action_space.keys())

    @staticmethod
    @abc.abstractmethod
    def output_space(action_space):
        raise NotImplementedError

    @classmethod
    def exp_spec_builder(cls, obs_space, act_space, internal_space, batch_sz, nb_agent):
        def build_fn(exp_len, nb_agent):
            exp_space = cls._exp_spec(
                exp_len, batch_sz, obs_space, act_space, internal_space, nb_agent
            )
            env_space = {
                "rewards": (exp_len, batch_sz,nb_agent+1),
                "terminals": (exp_len, batch_sz),
            }
            return {**exp_space, **env_space}

        key_types = cls._key_types(obs_space, act_space, internal_space)
        exp_keys = cls._exp_keys(obs_space, act_space, internal_space, nb_agent)
        return ExpSpecBuilder(
            obs_space, act_space, internal_space, key_types, exp_keys, nb_agent, build_fn
        )

    @classmethod
    @abc.abstractmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space, nb_agent):
        raise NotImplementedError

    @classmethod
    def _exp_keys(cls, obs_space, act_space, internal_space, nb_agent):
        dummy = cls._exp_spec(1, 1, obs_space, act_space, internal_space, nb_agent)
        return dummy.keys()

    @classmethod
    def _key_types(cls, obs_space, act_space, internal_space):
        return defaultdict(lambda: "float")

    @abc.abstractmethod
    def from_args(self, args, action_space):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_action_exp(self, preds, internals, obs, available_actions):
        """
        B = Batch Size

        :param preds: Dict[str, torch.Tensor]
        :return:
            actions: Dict[ActionKey, Tensor (B)]
            experience: Dict[str, Tensor (B, X)]
        """
        raise NotImplementedError

    def act(self, network, obs, prev_internals,is_host=False):
        """
        :param obs: Dict[str, Tensor]
        :param prev_internals: previous interal states. Dict[str, Tensor]
        :return:
            actions: Dict[ActionKey, Tensor (B)]
            experience: Dict[str, Tensor (B, X)]
            internal_states: Dict[str, Tensor]
        """
        # if isinstance(network, dict):
        #     actions_all = []
        #     exps_all = []
        #     internal_all = []
        #     for index, n in enumerate(network):
        #         predictions, internal_states, pobs = n(obs[index], prev_internals[index])
        #         if "available_actions" in obs:
        #             av_actions = obs["available_actions"]
        #         else:
        #             av_actions = None
        #         internal_all.append(internal_states)
        #         actions, exp = self.compute_action_exp(
        #             predictions, prev_internals, pobs, av_actions
        #         )
        #         actions_all.append(actions)
        #         exps_all.append(exp)
        #     return actions_all, exps_all, internal_all

        predictions, internal_states, pobs = network(obs, prev_internals,is_host)

        if "available_actions" in obs:
            av_actions = obs["available_actions"]
        else:
            av_actions = None

        actions, exp = self.compute_action_exp(
            predictions, prev_internals, pobs, av_actions
        )
        #print(internal_states['memory'][0].shape) 3 5 20 762
        return actions, exp, internal_states
