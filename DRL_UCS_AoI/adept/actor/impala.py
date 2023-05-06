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
from cmath import inf
from collections import OrderedDict, defaultdict
from functools import reduce

import torch
import copy
from adept.actor.base.ac_helper import ACActorHelperMixin
from adept.actor.base.actor_module import ActorModule
from adept.utils.util import listd_to_dlist, dlist_to_listd


class ImpalaHostActor(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space, args.nb_agent)

    @staticmethod
    def output_space(action_space):
        head_dict = {"critic": (1,), **action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        log_list = []
        entro_list = []
        values_list = []
        global_values_list = []

        for i in range(self.nb_agent):
            values = preds["critic"][i].squeeze(1)
            log_softmaxes = []
            entropies = []

            for key in self.action_keys:
                
                logit = self.flatten_logits(preds[key][i])

                if key=='Discrete' and  available_actions is not None:
                    logits_masks = available_actions[:,i,...] 
   
                    logit[logits_masks == 0] = -99999
                
                log_softmax, softmax = self.log_softmax(logit), self.softmax(logit)
                entropy = self.entropy(log_softmax, softmax)

                entropies.append(entropy)
                log_softmaxes.append(log_softmax)


            log_softmaxes = torch.stack(log_softmaxes, dim=1)
            entropies = torch.cat(entropies, dim=1)


            log_list.append(log_softmaxes)
            entro_list.append(entropies)
            values_list.append(values)
    

        log_list = torch.stack(log_list)
        entro_list = torch.stack(entro_list)
        values_list = torch.stack(values_list)
        return (
            None,
            {
                "log_softmaxes": log_list,
                "entropies": entro_list,
                "values": values_list
            },
        )

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space, nb_agent):
        flat_act_space = 0
        for k, shape in act_space.items():
            flat_act_space += reduce(lambda a, b: a * b, shape)
        act_key_len = len(act_space.keys())

        obs_spec = {
            k: (exp_len + 1, batch_sz, *shape) for k, shape in obs_space.items()
        }
        action_spec = {k: (exp_len, batch_sz, nb_agent) for k in act_space.keys()}
        internal_spec = {}
        for k, shape in internal_space.items():
            if k=='memory':
                internal_spec[k] = (exp_len,batch_sz, nb_agent,shape[1],shape[2],shape[3])
            elif k=='memory_critic':
                internal_spec[k] = (exp_len,batch_sz,shape[0],shape[1],shape[2])
            else:
                internal_spec[k] = (exp_len,batch_sz,*shape)
           

        spec = {
            "log_softmaxes": (exp_len, batch_sz, nb_agent, act_key_len, flat_act_space),
            "entropies": (exp_len, batch_sz, nb_agent, act_key_len),
            "values": (exp_len, batch_sz, nb_agent),
            # From Workers
            "log_probs": (exp_len, batch_sz, nb_agent, act_key_len),
            **obs_spec,
            **action_spec,
            **internal_spec,
        }

        return spec

    @classmethod
    def _key_types(cls, obs_space, act_space, internal_space):
        d = defaultdict(lambda: "float")
        for k in act_space.keys():
            d[k] = "long"
        # TODO this needs a better solution
        for k in obs_space.keys():
            d[k] = "byte"
        return d


class ImpalaWorkerActor(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space, args.nb_agent)

    @staticmethod
    def output_space(action_space):
        head_dict = {"critic": (1,),  **action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        actions_cpu_all = []
        actions_gpu_all = []
        log_probs_all = []
        internals = {k: torch.stack(vs) for k, vs in internals.items()}

        for i in range(self.nb_agent):
            log_probs = []
            actions_gpu = OrderedDict()
            actions_cpu = OrderedDict()

            for key in self.action_keys:
                logit = self.flatten_logits(preds[key][i])
                if key=='Discrete' and  available_actions is not None:
                    logits_masks = available_actions[:,i,...] 
                    logit[logits_masks == 0] = -99999
                
                log_softmax, softmax = self.log_softmax(logit), self.softmax(logit)
                action = self.sample_action(softmax)

                log_probs.append(self.log_probability(log_softmax, action))
                actions_gpu[key] = action
                actions_cpu[key] = action.cpu()

            log_probs = torch.cat(log_probs, dim=1)

            actions_cpu_all.append(actions_cpu)
            actions_gpu_all.append(actions_gpu)
            log_probs_all.append(log_probs)



        actions_cpu_all = listd_to_dlist(actions_cpu_all)
        for key in actions_cpu_all.keys():
            actions_cpu_all[key] = torch.stack(actions_cpu_all[key]).permute(1,0)

        actions_gpu_all = listd_to_dlist(actions_gpu_all)
        for key in actions_gpu_all.keys():
            actions_gpu_all[key] = torch.stack(actions_gpu_all[key]).permute(1,0)

        log_probs_all = torch.stack(log_probs_all)

        return actions_cpu_all, {"log_probs": log_probs_all, **actions_gpu_all, **internals}

    @classmethod
    def _exp_spec(cls, exp_len, batch_sz, obs_space, act_space, internal_space, nb_agent):
        act_key_len = len(act_space.keys())

        obs_spec = {
            k: (exp_len + 1, batch_sz, *shape) for k, shape in obs_space.items()
        }
        action_spec = {k: (exp_len, batch_sz, nb_agent) for k in act_space.keys()}
        #n_agent, self.n_layers+1, self.sequence_len, self.n_input,
      
        internal_spec = {}
        for k, shape in internal_space.items():
            ##n_agent, self.n_layers+1, self.sequence_len, self.n_input,
            if k=='memory':
                internal_spec[k] = (exp_len,batch_sz, nb_agent,shape[1],shape[2],shape[3])
            elif k=='memory_critic':
                internal_spec[k] = (exp_len,batch_sz,shape[0],shape[1],shape[2])
            else:
                internal_spec[k] = (exp_len,batch_sz,*shape)
        spec = {
            "log_probs": (exp_len, batch_sz, nb_agent, act_key_len),
            **obs_spec,
            **action_spec,
            **internal_spec,
        }

        return spec

    @classmethod
    def _key_types(cls, obs_space, act_space, internal_space):
        d = defaultdict(lambda: "float")
        for k in act_space.keys():
            d[k] = "long"
        # TODO this needs a better solution
        for k in obs_space.keys():
            d[k] = "byte"
        return d
