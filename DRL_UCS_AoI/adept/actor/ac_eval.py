from collections import OrderedDict

from adept.actor import ActorModule
from adept.actor.base.ac_helper import ACActorHelperMixin
from adept.utils import listd_to_dlist
import torch

class ACActorEval(ActorModule, ACActorHelperMixin):
    args = {}

    @classmethod
    def from_args(cls, args, action_space):
        return cls(action_space,args.nb_agent)

    @staticmethod
    def output_space(action_space):
        head_dict = {"critic": (1,), **action_space}
        return head_dict

    def compute_action_exp(self, preds, internals, obs, available_actions):
        actions_all = []
        softmax_all = []
        for i in range(self.nb_agent):
            actions = {}
            for key in self.action_keys:
                logit = self.flatten_logits(preds[key][i])
                if key=='Discrete' and  available_actions is not None:
                    logits_masks = available_actions[:,i,...] 
                    logit[logits_masks == 0] = -99999
                #print(key,logit)
                softmax = self.softmax(logit)
                action = self.select_action(softmax)
                actions[key] = action.cpu()
                softmax_all.append(softmax)
            actions_all.append(actions)
        softmax_all = torch.stack(softmax_all)
        actions_gpu_all = listd_to_dlist(actions_all)
        for key in actions_gpu_all.keys():
            actions_gpu_all[key] = torch.stack(actions_gpu_all[key]).permute(1,0)
        return actions_gpu_all, {}

    @classmethod
    def _exp_spec(
            cls, rollout_len, batch_sz, obs_space, act_space, internal_space, nb_agent
    ):
        return {}


class ACActorEvalSample(ACActorEval):
    def compute_action_exp(self, preds, internals, obs, available_actions):
        actions = OrderedDict()

        for key in self.action_keys:
            logit = self.flatten_logits(preds[key])

            softmax = self.softmax(logit)
            action = self.sample_action(softmax)

            actions[key] = action.cpu()
        return actions, {"value": preds["critic"].squeeze(-1)}
