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
from collections import namedtuple
from time import time

import numpy as np
import ray
import torch
import copy

from adept.manager import SubProcEnvManager
from adept.network import ModularNetwork
from adept.registry import REGISTRY
from adept.utils.util import dtensor_to_dev, listd_to_dlist

from adept.container.base import Container


class ActorLearnerWorker(Container):
    @classmethod
    def as_remote(
            cls,
            num_cpus=None,
            num_gpus=None,
            memory=None,
            object_store_memory=None,
            resources=None,
    ):
        # Worker can't use more than 1 gpu, but can also be cpu only
        assert num_gpus is None or num_gpus <= 1
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)

    def __init__(self, args, log_id_dir, initial_step_count, rank):
        seed = args.seed if rank == 0 else args.seed + args.nb_env * rank
        print("Worker {} using seed {}".format(rank, seed))

        # load saved registry classes
        REGISTRY.load_extern_classes(log_id_dir)

        # ENV
        engine = REGISTRY.lookup_engine(args.env)
        env_cls = REGISTRY.lookup_env(args.env)
        mgr_cls = REGISTRY.lookup_manager(args.manager)
        env_mgr = mgr_cls.from_args(args, engine, env_cls, seed=seed)

        # NETWORK
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        output_space = REGISTRY.lookup_output_space(
            args.actor_worker, env_mgr.action_space
        )
        if args.custom_network:
            net_cls = REGISTRY.lookup_network(args.custom_network)
        else:
            net_cls = ModularNetwork
        nets = net_cls.from_args(
            args,
            env_mgr.observation_space,
            output_space,
            env_mgr.gpu_preprocessor,
            REGISTRY,
        )
        actor_cls = REGISTRY.lookup_actor(args.actor_worker)
        actor = actor_cls.from_args(args, env_mgr.action_space)
        builder = actor_cls.exp_spec_builder(
            env_mgr.observation_space,
            env_mgr.action_space,
            nets.internal_space(),
            env_mgr.nb_env,
            args.nb_agent
        )
        exp = REGISTRY.lookup_exp(args.exp).from_args(args, builder)
        self.nb_agent = args.nb_agent

        self.actor = actor
        self.exp = exp.to(device)
        self.nb_step = args.nb_step
        self.env_mgr = env_mgr
        self.nb_env = args.nb_env

        self.networks = nets.to(device)
        self.networks.train()
        self.device = device

        self.initial_step_count = initial_step_count

        # TODO: this should be set to eval after some number of training steps

        # SETUP state variables for run
        self.step_count = self.initial_step_count
        self.global_step_count = self.initial_step_count
        self.ep_rewards = torch.zeros(self.nb_env)
        self.rank = rank

        self.obs = dtensor_to_dev(self.env_mgr.reset(), self.device)
        self.internals = listd_to_dlist(
            [
                self.networks.new_internals(self.device)
                for _ in range(self.nb_env)
            ]
        )
        self.start_time = time()
        self._weights_synced = False
        
        self.use_intrinsic = args.use_intrinsic
        if self.use_intrinsic:
            self.state_memory = []
            self.nb_agent = args.nb_agent
            self.intrinsic_coef = args.intrinsic_coef if args.intrinsic_coef is not None else 0.3
            
            if 'BJ' in args.dataset:
                self.x = 78.244
                self.y = 40.56
            else :
                self.x = 66.8
                self.y = 44.91
                
            self.running_count = 1
            self.running_sum = 0.001
        self.args = args

    def run(self):
        if not self._weights_synced:
            raise Exception("Must set weights before calling run")

        self.exp.clear()
        all_terminal_rewards = []
        all_terminal_infos = {}
        
        # loop to generate a rollout
        while not self.exp.is_ready():
            # print('obs_old', self.obs['Box'].shape)
            with torch.no_grad():
                actions, exp, self.internals = self.actor.act(
                    self.networks, self.obs, self.internals
                )
            # print(exp['hx'].shape)
            self.exp.write_actor(exp)
            next_obs, rewards, terminals, infos = self.env_mgr.step(actions)
            if self.use_intrinsic:
          
                with torch.no_grad():

                    intrin_input = next_obs['SmallBox'] if self.args.small_obs_num is not None and self.args.small_obs_num > -1 else next_obs['Box']
          
                    
                    knn_rewards = 0
                    predict_rewards = 0
        
                    knn_rewards = self.cal_knn_reward(self.state_memory,intrin_input)
                    for i in range(self.nb_agent):
                            self.state_memory.append(torch.cat([intrin_input[:,i,i*2:i*2+1]*self.x,intrin_input[:,i,i*2+1:i*2+2]*self.y],dim=-1))
                    
                    predict_rewards = torch.stack(self.internals['predictor']) 
                    coef = torch.clamp(torch.stack(self.internals['coef']),min=0.3,max=1)
                    intrinsic_rewards = (knn_rewards+predict_rewards)*coef*self.intrinsic_coef
                    intrinsic_rewards = intrinsic_rewards.detach().to('cpu')
                    
                    #coef = torch.mean(coef,dim=-1)
                    for i in range(self.nb_env):
                        for j in range(self.nb_agent):
                            infos[i]['i_coef_{}'.format(j)] = coef[i][j].item()
                            infos[i]['i_knn_rewards_{}'.format(j)] = knn_rewards[i][j].item()
                            infos[i]['i_eoi_rewards_{}'.format(j)] = predict_rewards[i][j].item()
                            infos[i]['i_intrinsic_rewards_{}'.format(j)] = intrinsic_rewards[i][j].item()
      
                    rewards[:,1:] += intrinsic_rewards
                    rewards[:,0] += torch.mean(intrinsic_rewards,dim=-1)

            next_obs = dtensor_to_dev(next_obs, self.device)

        
            self.exp.write_env(
                self.obs, rewards, terminals.float(), infos
            )

            self.step_count += self.nb_env
            self.ep_rewards += rewards[:, 0]
            self.obs = next_obs

            term_rewards = []
            for i, terminal in enumerate(terminals):
                if terminal:
                    for k, v in self.networks.new_internals(self.device).items():
                        self.internals[k][i] = v
                    rew = self.ep_rewards[i].item()
                    term_rewards.append(rew)
                    self.ep_rewards[i].zero_()

                    for k, v in infos[i].items():
                        if k not in all_terminal_infos:
                            all_terminal_infos[k] = []
                        all_terminal_infos[k].append(v)
                    if self.use_intrinsic:
                        self.state_memory = [] 
                        for i in range(self.nb_agent):
                            self.state_memory.append(self.obs['Box'][:,i,i*2:(i+1)*2].cpu())

            # avg rewards
            if term_rewards:
                term_reward = np.mean(term_rewards)
                all_terminal_rewards.append(term_reward)

                delta_t = time() - self.start_time
                print(
                    "RANK: {} "
                    "LOCAL STEP: {} "
                    "REWARD: {} "
                    "LOCAL STEP/S: {:.2f}".format(
                        self.rank,
                        self.step_count,
                        term_reward,
                        (self.step_count - self.initial_step_count) / delta_t,
                    )
                )

        # rollout is full return it
        self.exp.write_next_obs(self.obs)
        # TODO: compression?
        if len(all_terminal_rewards) > 0:
            return {
                "rollout": self._ray_pack(self.exp),
                "terminal_rewards": np.mean(all_terminal_rewards),
                "terminal_infos": {
                    k: np.mean(v) for k, v in all_terminal_infos.items()
                },
            }
        else:
            return {
                "rollout": self._ray_pack(self.exp),
                "terminal_rewards": None,
                "terminal_infos": None,
            }

    def set_weights(self, weights):
        for w, local_w in zip(weights, self.get_parameters()):
            # use data to ignore weights requiring grads
            local_w.data.copy_(w, non_blocking=True)
        self._weights_synced = True

    def set_global_step(self, global_step_count):
        self.global_step_count = global_step_count

    def get_parameters(self):
        params = [p for p in self.networks.parameters()]
        params.extend([b for b in self.networks.buffers()])
        return params

    def close(self):
        return self.env_mgr.close()

    def _ray_pack(self, exp):
        on_cpu = {k: self._to_cpu(v) for k, v in exp.items()}
        return on_cpu

    def _to_cpu(self, var):
        # TODO: this is a hack, should instead register a custom serializer for torch tensors to go
        # to CPU
        if isinstance(var, list):
            # list of dict -> dict of lists
            # observations/actions/internals
            if isinstance(var[0], dict):
                # if empty dict it doesn't matter
                if len(var[0]) == 0:
                    return {}
                first_v = next(iter(var[0].values()))
                # observations/actions
                if isinstance(first_v, torch.Tensor):
                    return {
                        k: torch.stack(v).cpu()
                        for k, v in listd_to_dlist(var).items()
                    }
                # internals
                elif isinstance(first_v, list):
                    # TODO: there's gotta be a better way to do this
                    assert len(var) == 1
                    return {
                        k: torch.stack(v).cpu().unsqueeze(0)
                        for k, v in var[0].items()
                    }
            # other actor stuff
            elif isinstance(var[0], torch.Tensor):

                return torch.stack(var).cpu()
            else:
                raise NotImplementedError(
                    "Expected rollout item to be a Tensor or dict(Tensors) got {}".format(
                        type(var[0])
                    )
                )
        elif isinstance(var, dict):
            # next obs
            if isinstance(first_v, torch.Tensor):
                return {k: v.cpu() for k, v in var.items()}
            else:
                raise NotImplementedError(
                    "Expected rollout dict item to be a tensor got {}".format(
                        type(var)
                    )
                )
        else:
            raise NotImplementedError(
                "Expected rollout object to be a list got {}".format(type(var))
            )

    def cal_knn_reward(self,
        episodic_memory,
        current_c_state,
        k = 10
        ):
        
        current_c_state = copy.deepcopy(current_c_state)
        batch_size,nb_agent,_ = current_c_state.shape
        if len(episodic_memory) == 0 or len(episodic_memory)<self.nb_agent*5: return torch.zeros(batch_size,nb_agent).to('cuda')
        memory  = torch.stack(episodic_memory)
        reward_list = []
        
        for i in range(nb_agent):
        
            agent_obs = current_c_state[:,i,i*2:(i+1)*2].unsqueeze(0)
            agent_obs[:,:,0] *= self.x
            agent_obs[:,:,1] *= self.y
            
            dist = torch.norm(memory-agent_obs,dim=2,p=None) 
        
            if memory.shape[0] > k:
                knn,_ = torch.topk(dist,k,dim=0,largest=False)
            else:
                knn = dist

            knn = knn.permute(1,0) 
            s = torch.mean(knn,dim=-1)
            reward_list.append(s)
        result = torch.stack(reward_list,dim=1).cuda()

        return result
        