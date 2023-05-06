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
import abc
from tkinter import W

import torch
import torch.nn as nn
from copy import deepcopy
from adept.network.base.base import BaseNetwork
from adept.network.base.network_module import NetworkModule
from adept.utils import dlist_to_listd, listd_to_dlist
from adept.network.my_net.grtxl import StableTransformerXL
from adept.network.my_net.rc import RND,Predictor_Network

class RealNetwork(NetworkModule, metaclass=abc.ABCMeta):
    """
    A neural network comprised of SubModules. Tries to be smart about
    converting dimensionality. Does not need or build submodules for unused
    source nets or heads.
    """
    args = {"example_arg1": True, "example_arg2": 5}

    def __init__(
            self,
            args,
            observation_space,
            output_space,
            gpu_preprocessor,
            net_reg
    ):
        """
        :param source_nets: Dict[ObsKey, SubModule]
        :param body_submodule: SubModule
        :param head_submodules: Dict[Dim, SubModule]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        """
        super().__init__()
        self.gpu_preprocessor = gpu_preprocessor
        self.nb_agent = args.nb_agent
        self.args = args

        obs_input_shape = observation_space['Box'][1]
        state_input_shape = observation_space['State'][0]
        action_output_shape = output_space['Discrete'][0]

        self.gamma = torch.nn.Linear(1, 16, bias=False)

        self.feature = nn.ModuleList([nn.Sequential(
                nn.Linear(obs_input_shape, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU()
            ) for _ in range(self.nb_agent)])

        if args.use_transformer:
            self.trans =  StableTransformerXL(d_input=256, n_layers=args.n_layers,n_heads=3, d_head_inner=32, d_ff_inner=64,args=args)
            self.memory = None
        


        self.actor = nn.ModuleList([nn.Sequential(
            nn.Linear(256, action_output_shape)
        ) for _ in range(self.nb_agent)])


        self.critic = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 1)
        ) for _ in range(self.nb_agent)])


        self.use_intrinsic = args.use_intrinsic
        small_box = False
        if args.small_obs_num is not None and args.small_obs_num > -1:
            int_input = observation_space['SmallBox'][1]
            small_box = True
        else:
            int_input= obs_input_shape
            
        if self.use_intrinsic:
            self.rnd = RND(state_input_shape,64)
            self.predictor = Predictor_Network(int_input,256,self.nb_agent)


    @classmethod
    def from_args(cls, args, observation_space, output_space, gpu_preprocessor, net_reg):
        """
        Construct a Modular Network from arguments.

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        :param net_reg: NetworkRegistry
        :return: ModularNetwork
        """
        return cls(
            args,
            observation_space,
            output_space,
            gpu_preprocessor,
            net_reg
        )

    def forward(self, observation_all, internals_all, is_host=False):
        """

        :param observation: Dict[str, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[str, torch.Tensor (ND)]
        :return: Tuple[
            Dict[str, torch.Tensor (1D | 2D | 3D | 4D)],
            Dict[str, torch.Tensor (ND)]
        ]
        """

        out_put_by_key_all = []
        proc_obs_all = []
        internals_result_all = []



        # if self.use_intrinsic and not is_host:
        #     reward_coef = self.rnd(observation_all['State'])
        
    
        feature_list = []
        for index in range(self.nb_agent):
            observation = {
                'Box': observation_all['Box'][:, index, ...],
                'available_actions':observation_all['available_actions'][:,index,...]}

            if self.args.small_obs_num is not None and self.args.small_obs_num > -1:
                observation['SmallBox'] = observation_all['SmallBox'][:,index,...]
            proc_obs = self.gpu_preprocessor(observation)
            out_put_by_key = {}
            internals_by_key={}
            feature_num = 0 if self.args.shared_params else index
            feature = self.feature[feature_num](proc_obs['Box'])

            if self.args.use_transformer:
                memory = torch.stack(internals_all['memory'])[:,index,...] 
                memory = memory.permute(1,2,0,3).contiguous()
                trans_state = self.trans(feature, memory)
                feature, memory = trans_state['logits'], trans_state['memory']
                internals_by_key['memory'] = memory
        
            out_put_by_key['Discrete'] = self.actor[index](feature)
                
            aux_input =  proc_obs['SmallBox'] if self.args.small_obs_num is not None and self.args.small_obs_num > -1 else proc_obs['Box']
            if self.use_intrinsic and not is_host:
                with torch.no_grad():   
                    internals_by_key['coef'] = self.rnd(aux_input)
                    internals_by_key['predictor'] = self.predictor(aux_input)[:,index]

            out_put_by_key['critic'] = self.critic[index](feature)
    
            internals_result_all.append(internals_by_key)
            out_put_by_key_all.append(out_put_by_key)
            proc_obs_all.append(proc_obs)

        out_put_by_key_all = listd_to_dlist(out_put_by_key_all)
        proc_obs_all = listd_to_dlist(proc_obs_all)
        internals_result_all = listd_to_dlist(internals_result_all)

        if self.args.use_transformer:
            for key in ['memory']:
                internals_result_all[key]=torch.stack(internals_result_all[key]).permute(3,0,1,2,4) 
                internals_result_all[key] = list(torch.unbind(internals_result_all[key],dim=0))
        

        if self.use_intrinsic and not is_host:
            for key in ['coef','predictor']:
                internals_result_all[key]=torch.stack(internals_result_all[key],dim=1)
                internals_result_all[key] = list(torch.unbind(internals_result_all[key],dim=0))
        
        for key in ['Discrete', 'critic']:
            out_put_by_key_all[key] = torch.stack(out_put_by_key_all[key])
 
        for key in ['Box']:
            proc_obs_all[key] = torch.stack(proc_obs_all[key])

        return out_put_by_key_all, internals_result_all, proc_obs_all



    def new_internals(self, device):

        """
        :param device:
        :return: Dict[
        """
        merged_internals={}
        if self.args.use_transformer:
            internals_all = []
            for i in range(self.nb_agent):
                internals_all.append(self.trans.init_memory(torch.device("cuda")))
            internals_all = torch.stack(internals_all)
            merged_internals['memory'] = internals_all

        return merged_internals #n_agent, self.n_layers+1, self.sequence_len, self.n_input,


    def to(self, device):
        super().to(device)
        return self




if __name__ == '__main__':
    pass
