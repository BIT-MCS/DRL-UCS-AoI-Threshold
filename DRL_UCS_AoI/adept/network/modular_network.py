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

import torch
from copy import deepcopy
from adept.network.base.base import BaseNetwork
from adept.utils import dlist_to_listd, listd_to_dlist


class ModularNetwork(BaseNetwork, metaclass=abc.ABCMeta):
    """
    A neural network comprised of SubModules. Tries to be smart about
    converting dimensionality. Does not need or build submodules for unused
    source nets or heads.
    """

    def __init__(
            self,
            source_nets,
            body_submodule,
            head_submodules,
            output_space,
            gpu_preprocessor,
            nb_agent
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
        self.nb_agent = nb_agent

        # Source Nets
        self.source_nets_single = torch.nn.ModuleDict(
            [(key, net) for key, net in source_nets.items()]
        )

        self.source_nets = torch.nn.ModuleList([self.source_nets_single])
        for _ in range(self.nb_agent - 1):
            self.source_nets.append(deepcopy(self.source_nets_single))

        # Body
        self.body_single = body_submodule
        self.body = torch.nn.ModuleList([self.body_single])
        for _ in range(self.nb_agent - 1):
            self.body.append(deepcopy(self.body_single))

        # Heads
        self.heads_single = torch.nn.ModuleDict(head_submodules)
        self.heads = torch.nn.ModuleList([self.heads_single])
        for _ in range(self.nb_agent - 1):
            self.heads.append(deepcopy(self.heads_single))

        # Outputs
        self.output_layers_single = self._build_out_layers(output_space, self.heads_single)
        self.output_layer = torch.nn.ModuleList([self.output_layers_single])
        for _ in range(self.nb_agent - 1):
            self.output_layer.append(deepcopy(self.output_layers_single))

        self._obs_keys = list(source_nets.keys())
        self._output_keys = list(output_space.keys())
        self._output_space = output_space
        self._output_dims = set(
            [len(shape) for shape in self._output_space.values()]
        )
        # print('obs_key', self._obs_keys)
        # print('output_key', self._output_keys)
        # print('output_space', self._output_space)
        # print('output_dim', self._output_dims)
        self._check_outputs_have_heads()
        self._validate_shapes()

    @staticmethod
    def _build_out_layers(output_space, heads):
        """
        Build output_layers to match the desired output space.
        * For 1D outputs, converts uses a Linear layer
        * For 2D outputs, uses a Conv1D, kernel size 1
        * For 3D outputs, uses a 1x1 Conv
        * For 4D outputs, uses a 1x1x1 Conv

        :param output_space: Dict[OutputKey, Shape]
        :param heads: Dict[DimStr, SubModule]
        :return: ModuleDict[OutputKey, torch.nn.Module]
        """
        outputs = []
        for output_name, shape in output_space.items():
            dim = len(shape)
            if dim == 1:
                layer = torch.nn.Linear(
                    heads[str(dim)].output_shape(dim)[0], shape[0]
                )
            elif dim == 2:
                layer = torch.nn.Conv1d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            elif dim == 3:
                layer = torch.nn.Conv2d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            elif dim == 4:
                layer = torch.nn.Conv3d(
                    heads[str(dim)].output_shape(dim)[0],
                    shape[0],
                    kernel_size=1,
                )
            else:
                raise ValueError("Invalid dim {}".format(dim))
            outputs.append((output_name, layer))
        return torch.nn.ModuleDict(outputs)

    def _validate_shapes(self):
        """
        Ensures SubModule graph is valid.
        :return:
        """
        # non feature dims of source nets match non feature dim of body
        # Doesn't matter if converting to 1D
        if self.body_single.dim > 1:
            for submod in self.source_nets_single.values():
                if submod.dim > 1:
                    shape = submod.output_shape(dim=self.body_single.dim)
                    for a, b in zip(shape[1:], self.body_single.input_shape[1:]):
                        assert (
                                a == b or a == 1 or b == 1
                        ), "Source-Body conflict: {} {}".format(
                            shape, self.body_single.input_shape
                        )
            # non feature dims of body out must match non feature dims of head
            for submod in self.heads_single.values():
                if submod.dim > 1:
                    shape = self.body_single.output_shape(dim=submod.dim)
                    for a, b in zip(shape[1:], submod.input_shape[1:]):
                        assert (
                                a == b or a == 1 or b == 1
                        ), "Body-Head conflict: {} {}".format(
                            shape, submod.input_shape
                        )

        # non-feature dims of heads == non-feature dims of output shapes
        for shape in self._output_space.values():
            dim = len(shape)
            if dim > 1:
                submod = self.heads_single[str(dim)]
                head_shp = submod.output_shape(dim)
                for a, b in zip(shape[1:], head_shp[1:]):
                    assert a == b, "Head-Output conflict: {}-{}".format(
                        head_shp, shape
                    )

    def _check_outputs_have_heads(self):
        for dim in self._output_dims:
            assert str(dim) in self.heads_single

    @classmethod
    def from_args(
            cls, args, observation_space, output_space, gpu_preprocessor, net_reg
    ):
        """
        Construct a Modular Network from arguments.

        :param args: Dict[ArgName, Any]
        :param observation_space: Dict[ObsKey, Shape]
        :param output_space: Dict[OutputKey, Shape]
        :param gpu_preprocessor: ObsPreprocessor
        :param net_reg: NetworkRegistry
        :return: ModularNetwork
        """
        # Dict[ObsKey, SubModule]
        # for shape in observation space, get dim
        # instantiate input submodule of that dim
        obs_key_to_submod = {}
        for obs_key, shape in observation_space.items():

            dim = len(shape)
            if dim == 1:
                submod = net_reg.lookup_submodule(args.net1d).from_args(
                    args, shape, obs_key
                )
            elif dim == 2:
                submod = net_reg.lookup_submodule(args.net1d).from_args(
                    args, (shape[1],), obs_key
                )
            elif dim == 3:
                submod = net_reg.lookup_submodule(args.net3d).from_args(
                    args, shape, obs_key
                )
            elif dim == 4:
                submod = net_reg.lookup_submodule(args.net4d).from_args(
                    args, shape, obs_key
                )
            else:
                raise ValueError("Invalid dim: {}".format(dim))
            obs_key_to_submod[obs_key] = submod

        # SubModule
        # initialize body submodule
        body_cls = net_reg.lookup_submodule(args.netbody)
        nb_body_feature = sum(
            [
                submod.output_shape(dim=body_cls.dim)[0]
                for submod in obs_key_to_submod.values()
            ]
        )
        if body_cls.dim > 1:
            other_dims = [
                submod.output_shape(dim=body_cls.dim)[1:]
                for submod in obs_key_to_submod.values()
                if submod.dim == body_cls.dim
            ][0]
        else:
            other_dims = []
        input_shape = [nb_body_feature, ] + list(other_dims)
        body_submod = body_cls.from_args(args, input_shape, "body")

        # Dict[Dim, SubModule]
        # instantiate heads based on output_shapes
        head_submodules = {}
        for output_key, shape in output_space.items():
            dim = len(shape)
            if dim in head_submodules:
                continue
            elif dim == 1:
                submod_cls = net_reg.lookup_submodule(args.head1d)
            elif dim == 2:
                submod_cls = net_reg.lookup_submodule(args.head2d)
            elif dim == 3:
                submod_cls = net_reg.lookup_submodule(args.head3d)
            elif dim == 4:
                submod_cls = net_reg.lookup_submodule(args.head4d)
            else:
                raise ValueError("Invalid dim: {}".format(dim))
            submod = submod_cls.from_args(
                args,
                body_submod.output_shape(submod_cls.dim),
                "head" + str(dim) + "d",
            )
            head_submodules[str(dim)] = submod
        return cls(
            obs_key_to_submod,
            body_submod,
            head_submodules,
            output_space,
            gpu_preprocessor,
            args.nb_agent
        )

    def forward(self, observation_all, internals_all):
        """

        :param observation: Dict[str, torch.Tensor (1D | 2D | 3D | 4D)]
        :param internals: Dict[str, torch.Tensor (ND)]
        :return: Tuple[
            Dict[str, torch.Tensor (1D | 2D | 3D | 4D)],
            Dict[str, torch.Tensor (ND)]
        ]
        """
        ## internals_all 32,3,512
        obs_new = {}
        for key in observation_all.keys():
            obs_new[key] = observation_all[key].permute(1, 0, 2).contiguous()

        for key in internals_all.keys():
            # print(len(internals_all[key])) # 32
            internals_all[key] = torch.stack(internals_all[key]).permute(1, 0, 2).contiguous()
            # print(internals_all[key].shape) # 3,32,512

        # print('~~~', observation_all['Box'].shape)  # [3,32,304]
        obs_new = dlist_to_listd(obs_new)

        #internals_all = dlist_to_listd(internals_all)
        internals_all = [{} for i in range(self.nb_agent)]

        out_put_by_key_all = []
        merged_internals_all = []
        proc_obs_all = []

        for index in range(self.nb_agent):
            observation = obs_new[index]
            internals = internals_all[index]

            proc_obs = self.gpu_preprocessor(observation)
            # Process input network
            nxt_internals = []
            processed_inputs = []
            for key in self._obs_keys:
                # print(proc_obs[key].shape)
                result, nxt_internal = self.source_nets[index][key].forward(
                    proc_obs[key], internals, dim=self.body_single.dim
                )
                processed_inputs.append(result)
                nxt_internals.append(nxt_internal)

            # Process body
            processed_inputs = self._expand_dims(processed_inputs)
            body_out, nxt_internal = self.body[index].forward(
                torch.cat(processed_inputs, dim=1), internals
            )
            nxt_internals.append(nxt_internal)

            # Process heads
            head_out_by_dim = {}
            for dim in self._output_dims:
                cur_head = self.heads[index][str(dim)]
                head_out, next_internal = cur_head.forward(
                    self.body[index].to_dim(body_out, cur_head.dim), internals, dim=dim
                )
                head_out_by_dim[dim] = head_out
                nxt_internals.append(nxt_internal)

            # Process final outputs
            output_by_key = {}
            for key in self._output_keys:
                output = self.output_layer[index][key].forward(
                    head_out_by_dim[len(self._output_space[key])]
                )
                output_by_key[key] = output

            merged_internals = {}
            for internal in nxt_internals:
                for k, v in internal.items():
                    merged_internals[k] = v
                    # print(k, merged_internals[k].shape)

            out_put_by_key_all.append(output_by_key)
            merged_internals_all.append(merged_internals)
            proc_obs_all.append(proc_obs)

        out_put_by_key_all = listd_to_dlist(out_put_by_key_all)
        merged_internals_all = listd_to_dlist(merged_internals_all)
        proc_obs_all = listd_to_dlist(proc_obs_all)

        for key in self._output_keys:
            out_put_by_key_all[key] = torch.stack(out_put_by_key_all[key])
            # print('out_put_by_key_all[' + key + ']', out_put_by_key_all[key].shape) critic torch.Size([3, 32, 1]) Box torch.Size([3, 32, 1])

        for key in self._obs_keys:
            proc_obs_all[key] = torch.stack(proc_obs_all[key])
            # print(' proc_obs_all[' + key + ']', proc_obs_all[key].shape) [Box] torch.Size([3, 32, 304])

        for key in merged_internals_all.keys():
            #print(key, len(merged_internals_all[key]))
            merged_internals_all[key] = list(torch.stack(merged_internals_all[key]).permute(1,0,2).unbind(dim=0))
            #print('merged_internals_all[' + key + ']', len(merged_internals_all[key]))



        return out_put_by_key_all, merged_internals_all, proc_obs_all

    @staticmethod
    def _expand_dims(inputs):
        """
        Expands dimensions when input dimension is 1.

        :param inputs: List[torch.Tensor]
        :return: List[torch.Tensor]
        """
        if len(inputs[0].shape) <= 2:
            return inputs

        target_shape = max([inpt.shape[2:] for inpt in inputs])
        processed_inputs = []
        for inpt in inputs:
            if inpt.shape[2:] < target_shape:
                processed_inputs.append(inpt.expand(-1, -1, *target_shape))
            else:
                processed_inputs.append(inpt)

        return processed_inputs

    def new_internals(self, device):
        """

        :param device:
        :return: Dict[
        """
        internals_all = []
        for i in range(self.nb_agent):
            internals = [
                submod.new_internals(device) for submod in self.source_nets[i].values()
            ]
            internals.append(self.body[i].new_internals(device))
            internals += [
                submod.new_internals(device) for submod in self.heads[i].values()
            ]

            merged_internals = {}
            for internal in internals:
                for k, v in internal.items():
                    merged_internals[k] = v
            internals_all.append(merged_internals)

        internals_all = listd_to_dlist(internals_all)
        for key in internals_all.keys():
            internals_all[key] = torch.stack(internals_all[key])
            # print('key', internals_all[key].shape)
        return internals_all # n_agent, n_layers+1, self.sequence_len, self.n_input

    def to(self, device):
        super().to(device)
        # self.gpu_preprocessor = self.gpu_preprocessor.to(device)
        return self


if __name__ == '__main__':
    pass
