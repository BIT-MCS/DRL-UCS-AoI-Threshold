#!/usr/bin/env python
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
             __           __
  ____ _____/ /__  ____  / /_
 / __ `/ __  / _ \/ __ \/ __/
/ /_/ / /_/ /  __/ /_/ / /_
\__,_/\__,_/\___/ .___/\__/
               /_/
Evaluate

Evaluates an agent after training. Computes N-episode average reward by
loading a saved model from each epoch. N-episode averages are computed by
running N env in parallel.

Usage:
    evaluate (--logdir <path>) [options]
    evaluate (-h | --help)

Required:
    --logdir <path>     Path to train logs (.../logs/<env-id>/<log-id>)

Options:
    --epoch <int>           Epoch number to load [default: None]
    --actor <str>           Name of the eval actor [default: ACActorEval]
    --gpu-id <int>          CUDA device ID of GPU [default: 0]
    --nb-episode <int>      Number of episodes to average [default: 30]
    --start <float>         Epoch to start from [default: 0]
    --end <float>           Epoch to end on [default: -1]
    --seed <int>            Seed for random variables [default: 512]
    --custom-network <str>  Name of custom network class
"""
from tkinter.filedialog import Directory
from adept.container import EvalContainer
from adept.container import Init
from adept.registry import REGISTRY as R
from adept.utils.script_helpers import parse_path, parse_none
from adept.utils.util import DotDict
import time

import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    from docopt import docopt
    args = docopt(__doc__)
    args = {k.strip("--").replace("-", "_"): v for k, v in args.items()}
    del args["h"]
    del args["help"]

    #args['logdir'] = '/home/liuchi/wh/log_env_hyper/Env3d-v0/' + 'impact_service_8_ActorLearner_ImpalaHostActor_Linear_2021-01-16_23-20-03'
    args['logdir']= '/home/liuchi/wh/logs/EnvUCS-v0/'

    args = DotDict(args)
    args.logdir = parse_path(args.logdir)
    # TODO implement Option utility
    epoch_option = parse_none(args.epoch)
    if epoch_option:
        args.epoch = int(float(epoch_option))
    else:
        args.epoch = epoch_option

    args.nb_episode = 1

    args.gpu_id = int(args.gpu_id)
    args.nb_episode = int(args.nb_episode)
    args.start = float(args.start)
    args.end = float(args.end)
    args.seed = int(time.time())
    return args

def print_list_dir(dir_path):
    dir_list =[]
    dir_files=os.listdir(dir_path) #得到该文件夹下所有的文件
    for file in dir_files:
        file_path=os.path.join(dir_path,file)  #路径拼接成绝对路径
        if os.path.isdir(file_path):  #如果目录，就递归子目录
            #print(file_path)
            dir_list.append(file_path)
    return dir_list
            


def main(args):
    """
    Run an evaluation.
    :param args: Dict[str, Any]
    :return:
    """
    args = DotDict(args)

    dir_list = print_list_dir(args.logdir)

    for dir in dir_list:
        # if 'trans' in dir:
        #     continue
        new_args = copy.deepcopy(args)
        new_args['logdir'] = dir 
        print('test path',dir)
        Init.print_ascii_logo()
        logger = Init.setup_logger(new_args.logdir, "eval")
        Init.log_args(logger, new_args)
        R.load_extern_classes(new_args.logdir)

        eval_container = EvalContainer(
            new_args.actor,
            new_args.epoch,
            logger,
            new_args.logdir,
            new_args.gpu_id,
            new_args.nb_episode,
            new_args.start,
            new_args.end,
            new_args.seed,
        )
        try:
            eval_container.run()
        finally:
            eval_container.close()


if __name__ == "__main__":
    main(parse_args())
