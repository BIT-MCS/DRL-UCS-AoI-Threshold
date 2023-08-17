<<<<<<< HEAD
from operator import truediv
import sndhdr
from adept.env.env_ucs.util.utils import *
from adept.env.env_ucs.env_ucs import EnvUCS
import os
import gym
import numpy as np
import pandas as pd
import sys
import os
import time 

import torch
sys.path.append(os.getcwd())


def test(data_amount=1,num_uav=3,bar=100,snr=500):
    print('uav_num:{},snr:{},data_amount:{},bar:{}'.format(num_uav,snr,data_amount,bar))
    env = EnvUCS({   
        'test_mode': True,
        'save_path': '.',
        "controller_mode": True,
        "seed": 1,
        "action_mode": 3,
        "weighted_mode":True,
        "mip_mode":False,
        "noisy_power":-90,
        "tx_power":20,
        "render_mode":True,
        
        #
        "user_data_amount":data_amount,
        "num_uav":num_uav,
        "emergency_threshold":bar,
        "collect_range":snr,
    })
    new_obs_n = env.reset()
    total = []
    iteration = 0

    done_count = 0
    done_max = 1
    poi_collect_list = []
    task_collect_list = []
    reward_list = [0 for _ in range(4)]
    retdict = {'collection_ratio':[], 'violation_ratio':[],	'episodic_aoi':[], 'T-AoI':[], 'consumption_ratio':[]}
    
    for i in range(1):

        episode_step = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_action = []
        while True:
            #action = [np.random.randint(10),np.random.randint(10)] 
            actions = []
            for i in range(num_uav):
                a = np.random.randint(9)
                while new_obs_n['available_actions'][i][a] != 1:
                    a = np.random.randint(13)
                actions.append(a)

            action = {'Discrete': [actions[i] for i in range(num_uav)]}
            new_obs_n, rew_n, done_n, info_n = env.step(action)
            #print(torch.max(new_obs_n['Box']))
            #print(np.sum(new_obs_n['Box'].numpy()[0,:]>0))
            #print(rew_n)
            #print(new_obs_n['available_actions'])
            episode_action.append(action)
            obs_n = new_obs_n
            done = done_n
            episode_rewards[-1] += rew_n[0]  # 每一个step的总reward
            episode_step += 1
            if done:
                end = time.time()
                done_count += 1
                #print('running cost:{}'.format((end-start)/60))
                #print('\n%d th episode:\n' % iteration)
                #print('\treward:', np.sum(episode_rewards))
                #print('discount_reward',reward_list)
                total.append(np.sum(episode_rewards))
                #print([np.sum(i) for i in reward_list])
                #data = env.get_heatmap()
                #get_heatmap(data.reshape(6,6),'./heatmap.jpg',0,1)
                # print(episode_step)
           
                
                poi_collect_list.append(info_n['a_poi_collect_ratio'])
                retdict['collection_ratio'] = info_n['a_poi_collect_ratio']
                retdict['violation_ratio'] = info_n['b_emergency_violation_ratio']
                retdict['episodic_aoi'] = info_n['e_weighted_aoi']
                retdict['T-AoI'] = info_n['f_weighted_bar_aoi']
                retdict['consumption_ratio'] = info_n['h_energy_consuming_ratio']
                
                #R = RenderEnv(info_n)
                #R.start()
                obs_n = env.reset()
                episode_step = 0
                episode_rewards = [0.0]
                iteration += 1
                break

    print('collection_ratio:', np.mean(retdict['collection_ratio']) ,'violation_ratio:',np.mean(retdict['violation_ratio']),
          '\nepisodic_aoi:', np.mean(retdict['episodic_aoi']) ,'T-AoI:', np.mean(retdict['T-AoI']) , 'consumption_ratio:', np.mean(retdict['consumption_ratio']))
    print('\n')


if __name__ == '__main__':
    test()



    
    
=======
from operator import truediv
import sndhdr
from adept.env.env_ucs.util.utils import *
from adept.env.env_ucs.env_ucs import EnvUCS
import os
import gym
import numpy as np
import pandas as pd
import sys
import os
import time 

import torch
sys.path.append(os.getcwd())


def test(data_amount=1,num_uav=3,bar=100,snr=500):
    print('uav_num:{},snr:{},data_amount:{},bar:{}'.format(num_uav,snr,data_amount,bar))
    env = EnvUCS({   
        'test_mode': True,
        'save_path': '.',
        "controller_mode": True,
        "seed": 1,
        "action_mode": 3,
        "weighted_mode":True,
        "mip_mode":False,
        "noisy_power":-90,
        "tx_power":20,
        "render_mode":True,
        
        #
        "user_data_amount":data_amount,
        "num_uav":num_uav,
        "emergency_threshold":bar,
        "collect_range":snr,
    })
    new_obs_n = env.reset()
    total = []
    iteration = 0

    done_count = 0
    done_max = 1
    poi_collect_list = []
    task_collect_list = []
    reward_list = [0 for _ in range(4)]
    retdict = {'collection_ratio':[], 'violation_ratio':[],	'episodic_aoi':[], 'T-AoI':[], 'consumption_ratio':[]}
    
    for i in range(1):

        episode_step = 0
        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_action = []
        while True:
            #action = [np.random.randint(10),np.random.randint(10)] 
            actions = []
            for i in range(num_uav):
                a = np.random.randint(9)
                while new_obs_n['available_actions'][i][a] != 1:
                    a = np.random.randint(13)
                actions.append(a)

            action = {'Discrete': [actions[i] for i in range(num_uav)]}
            new_obs_n, rew_n, done_n, info_n = env.step(action)
            #print(torch.max(new_obs_n['Box']))
            #print(np.sum(new_obs_n['Box'].numpy()[0,:]>0))
            #print(rew_n)
            #print(new_obs_n['available_actions'])
            episode_action.append(action)
            obs_n = new_obs_n
            done = done_n
            episode_rewards[-1] += rew_n[0]  # 每一个step的总reward
            episode_step += 1
            if done:
                end = time.time()
                done_count += 1
                #print('running cost:{}'.format((end-start)/60))
                #print('\n%d th episode:\n' % iteration)
                #print('\treward:', np.sum(episode_rewards))
                #print('discount_reward',reward_list)
                total.append(np.sum(episode_rewards))
                #print([np.sum(i) for i in reward_list])
                #data = env.get_heatmap()
                #get_heatmap(data.reshape(6,6),'./heatmap.jpg',0,1)
                # print(episode_step)
           
                
                poi_collect_list.append(info_n['a_poi_collect_ratio'])
                retdict['collection_ratio'] = info_n['a_poi_collect_ratio']
                retdict['violation_ratio'] = info_n['b_emergency_violation_ratio']
                retdict['episodic_aoi'] = info_n['e_weighted_aoi']
                retdict['T-AoI'] = info_n['f_weighted_bar_aoi']
                retdict['consumption_ratio'] = info_n['h_energy_consuming_ratio']
                
                #R = RenderEnv(info_n)
                #R.start()
                obs_n = env.reset()
                episode_step = 0
                episode_rewards = [0.0]
                iteration += 1
                break

    print('collection_ratio:', np.mean(retdict['collection_ratio']) ,'violation_ratio:',np.mean(retdict['violation_ratio']),
          '\nepisodic_aoi:', np.mean(retdict['episodic_aoi']) ,'T-AoI:', np.mean(retdict['T-AoI']) , 'consumption_ratio:', np.mean(retdict['consumption_ratio']))
    print('\n')


if __name__ == '__main__':
    test()



    
    
>>>>>>> ac3d12f4877af33c205f99ac139d78ec10220b20
