

from ast import Break
from genericpath import exists
from logging import exception
from adept.env.base.env_module import EnvModule
from adept.preprocess.observation import ObsPreprocessor
from adept.preprocess.ops import (
    CastToFloat,
    GrayScaleAndMoveChannel,
    FrameStackCPU,
    FromNumpy,
)
from adept.env._spaces import Space
from adept.env.env_ucs.util.config_3d import Config
from adept.env.env_ucs.util.utils import IsIntersec

from pandas import Interval

import pandas as pd 
import numpy as np
import copy
import random
import gym
import pickle
import math
import sys
import warnings
import torch
import operator
import traceback
from functools import reduce

from itertools import combinations
from gym.utils import seeding
from gym import spaces



import sys
import os
sys.path.append(os.getcwd())


np.seterr(all="raise")


class EnvUCS(EnvModule):
    ids = ['EnvUCS-v0']
    args = {}

    def __init__(self, args=None, **kwargs):

        if args != None:
            self.args = args

        self.config = Config(args)

        self.DISCRIPTION = self.config('description')
        self.CONTROLLER = self.config("controller_mode")

        self.MAP_X = self.config("map_x")
        self.MAP_Y = self.config("map_y")

        self.WEIGHTED_MODE = self.config("weighted_mode")
        self.ACTION_MODE = self.config("action_mode")
        self.SCALE = self.config("scale")
        self.NUM_UAV = self.config("num_uav")
        self.INITIAL_ENERGY = self.config("initial_energy")
        self.EPSILON = self.config("epsilon")
        self.DEBUG_MODE = self.config("debug_mode")
        self.TEST_MODE = self.config("test_mode")
        self.USER_LENGTH_REWARD = self.config("user_length_reward")
        self.ACTION_ROOT = self.config("action_root")
        self.MAX_EPISODE_STEP = self.config("max_episode_step")
        self._max_episode_steps = self.MAX_EPISODE_STEP
        self.TIME_SLOT = self.config("time_slot")
        self.TOTAL_TIME = self.MAX_EPISODE_STEP * self.TIME_SLOT
        self.UAV_SPEED = self.config("uav_speed")
        self.SAVE_COUNT = self.args['seed']
        self.LOG_REWARD = self.config("log_reward")
        self.POI_VISIBLE_NUM  = self.config("poi_visible_num")
        self.SMALL_OBS_NUM = self.config("small_obs_num")


        self.UPDATE_NUM = self.config("update_num")

        self.COLLECT_RANGE = self.config("collect_range")
        self.POI_INIT_DATA = self.config("poi_init_data")
        self.POI_NUM = self.config("poi_num")
        self.POI_CLUSTERS_NUM = self.config("poi_cluster_num")
        self.SINGLE_CLUSTER_NUM = self.POI_NUM//self.POI_CLUSTERS_NUM
        self.RATE_THRESHOLD = self.config("rate_threshold")
        self.EMERGENCY_BAR = self.config("emergency_threshold")
        self.EMERGENCY_REWARD_RATIO = self.config("emergency_reward_ratio")
        self.ADD_EMERGENCY = self.config("add_emergency")
        self.EMERGENCY_PENALTY = self.config("emergency_penalty")
        self.RATE_DISCOUNT = self.config("rate_discount")
        self.UPDATE_USER_NUM = self.config("update_user_num")
        self.USER_DATA_AMOUNT = self.config("user_data_amount")
        self.CONCAT_OBS = self.config("concat_obs")

        self.n_agents = self.NUM_UAV
        self.episode_limit = self._max_episode_steps
        self.n_actions = 1 if self.ACTION_MODE else self.ACTION_ROOT
        self.agent_field = self.config("agent_field")
        self.reward_scale = self.config('reward_scale')

        self.MAX_FIY_DISTANCE = self.TIME_SLOT*self.UAV_SPEED/self.SCALE

        self.RATE_MAX = self._get_data_rate((0, 0), (0, 0))
        self.RATE_MIN = self._get_data_rate((0, 0), (self.MAP_X, self.MAP_Y))
        
        self.OBSTACLE  = self.config('obstacle')

        self._uav_energy = np.asarray(
            [self.config("initial_energy") for i in range(self.NUM_UAV)], dtype=np.float64)
        self._uav_position = np.asarray(
            [[self.config("init_position")[i][0], self.config("init_position")[i][1]] for i in
             range(self.NUM_UAV)],
            dtype=np.float16)
        self._get_energy_coefficient()
        
        self.POI_CLUSTERS_NUM = 1
        self.OLD_CLUSTERS_NUM = 1
        self.SINGLE_CLUSTER_NUM = self.POI_NUM

        if self.ACTION_MODE == 1:
            self.gym_action_space = spaces.Box(min=-1, max=1, shape=(2,))
        elif self.ACTION_MODE == 0:
            self.gym_action_space = spaces.Discrete(self.ACTION_ROOT)
        else:
            self.gym_action_space = spaces.Discrete(1)



        self._poi_position = np.load(os.path.join(self.config("data_file"),'poi_location.npy')).reshape(1, -1, 2)
        self._poi_arrival  = np.load(os.path.join(self.config("data_file"),'arrival.npy'))
        self._poi_weight= np.load(os.path.join(self.config("data_file"),'poi_weights.npy'))
        

        self._poi_value = [[] for _ in range(self.POI_NUM)]
        self._poi_last_visit = [-1 for _ in range(self.POI_NUM)]

        self._poi_position[:,:,0] *= self.MAP_X
        self._poi_position[:,:,1] *= self.MAP_Y


        self.poi_property_num = 2 + self.UPDATE_USER_NUM + 1 + 1
        self.task_property_num = 4
        info = self.get_env_info()

        obs_dict = {
            'Box': spaces.Box(low=-1, high=1, shape=(self.n_agents, info['obs_shape'])),
            'State': spaces.Box(low=-1, high=1, shape=(info['state_shape'],)),
            'available_actions':spaces.Box(low=0,high=1,shape=(self.n_agents,self.ACTION_ROOT)),
        }
        if self.SMALL_OBS_NUM !=-1: obs_dict ={'SmallBox':spaces.Box(low=-1, high=1, shape=(self.n_agents, info['small_obs'])),**obs_dict}
        self.obs_space = spaces.Dict(obs_dict)
    
        cpu_preprocessor = ObsPreprocessor(
            [FromNumpy()],
            Space.from_gym(self.obs_space),
            Space.dtypes_from_gym(self.obs_space),
        )
        gpu_preprocessor = ObsPreprocessor(
            [CastToFloat()],
            cpu_preprocessor.observation_space,
            cpu_preprocessor.observation_dtypes,
        )
        action_space = Space.from_gym(self.gym_action_space)

        super(EnvUCS, self).__init__(action_space,
                                     cpu_preprocessor, gpu_preprocessor)

        self.reset()

    def reset(self):

        self.uav_trace = [[] for i in range(self.NUM_UAV)] 
        self.uav_state = [[] for i in range(self.NUM_UAV)]  
        self.uav_energy_consuming_list = [[]
                                          for i in range(self.NUM_UAV)]  
        self.uav_data_collect = [[]
                                 for i in range(self.NUM_UAV)] 
        self.uav_task_collect = [[]
                                 for i in range(self.NUM_UAV)] 

        self.last_action = [[0, 0] for _ in range(self.NUM_UAV)]
        self.task_allocated = [[] for i in range(self.NUM_UAV)]
        self.dead_uav_list = [False for i in range(self.NUM_UAV)]

        self.update_list = [[] for i in range(self.NUM_UAV)]
        self.collect_list = [[] for i in range(self.NUM_UAV)]

        self.single_uav_reward_list = []  
        self.task_uav_reward_list = [] 
        self.episodic_reward_list = [] 
        self.fairness_list = []
        self.count_list = []


        self.poi_history = []
        self.emergency_ratio_list = [0]
        self.task_history = []
        self.aoi_history = [0]
        self.area_aoi_history = [0]
        self.activate_list = []
        self.total_poi = []
        self.total_data_collect = 0
        self.total_data_arrive = 0

        self.step_count = 0
        self.emergency_status = False

        self.uav_energy = copy.deepcopy(self._uav_energy)
        self.poi_value = copy.deepcopy(self._poi_value)

        self.uav_position = copy.deepcopy(self._uav_position)
        self.poi_position = copy.deepcopy(self._poi_position)

        self.poi_arrive_time = [[-1] for _ in range(self.POI_NUM)]
        self.poi_delta_time  = [[] for _ in range(self.POI_NUM)]
        self.poi_collect_time = [[] for _ in range(self.POI_NUM)]
        self.poi_aoi = [[] for _ in range(self.POI_NUM)]
        self.poi_wait_time = [[] for _ in range(self.POI_NUM)]

        self.poi_emergency = [[] for _ in range(self.POI_NUM)]

        self.collision_count = 0

        self.check_arrival(self.step_count)
        self.cpu_preprocessor.reset()
        return self.get_obs()

    def render(self, mode='human'):
        pass

    def step(self, action):

        action = action['Discrete']
        uav_reward = np.zeros([self.NUM_UAV])
        uav_penalty = np.zeros([self.NUM_UAV])
        uav_data_collect = np.zeros([self.NUM_UAV])
        uav_aoi_collect = np.zeros([self.NUM_UAV])
        uav_task_finish = np.zeros([self.NUM_UAV])
        uav_em_count = np.zeros([self.NUM_UAV])
        uav_trajectory = []
        for i in range(self.NUM_UAV):
            uav_trajectory.extend(self.uav_trace[i])

        for uav_index in range(self.NUM_UAV):
            self.update_list[uav_index].append([])
            self.collect_list[uav_index].append([])

            new_x, new_y, distance, energy_consuming = self._cal_uav_next_pos(uav_index,action[uav_index])
            Flag = self._judge_obstacle(self.uav_position[uav_index], (new_x, new_y))
            if not Flag:
                self.uav_position[uav_index] = (new_x, new_y)

            self.uav_trace[uav_index].append(self.uav_position[uav_index].tolist())      

            self._use_energy(uav_index, energy_consuming)
            
            collect_time = max(0,self.TIME_SLOT-distance/self.UAV_SPEED) if not Flag else 0


            r , uav_data_collect[uav_index]= self._collect_data_from_poi(
                    uav_index, 0, collect_time)

            self.uav_data_collect[uav_index].append(
                uav_data_collect[uav_index])
            
            uav_reward[uav_index] += r*(10**-3) # * (2**-4)

        done = self._is_episode_done()
        self.step_count += 1

        user_num = np.array([len(p) for p in self.poi_value])
        if not done:
            self.check_arrival(self.step_count)

        now_aoi = 0
        em_now = 0
        em_penalty = 0 
        temp_time = self.step_count*self.TIME_SLOT
        aoi_list = []
        for i in range(self.POI_NUM):
            if len(self.poi_collect_time[i])>0:
                aoi = temp_time-self.poi_collect_time[i][-1]
            else:
                aoi = temp_time
            if aoi > self.EMERGENCY_BAR * self.TIME_SLOT:
                self.poi_emergency[i].append(1)
                em_now += 1
                em_penalty += self.get_emergency_penalty((aoi-self.EMERGENCY_BAR * self.TIME_SLOT)//self.TIME_SLOT)
            now_aoi += aoi 
            aoi_list.append(aoi)

        self.poi_history.append({
                'pos': copy.deepcopy(self.poi_position).reshape(-1, 2),
                'val': copy.deepcopy(user_num).reshape(-1),
                'aoi':np.array(aoi_list)
            })
        self.aoi_history.append(now_aoi/self.POI_NUM)

        self.emergency_ratio_list.append(em_now/self.POI_NUM)
        
        for u in range(self.NUM_UAV):
            uav_reward[u] -= (em_penalty/self.POI_NUM)*self.EMERGENCY_REWARD_RATIO[0]

        info = {}
        info_old = {}
        if done:
            poi_visit_ratio = sum([int(len(p)>0) for p in self.poi_collect_time])/self.POI_NUM
            info['f_poi_visit_ratio'] = poi_visit_ratio

            for poi_index in range(self.POI_NUM):
                while len(self.poi_value[poi_index])>0:
                    index =self.poi_arrive_time[poi_index].index(self.poi_value[poi_index].pop(0)) - 1 
                    self.poi_collect_time[poi_index].append(self.TOTAL_TIME)
                    yn = self.poi_delta_time[poi_index][index]
                    tn  = self.TOTAL_TIME - self.poi_arrive_time[poi_index][index+1]

                    self.poi_wait_time[poi_index].append(self.TOTAL_TIME-self.poi_arrive_time[poi_index][index+1])
                    self.poi_aoi[poi_index].append(yn*tn+0.5*yn*yn)
    
                    if len(self.poi_value[poi_index]) == 0:
                        self.poi_aoi[poi_index].append(0.5*tn*tn)
    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                info = self.summary_info(info)
                info_old = copy.deepcopy(info)
                info = self.save_trajectory(info)
                
        global_reward = np.mean(uav_reward) + np.mean(uav_penalty)

        obs = self.get_obs(None, None)
        self.episodic_reward_list.append(global_reward)
        reward_new = [global_reward] + uav_reward.tolist()
        return obs, reward_new, done, info_old

    def summary_info(self, info):
        
        poi_weights = copy.deepcopy(self._poi_weight)/np.mean(self._poi_weight)
        
        t_e = np.sum(np.sum(self.uav_energy_consuming_list))

        total_arrive_user  = 0
        collect_user = 0 
        for index,p in enumerate(self.poi_collect_time):
            total_arrive_user += len(p)
            collect_user += np.sum([c < self.TOTAL_TIME for c in p])

        mean_aoi = np.sum([np.sum(p) for p in self.poi_aoi])/(self.POI_NUM*self.TOTAL_TIME*self.TIME_SLOT)
        weighted_mean_aoi = np.sum([np.sum(p)*poi_weights[index] for index,p in enumerate(self.poi_aoi)])/(self.POI_NUM*self.TOTAL_TIME*self.TIME_SLOT)

        em_coef = 480*self.TIME_SLOT*1000/(self.TIME_SLOT*self.POI_NUM)
        weighted_bar_aoi = np.sum([(np.sum(p)+np.sum(self.poi_emergency[index])*self.TIME_SLOT*em_coef)*poi_weights[index]  for index,p in enumerate(self.poi_aoi)])/(self.POI_NUM*self.TOTAL_TIME*self.TIME_SLOT)

  
        info['a_poi_collect_ratio'] = float(collect_user / total_arrive_user)
        info['b_emergency_violation_ratio'] = (np.sum(self.emergency_ratio_list)/self.step_count).item()
        info['c_emergency_time'] = (np.sum(self.emergency_ratio_list)*self.POI_NUM).item()
        info['d_aoi'] = mean_aoi.item()
        info['e_weighted_aoi'] = weighted_mean_aoi.item()
        info['f_weighted_bar_aoi'] = weighted_bar_aoi.item()
        
        info['h_total_energy_consuming'] = t_e.item()
        info['h_energy_consuming_ratio'] = t_e / (self.NUM_UAV*self.INITIAL_ENERGY)
        info['f_episode_step'] = self.step_count

 
        return info

    def save_trajectory(self, info):
        if self.TEST_MODE:
            info['uav_trace'] = self.uav_trace
            info['update_list'] = self.update_list
            info['collect_list'] = self.collect_list
            max_len = max((len(l) for l in self.uav_data_collect))
            new_matrix = list(
                map(lambda l: l + [0] * (max_len - len(l)), self.uav_data_collect))
            info['uav_collect'] = np.sum(new_matrix, axis=0).tolist()
            info['poi_history'] = self.poi_history
            info['reward_history'] = self.episodic_reward_list
            info['uav_reward'] = self.single_uav_reward_list
            info['poi_arrival'] = self._poi_arrival
            info['aoi_collect'] = self.poi_collect_time
            info['aoi_arrival'] = self.poi_arrive_time
            path = self.args['save_path'] + \
                '/{}_{}.txt'.format(self.DISCRIPTION, self.SAVE_COUNT)

            self.save_variable(info, path)
            info = {}

        return info

    def save_variable(self, v, filename):
        print('save variable to {}'.format(filename))
        f = open(filename, 'wb')
        pickle.dump(v, f)
        f.close()
        return filename

    def p_seed(self, seed=None):
        pass

    def _cal_distance(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_distance function only for 2d vector'
        distance = np.sqrt(
            np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(pos1[1] * self.SCALE
                                                                                - pos2[1] * self.SCALE, 2) + np.power(
                100, 2))
        return distance

    def _cal_theta(self, pos1, pos2):
        assert len(pos1) == len(
            pos2) == 2, 'cal_theta function only for 3d vector'
        r = np.sqrt(np.power(pos1[0] * self.SCALE - pos2[0] * self.SCALE, 2) + np.power(
            pos1[1] * self.SCALE - pos2[1] * self.SCALE, 2))
        h = 100
        theta = math.atan2(h, r)
        return theta

    def _cal_energy_consuming(self, move_distance):
        moving_time = move_distance/self.UAV_SPEED
        hover_time = self.TIME_SLOT - moving_time
        return self.Power_flying*moving_time + self.Power_hovering*hover_time

    def _cal_uav_next_pos(self, uav_index, action):
        if self.ACTION_MODE == 1:
            dx, dy = self._get_vector_by_theta(action)
        else:
            dx, dy = self._get_vector_by_action(int(action))
        self.last_action[uav_index] = [dx, dy]
        distance = np.sqrt(np.power(dx * self.SCALE, 2) +
                           np.power(dy * self.SCALE, 2))
        energy_consume = self._cal_energy_consuming(distance)

        if self.uav_energy[uav_index] >= energy_consume:
            new_x, new_y = self.uav_position[uav_index] + [dx, dy]
        else:
            new_x, new_y = self.uav_position[uav_index]

        return new_x, new_y, distance, min(self.uav_energy[uav_index], energy_consume)

    
    def _collect_data_from_poi(self, uav_index, cluster_index=-1, collect_time=0):
        count = 0
        position_list = []

        reward_list = []
        if collect_time >= 0:
            for poi_index, (poi_position, poi_value) in enumerate(zip(self.poi_position[cluster_index], self.poi_value)):
                d = self._cal_distance(
                    poi_position, self.uav_position[uav_index])
                if d < self.COLLECT_RANGE and len(poi_value)>0:
                    position_list.append((poi_index, d))
            position_list = sorted(position_list, key=lambda x: x[1])

            update_num = min(len(position_list), self.UPDATE_NUM)

            now_time = (self.step_count+1) * self.TIME_SLOT - collect_time
            for i in range(update_num):
                poi_index = position_list[i][0]
                rate = self._get_data_rate(
                    self.uav_position[uav_index], self.poi_position[cluster_index][poi_index])
                if rate <= self.RATE_THRESHOLD:
                    rate = 0
                update_user_num = min(50,len(self.poi_value[poi_index]))
                delta_t = self.USER_DATA_AMOUNT/rate
                weight = 1 if not self.WEIGHTED_MODE else self._poi_weight[poi_index]
   
                for u in range(update_user_num):
                    if now_time+delta_t>=(self.step_count+1) * self.TIME_SLOT:
                        break 
                    if now_time <= self.poi_value[poi_index][0]: break;
                    index = self.poi_arrive_time[poi_index].index(self.poi_value[poi_index].pop(0)) - 1 
                    self.poi_collect_time[poi_index].append(now_time)
                    yn = self.poi_delta_time[poi_index][index]
                    tn  = max(0,now_time - self.poi_arrive_time[poi_index][index+1])
                    assert tn>=0 and yn>0
                    self.poi_aoi[poi_index].append(yn*tn+0.5*yn*yn)
                    self.poi_wait_time[poi_index].append(now_time-self.poi_arrive_time[poi_index][index+1])
                    reward = yn
                    reward_list.append(reward*weight)
                    now_time+=delta_t
                    assert now_time <= (self.step_count+1) * self.TIME_SLOT+1
                  
                if now_time>=(self.step_count+1) * self.TIME_SLOT:
                    break 
        return  sum(reward_list),len(reward_list)


    def _get_vector_by_theta(self, action):
        theta = action[0] * np.pi
        l = action[1]+1
        dx = l * np.cos(theta)
        dy = l * np.sin(theta)
        return dx, dy

    def _get_vector_by_action(self, action):
        single = 1.5
        base = single/math.sqrt(2)
        action_table = [
            [0, 0],
            [-base, base],
            [0, single],
            [base, base],
            [-single, 0],
            [single, 0],
            [-base, -base],
            [0, -single],
            [base, -base],

            [0,self.MAX_FIY_DISTANCE],
            [0,-self.MAX_FIY_DISTANCE],
            [self.MAX_FIY_DISTANCE,0],
            [-self.MAX_FIY_DISTANCE,0],
        ]
        return action_table[action]


    def _is_uav_out_of_energy(self, uav_index):
        return self.uav_energy[uav_index] < self.EPSILON

    def _is_episode_done(self):
        if (self.step_count + 1) >= self.MAX_EPISODE_STEP:
            return True
        else:
            for i in range(self.NUM_UAV):
                if self._judge_obstacle(None, self.uav_position[i]):
                    print('cross the border!')
                    return True
            # return np.bool(np.all(self.dead_uav_list))
        return False

    def _judge_obstacle(self, cur_pos, next_pos):
        if self.ACTION_MODE==2 or self.ACTION_MODE == 3: return False
        if cur_pos is not None:
            for o in self.OBSTACLE:
                vec = [[o[0],o[1]],
                        [o[2],o[3]],
                        [o[4],o[5]],
                        [o[6],o[7]]]
                if IsIntersec(cur_pos,next_pos,vec[0],vec[1]):
                    return True
                if IsIntersec(cur_pos,next_pos,vec[1],vec[2]):
                    return True
                if IsIntersec(cur_pos,next_pos,vec[2],vec[3]):
                    return True
                if IsIntersec(cur_pos,next_pos,vec[3],vec[0]):
                    return True
          
        if (0 <= next_pos[0] <= self.MAP_X) and (0 <= next_pos[1] <= self.MAP_Y):
            return False
        else:
            return True
        
    def _use_energy(self, uav_index, energy_consuming):
        self.uav_energy_consuming_list[uav_index].append(
            min(energy_consuming, self.uav_energy[uav_index]))
        self.uav_energy[uav_index] = max(
            self.uav_energy[uav_index] - energy_consuming, 0)

        if self._is_uav_out_of_energy(uav_index):
            if self.DEBUG_MODE:
                print("Energy should not run out!")
            self.dead_uav_list[uav_index] = True
            self.uav_state[uav_index].append(0)
        else:
            self.uav_state[uav_index].append(1)

    def _get_energy_coefficient(self):

        P0 = 58.06  # blade profile power, W
        P1 = 79.76  # derived power, W
        U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
        v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
        d0 = 0.2  # fuselage drag ratio
        rho = 1.225  # density of air,kg/m^3
        s0 = 0.05  # the rotor solidity
        A = 0.503  # the area of the rotor disk, m^2
        Vt = self.config("uav_speed")  # velocity of the UAV,m/s ???

        self.Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
            P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
            0.5 * d0 * rho * s0 * A * Vt ** 3

        self.Power_hovering = P0 + P1

    def _get_data_rate(self, uav_position, poi_position):
        eta = 2
        alpha = 4.88
        beta = 0.43
        distance = self._cal_distance(uav_position, poi_position)
        theta = self._cal_theta(uav_position, poi_position)
        path_loss = 54.05+ 10 * eta * math.log10(distance) + (-19.9) / (1 + alpha *math.exp(-beta * (theta - alpha)))  
        w_tx = 20  
        w_noise = -104 
        w_s_t = w_tx - path_loss - w_noise 
        w_w_s_t = math.pow(10, (w_s_t - 30) / 10)  
        bandwidth = 20e6  
        data_rate = bandwidth * math.log2(1 + w_w_s_t)
        return data_rate / 1e6


    def get_obs(self, aoi_now=None, aoi_next=None):
    
        agents_obs = [self.get_obs_agent(i) for i in range(self.NUM_UAV)]
        agents_obs = np.vstack(agents_obs)
        
        obs_dict = {
            'Box': agents_obs,
            'State': self.get_state() if not self.CONCAT_OBS else self.get_concat_obs(agents_obs),
            'available_actions':self.get_avail_actions()
        }

        if self.SMALL_OBS_NUM != -1:
            obs_dict = {'SmallBox': np.vstack([self.get_obs_agent(i,visit_num=self.SMALL_OBS_NUM) for i in range(self.NUM_UAV)]),**obs_dict}
           
        return self._wrap_observation(obs_dict)

    def get_obs_agent(self, agent_id, cluster_id=-1, global_view=False,visit_num=None):
        
        if visit_num is None:
            visit_num = self.POI_VISIBLE_NUM

        if global_view:
            distance_limit = 1e10
        else:
            distance_limit = self.agent_field

        cluster_id = 0 
        poi_position_all = self.poi_position[cluster_id]
        poi_value_all = self.poi_value

        
        obs = []

        for i in range(self.NUM_UAV):
            if i == agent_id:
                obs.append(self.uav_position[i][0] / self.MAP_X)
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            elif self._cal_distance(self.uav_position[agent_id], self.uav_position[i]) < distance_limit:
                obs.append(self.uav_position[i][0] / self.MAP_X)
                obs.append(self.uav_position[i][1] / self.MAP_Y)
            else:
                obs.extend([0, 0])
                
        if visit_num == -1:
            for poi_index, (poi_position, poi_value) in enumerate(zip(poi_position_all, poi_value_all)):
                d = self._cal_distance(
                    poi_position, self.uav_position[agent_id])
                if d < distance_limit:
                    obs.append(
                        (poi_position[0]) / self.MAP_X)
                    obs.append(
                        (poi_position[1]) / self.MAP_Y)
                
                    obs.append(len(poi_value)/240)

                    if len(self.poi_collect_time[poi_index])>0: 
                        obs.append(((self.step_count)*self.TIME_SLOT-self.poi_collect_time[poi_index][-1])/self.TOTAL_TIME)
                    else:
                        obs.append(((self.step_count)*self.TIME_SLOT)/self.TOTAL_TIME)
        
                    delta_list = []
                    for arrive in poi_value:
                        index = self.poi_arrive_time[poi_index].index(arrive) -1 
                        delta_list.append(0) if self.poi_arrive_time[poi_index][index] <0 else delta_list.append(self.poi_arrive_time[poi_index][index]/self.TOTAL_TIME)
                        
                        if len(delta_list) == self.UPDATE_USER_NUM: break
                    if len(delta_list)<self.UPDATE_USER_NUM: delta_list+=[0 for _ in range(self.UPDATE_USER_NUM-len(delta_list))]
                    
                    obs.extend(delta_list)
                else:
                    num = self.poi_property_num-1 if self.ADD_EMERGENCY else self.poi_property_num
                    for _ in range(num):
                        obs.append(0)
        
        else:
            position_list = []
            for poi_index, (poi_position, poi_value) in enumerate(zip(self.poi_position[cluster_id], self.poi_value)):
                d = self._cal_distance(poi_position, self.uav_position[agent_id])
                if d < distance_limit:
                    position_list.append((poi_index, d))
            position_list = sorted(position_list, key=lambda x: x[1])
            exist_num = min(visit_num, len(position_list))
            for  i in range(exist_num):
                poi_index = position_list[i][0]
                obs.append(
                    (poi_position_all[poi_index][0]) / self.MAP_X)
                obs.append(
                    (poi_position_all[poi_index][1]) / self.MAP_Y)
            
                obs.append(len(poi_value_all[poi_index])/240)
                
                if len(self.poi_collect_time[poi_index])>0: 
                    obs.append(((self.step_count)*self.TIME_SLOT-self.poi_collect_time[poi_index][-1])/self.TOTAL_TIME)
                else:
                    obs.append(((self.step_count)*self.TIME_SLOT)/self.TOTAL_TIME)

                delta_list = []
                for arrive in poi_value_all[poi_index]:
                    index = self.poi_arrive_time[poi_index].index(arrive) -1 
                    delta_list.append(self.poi_arrive_time[poi_index][index]/self.TOTAL_TIME)
                    if len(delta_list) == self.UPDATE_USER_NUM: break
                if len(delta_list)<self.UPDATE_USER_NUM: delta_list+=[0 for _ in range(self.UPDATE_USER_NUM-len(delta_list))]
                obs.extend(delta_list)
            for i in range(visit_num-exist_num):
                obs.extend([0 for _ in range(self.poi_property_num)])
        
        obs.append(self.step_count/self.MAX_EPISODE_STEP)
        obs = np.asarray(obs)
        return obs

    def get_obs_size(self,visit_num=None):
        if visit_num is None:
            visit_num = self.POI_VISIBLE_NUM

        if self.CONTROLLER:
            size = 2*self.NUM_UAV+1
        else:
            size = 2*self.NUM_UAV+1

        if visit_num == -1:
            size += self.POI_NUM * self.poi_property_num
        else:
            size += visit_num * self.poi_property_num

        return size

    def get_state(self):
        state = [0]    
        return np.asarray(state)

    def get_concat_obs(self, agent_obs):
        state = np.zeros_like(agent_obs[0])
        for i in range(self.NUM_UAV):
            mask = agent_obs[i] != 0
            np.place(state, mask, agent_obs[i][mask])
        return state

    def get_state_size(self):
        if self.CONCAT_OBS:
            return self.get_obs_size()
        size = 1
        return size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return np.vstack(avail_actions)

    def get_avail_agent_actions(self, agent_id):

        avail_actions = []


        temp_x, temp_y = self.uav_position[agent_id]
        for i in range(self.ACTION_ROOT):
            dx, dy = self._get_vector_by_action(i)
            if not self._judge_obstacle((temp_x,temp_y), (dx + temp_x, dy + temp_y)):
                avail_actions.append(1)
            else:
                avail_actions.append(0)


        return np.array(avail_actions)

    def get_total_actions(self):
        return self.n_actions

    def get_num_of_agents(self):
        return self.NUM_UAV

    def close(self):
        pass

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        if self.SMALL_OBS_NUM != -1:
            env_info['small_obs'] = self.get_obs_size(self.SMALL_OBS_NUM)

        return env_info

    @classmethod
    def from_args(cls, args, seed, **kwargs):
        args['seed'] = seed
        return cls(args)

    def _wrap_observation(self, observation):
        space = self.obs_space
        # print(observation)
        if isinstance(space, spaces.Box):
            return self.cpu_preprocessor({"Box": observation})
        elif isinstance(space, spaces.Discrete):
            # one hot encode net1d inputs
            longs = torch.from_numpy(observation)
            if longs.dim() > 2:
                raise ValueError(
                    "observation is not net1d, too many dims: "
                    + str(longs.dim())
                )
            elif len(longs.dim()) == 1:
                longs = longs.unsqueeze(1)
            one_hot = torch.zeros(observation.size(0), space.n)
            one_hot.scatter_(1, longs, 1)
            return self.cpu_preprocessor({"Discrete": one_hot.numpy()})
        elif isinstance(space, spaces.MultiBinary):
            return self.cpu_preprocessor({"MultiBinary": observation})
        elif isinstance(space, spaces.Dict):
            return self.cpu_preprocessor(
                {name: obs for name, obs in observation.items()}
            )
        elif isinstance(space, spaces.Tuple):
            return self.cpu_preprocessor(
                {idx: obs for idx, obs in enumerate(observation)}
            )
        else:
            raise NotImplementedError

    def close(self):
        pass

    def check_arrival(self,step):
        delta_step = 240-self.MAX_EPISODE_STEP
        time = step*self.TIME_SLOT
        temp_arrival = self._poi_arrival[:,delta_step+step]
        for p_index in range(len(temp_arrival)):
            if temp_arrival[p_index]>0:
                self.poi_value[p_index].append(time)
                self.poi_delta_time[p_index].append(time - self.poi_arrive_time[p_index][-1])
                self.poi_arrive_time[p_index].append(time)

    def get_num_over_threshold(self):
        count = 0
        for cluster_id in range(self.POI_CLUSTERS_NUM):
            for i in range(self.SINGLE_CLUSTER_NUM):
                if self.poi_value[cluster_id][i] > self.EMERGENCY_BAR:
                    count += 1
        return count

    def get_emergency_penalty(self,emergency_times):
        emergency_mode = self.EMERGENCY_PENALTY
        if emergency_mode == 'const':
            return 1
        elif emergency_mode == 'e_t':
            return (1.02**emergency_times)*(1.02-1)+emergency_times+0.5
        else:
            raise NotImplementedError


def myfloor(x):

    a = x.astype(np.int)
    return a


