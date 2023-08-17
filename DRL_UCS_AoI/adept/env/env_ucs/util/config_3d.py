from unicodedata import name
import json5
import numpy as np
import random
'''环境配置类'''


class Config(object):

    def __init__(self,
                 args=None):
        self.default_config()
        if args is not None:
            for key in args.keys():
                self.dict[key] = args[key]
        if self.dict['dataset'] == "beijing":
                from adept.env.env_ucs.util.Beijing.env_config import dataset_config
        else:
            from adept.env.env_ucs.util.Sanfrancisco.env_config import dataset_config
        self.dict={
            **self.dict,
            **dataset_config
        }

        if self('debug_mode'):
            print("Init config")
            print("Config:")
            print(self.dict)

    def __call__(self, attr):
        assert attr in self.dict.keys(), print('key error[', attr, ']')
        return self.dict[attr]

    def save_config(self, outfile=None):
        if outfile is None:
            outfile = 'default_save.json5'
        json_str = json5.dumps(self.dict, indent=4)
        with open(outfile, 'w') as f:
            f.write(json_str)

    def default_config(self):
        self.dict = {
            "description": "default",
            # Env
            "task_id": 0,
            "action_mode": 0,  # 1 for continuous,  0 for discrete, 
            "seed": 0,
            "debug_mode": False,
            "action_root": 13,
            "max_episode_step": 240,
            "use_obs_instead_of_state": False,
            "dataset":"beijing",

            "controller_mode": True,
            "test_mode": False,

            "add_emergency": False,
            "concat_obs": True,
            "log_reward": False,
            "weighted_mode": True,
            "user_length_reward": False,
            "poi_visible_num": -1,
            "small_obs_num": -1,
    
            "reward_scale": False,
            "scale": 100,
            "max_poi_value": 1,
            "uav_speed": 20,
            "time_slot": 20,

            # Energy
            "initial_energy": 719280.0, 
            "obstacle_penalty": -1,
            "normalize": 0.1,
            "epsilon": 1e-3,

            # UAV
            "num_uav": 3,
            "agent_field": 500,

            # PoI
            "update_num": 10,
            "update_user_num": 3,
            "user_data_amount": 1,
            
            "poi_cluster_num": 1,
            "collect_range": 500,
            "poi_init_data": 0,
            "rate_threshold": 0.05,
            "rate_discount": 1,
            "emergency_threshold": 100,
            "emergency_reward_ratio": [0.5,0],
            "max_data": 1000,
            "emergency_penalty": "const",


            # Manager
            "log_path": './log_result',
        }

    def generate_poi(self):
        location = [[0, 0, 10, 10], [0, 10, 10, 20], [0, 20, 10, 30], [0, 30, 10, 40], [0, 40, 10, 50], [0, 50, 10, 60], [10, 0, 20, 10], [10, 10, 20, 20], [10, 20, 20, 30], [10, 30, 20, 40], [10, 40, 20, 50], [10, 50, 20, 60], [20, 0, 30, 10], [20, 10, 30, 20], [20, 20, 30, 30], [20, 30, 30, 40], [20, 40, 30, 50], [20, 50, 30, 60], [
            30, 0, 40, 10], [30, 10, 40, 20], [30, 20, 40, 30], [30, 30, 40, 40], [30, 40, 40, 50], [30, 50, 40, 60], [40, 0, 50, 10], [40, 10, 50, 20], [40, 20, 50, 30], [40, 30, 50, 40], [40, 40, 50, 50], [40, 50, 50, 60], [50, 0, 60, 10], [50, 10, 60, 20], [50, 20, 60, 30], [50, 30, 60, 40], [50, 40, 60, 50], [50, 50, 60, 60]],

        for l in location[0]:
            x_min, y_min, x_max, y_max = l
            poi_num = 7
            poi_list = []
            for i in range(poi_num):
                x = random.random() * (x_max - x_min) + x_min
                y = random.random() * (y_max - y_min) + y_min
                poi_list.append((x, y))
            print('[', end='')
            for index, p in enumerate(poi_list):
                if index != poi_num-1:
                    print('[{:.2f},{:.2f}],'.format(p[0], p[1]))
                else:
                    print('[{:.2f},{:.2f}]'.format(p[0], p[1]), end='')
            print('],', end='')

    def generate_task(self):
        location = [[0, 0, 20, 20], [20, 20, 40, 40], [40, 40, 60, 60], [0, 20, 20, 40], [0, 40, 20, 60],
                    [20, 0, 40, 20], [40, 0, 60, 20], [40, 20, 60, 40], [20, 40, 40, 60]]
        task_num = 5
        for i in range(task_num):
            l = location[np.random.randint(0, 9)]
            x_min, y_min, x_max, y_max = l
            poi_num = 35
            poi_list = []
            for i in range(poi_num):
                x = random.random() * (x_max - x_min) + x_min
                y = random.random() * (y_max - y_min) + y_min
                poi_list.append((x, y))
            print('[', end='')
            for index, p in enumerate(poi_list):
                if index != poi_num-1:
                    print('[{:.2f},{:.2f}],'.format(p[0], p[1]))
                else:
                    print('[{:.2f},{:.2f}]'.format(p[0], p[1]))
            print('],', end='')

    def generate_speed(self):
        poi_num = 360
        poi_list = []
        for i in range(poi_num):
            poi_list.append(np.random.poisson(2)*0.1)
        print(poi_list)


if __name__ == '__main__':
    pass
