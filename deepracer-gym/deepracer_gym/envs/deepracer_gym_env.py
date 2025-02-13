import numpy as np
import gym
from deepracer_gym.zmq_client import DeepracerEnvHelper
import time
import warnings
class DeepracerGymEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)
        self.deepracer_helper = DeepracerEnvHelper()
        self.last_step_time = None
        self.max_step_time = 0.03 # seconds
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        return observation
    
    def step(self, action):
        
        if self.last_step_time is not None:
            time_delta = time.time() - self.last_step_time - self.max_step_time
        else:
            time_delta = -1
        done = False
        n_repeats = 0

        # This is to emulate async nature of the real world track
        # If action is not returned within required time limit, the same action would be repeated
        while (not done) and time_delta > 0:
            time_delta -= self.max_step_time
            observation, reward, done, info = self._step_sim(self.last_action)
            n_repeats += 1
        
        if n_repeats > 0:
            warn_msg = f"Action was repeated {n_repeats} times, try to reduce model step time to {self.max_step_time} seconds" 
            warnings.warn(warn_msg)
        
        if not done:
            observation, reward, done, info = self._step_sim(action)

        self.last_action = action
        self.last_step_time = time.time()
        return observation, reward, done, info
    
    def _step_sim(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        return observation, reward, done, info

if __name__ == '__main__':
    env = DeepracerGymEnv()
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            steps_completed = 0
            total_reward = 0
