import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

if torch.backends.mps.is_avilable():
    device="mps"
elif torch.cuda.is_avilable():
    device="cuda"
else :
    device="cpu"   

class Agent:
    def __init__(self,param_set):
         self.param_set=param_set
         with open("parameters.yaml","r")as f:
             all_param_set=yaml.safe_load(f)
             params=all_param_set[param_set]

         self.alpha=params["alpha"]  
         self.gamma=params["gamma"]   
         self.epsilon_init=params["epsilon_init"] 
         self.epsilon_min=params["epsilon_min"]   
         self.replay_memory_size=params["replay_memory_size"]   
         self.mini_batch_size=params["mini_batch_size"]
         self.network_sync_rate=params["network_sync_rate:10"]   
         self.reward_threshold:1000=params["reward_threshold:1000"]    

         self.loss_fn=nn.MSELoss()
         self.optimizer=None
              
    def run(self,is_training=True,render_mode="human" if render else None)
    

    
    num_states=env.observation_space.shape[0]
    num_actions=env.action_space.n

    polacy_dqn=DQN(num_states,num_actions).to(device)

    
    if is_training:
        memory=ReplayMemory(self.replay_memory_size)
        epsilon=self.epsilon_init
    for episode in itertools.count():

        state, _ = env.reset()
        episode_rewards=0
        terminated=False
        
        while not terminated:
             if is_trainig and random.random()<epsilon:
                 action=env.action_space.sample()
             else:
                 action=polacy_dqn(state).argmax()
            
                 #action = env.action_space.sample()
        
                 next_state, reward, terminated, _, _ = env.step(action)
        
                 if is_training:
                     memory.append(state,action,new_state,reward,terminated)

                     state=new_state
                     episode_rewards+=rewards
        
           
                 if terminated:
                     break

                 episode_rewards+=rewards
        print(f"episode={episode+1}with total reward={episode_rewards} & epsilon{epsilon}")
        #env.close()
