!pip install gymnasium
!pip install "gymnasium[toy-text]"
import gymnasium as gym
import numpy as np
import random
# sample environment
env=gym.make("CliffWalking-v1")
print(env.observation_space.n)
print(env.action_space.n)

starting_state, _ = env.reset()
print(starting_state)
#sarsa
gamma=0.99
alpha=0.5
epsilon=0.1
episodes=500
#qtable
Q=np.zeros((48,4))
#polacy-episolon greedy

def epsilon_greedy(state):
    if random.random()<epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
    
#code for sarsa
for episode in range (episodes):

    env=gym.make("CliffWalking-v1")

    done=False
    state, _ =env.reset()
    action=epsilon_greedy(state)

    total_reward=0
    episode_len=0

    while not done:
        next_state, reward,terminated,trucated, _ =env.step(action)
        next_action=epsilon_greedy(next_state)

        #SARSA update
        Q[state,action]+=alpha * (reward+gamma*(Q[next_state,next_action]- Q[state,action]))

        state=next_state
        action=next_action

        total_reward+=reward
        episode_len+=1

    print(f"episode={episode+1}/500 = {total_reward} & ep length={episode_len}")   
    env.close()
env=gym.make("CliffWalking-v1",render_mode="human")
state, _ =env.reset()
done=False
totla_reward=0
episode_len=0

while not done:
    action=np.argmax(Q[state])
    next_state,reward,terminated,truncated,_=env.step(action)
    done=terminated or truncated

    total_reward+=reward
    episode_len+=1

print(f"total reward={total_reward}&episode len={episode_len}")
env.close()
what is the out put of this program
