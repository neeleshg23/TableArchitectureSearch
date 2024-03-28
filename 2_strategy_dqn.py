from numpy import log2
import torch
import random
from collections import deque
from main import main as run
from models_amm.amm.kmeans import set_gpu, get_gpu

import torch.nn as nn
import torch.optim as optim

def print_to_file_and_terminal(message, file):
    print(message)
    print(message, file=file)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
file = open('1_results_tables/dqn.log', 'w')
 
class HyperparameterOptimizationEnv:
    def __init__(self, N):
        self.N = N
        self.state = None
        self.reset()

    def reset(self):
        # Reset state to initial conditions
        self.state = [0]*self.N + [4]*self.N  # Assuming initial S=1, K=16 for simplicity
        return self.state
    
    def step(self, action):
        # Determine parameter to modify (S or K), which layer, and action (increase or decrease)
        increase = action % 2 == 0  # Define increase for even actions, decrease for odd
        param_index = action // 2  # Determines which S or K to modify
        is_N = param_index < self.N  # True for S, False for K
        
        # Apply the action
        if is_N:
            layer_index = param_index
            if increase:
                self.state[layer_index] = min(self.state[layer_index] + 1, 4)  # Assuming max log2(S) is 4
            else:
                self.state[layer_index] = max(self.state[layer_index] - 1, 0)  # Assuming min log2(S) is 0
        else:
            layer_index = param_index - self.N
            if increase:
                self.state[self.N + layer_index] = min(self.state[self.N + layer_index] + 1, 13)  # max log2(K) is 13
            else:
                self.state[self.N + layer_index] = max(self.state[self.N + layer_index] - 1, 4)  # min log2(K) is 4
        
        # Rerun with new size hyperparameters and obtain the next cosine similarities
        n_list, k_list = [int(2**s) for s in self.state[:self.N]], [int(2**k) for k in self.state[self.N:]]
        print_to_file_and_terminal("n_list: " + str(n_list), file)
        print_to_file_and_terminal("k_list: " + str(k_list), file)
        cosine_similarities = run(n_list, k_list)
        print_to_file_and_terminal("Cosine Similarities: " + str(cosine_similarities), file) 
        # Compute reward and update state
        reward = sum(cosine_similarities)
        self.state = [log2(s) for s in n_list] + [log2(k) for k in k_list]  # Update state in log space
        
        done = reward > 0.9*self.N # Terminating condition when we have high cosine similarities 
        
        return self.state, reward, done, {}


def train_dqn(model, episodes, gamma, epsilon, epsilon_decay, epsilon_min, batch_size):
    optimizer = optim.Adam(model.parameters())
    memory = deque(maxlen=20000)
    env = HyperparameterOptimizationEnv(N=4)  # Example with 4 layers
    
    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        i = 1
        
        while True:
            # Epsilon-greedy action selection
            print_to_file_and_terminal("Iteration: " + str(i), file)
            if random.random() > epsilon:
                action = model(state).argmax().item()
            else:
                # TODO: 16 is hardcoded here, but should be 4*model.output_size
                action = random.randrange(16) # model.output_size = 8
            
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            
            # Store transition in memory
            memory.append((state, action, reward, next_state, done))
            
            # Experience replay
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                # Train model on the batch of transitions
                
            state = next_state
            total_reward += reward
            
            if done:
                break
            
            i+=1
        
        # Update epsilon
        epsilon = max(epsilon_min, epsilon_decay*epsilon)
        
        print_to_file_and_terminal(f'Episode: {episode}, Total Reward: {total_reward}', file)

def main():
    set_gpu(1)
    N = 4
    input_size = 2*N  # 4 layers, each with log2(S) and log2(K)
    output_size = 2*N # 4 layers, each with 2 actions (increase/decrease S, increase/decrease K)
    model = DQN(input_size, output_size)
    device = torch.device(f'cuda:{get_gpu()}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) 
    episodes = 10 
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    
    train_dqn(model, episodes, gamma, epsilon, epsilon_decay, epsilon_min, batch_size)
    
main()
