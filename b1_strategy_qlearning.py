import argparse
import numpy as np
import random
from tqdm import trange
from main import run_experiment_mask
from models_amm.amm.kmeans import set_gpu

def print_to_file_and_terminal(message, file):
    print(message)
    print(message, file=file)

def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable, learning_rate, gamma, n_range, k_range, log_file):
    for episode in trange(n_training_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        state = (random.choice(n_range), random.choice(n_range), random.choice(n_range),
                 random.choice(k_range), random.choice(k_range), random.choice(k_range))
        done = False
        
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, n_range, k_range)
            n1, n2, n3, k1, k2, k3 = state
            
            if action == 0:
                n1 = min(max(n1 + 1, n_range[0]), n_range[-1])
            elif action == 1:
                n1 = min(max(n1 - 1, n_range[0]), n_range[-1])
            elif action == 2:
                n2 = min(max(n2 + 1, n_range[0]), n_range[-1])
            elif action == 3:
                n2 = min(max(n2 - 1, n_range[0]), n_range[-1])
            elif action == 4:
                n3 = min(max(n3 + 1, n_range[0]), n_range[-1])
            elif action == 5:
                n3 = min(max(n3 - 1, n_range[0]), n_range[-1])
            elif action == 6:
                k1 = min(max(k1 + 1, k_range[0]), k_range[-1])
            elif action == 7:
                k1 = min(max(k1 - 1, k_range[0]), k_range[-1])
            elif action == 8:
                k2 = min(max(k2 + 1, k_range[0]), k_range[-1])
            elif action == 9:
                k2 = min(max(k2 - 1, k_range[0]), k_range[-1])
            elif action == 10:
                k3 = min(max(k3 + 1, k_range[0]), k_range[-1])
            elif action == 11:
                k3 = min(max(k3 - 1, k_range[0]), k_range[-1])
            
            new_state = (n1, n2, n3, k1, k2, k3)
            n_list = [n1, n2, n3]
            k_list = [k1, k2, k3]
            train_accuracy, test_accuracy, train_accuracy_amm, test_accuracy_amm, train_mse, test_mse = run_experiment_mask('n', 'm', n_list, k_list, None)
            reward = test_accuracy_amm
            
            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]) 
            
            print_to_file_and_terminal(f"Episode: {episode+1}, Step: {step+1}, State: {state}, Action: {action}, Reward: {reward:.4f}", log_file)
            
            if done:
                break
            
            state = new_state
    
    return Qtable

def greedy_policy(Qtable, state, n_range, k_range):
    return np.argmax(Qtable[state])

def epsilon_greedy_policy(Qtable, state, epsilon, n_range, k_range):
    random_int = random.uniform(0, 1)
    if random_int > epsilon:
        return greedy_policy(Qtable, state, n_range, k_range)
    else:
        return random.randint(0, 11)

def initialize_q_table(n_range, k_range):
    Qtable = {}
    for n1 in n_range:
        for n2 in n_range:
            for n3 in n_range:
                for k1 in k_range:
                    for k2 in k_range:
                        for k3 in k_range:
                            state = (n1, n2, n3, k1, k2, k3)
                            Qtable[state] = np.zeros(12)
    return Qtable

def run_q_learning(episodes, gpu):
    # Set GPU
    set_gpu(gpu)
    
    # Training parameters
    n_training_episodes = episodes
    learning_rate = 0.7
    
    # Evaluation parameters
    n_eval_episodes = 100
    
    # Hyperparameter ranges
    n_range = [0, 1, 2, 3, 4]
    k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    # Q-Learning parameters
    max_steps = 1 
    gamma = 0.95
    max_epsilon = 1.0
    min_epsilon = 0.05
    decay_rate = 0.0005
    
    # Initialize Q-table
    Qtable = initialize_q_table(n_range, k_range)
    
    # Create log file
    code = random.randint(0, 1000)
    log_file = open(f"1_results_tables/q_learning-{episodes}-{code}.log", "w")
    
    # Train Q-Learning agent
    Qtable = train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, Qtable, learning_rate, gamma, n_range, k_range, log_file)
    
    # Get the best hyperparameters
    best_state = np.unravel_index(np.argmax(Qtable), Qtable.shape)
    best_n_list = [best_state[0], best_state[1], best_state[2]]
    best_k_list = [best_state[3], best_state[4], best_state[5]]
    print_to_file_and_terminal("Best hyperparameters: n_list = {}, k_list = {}".format(best_n_list, best_k_list), log_file)
    
    log_file.close()

if __name__ == "__main__":
    # Parse command line arguments
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--episodes", "-e", type=int, default=10)
    argsparse.add_argument("--gpu", "-g", type=int, default=0)
    args = argsparse.parse_args()
    
    run_q_learning(args.episodes, args.gpu)
