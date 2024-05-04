import argparse
from main import run_experiment_mask
from models_amm.amm.kmeans import set_gpu
import random

n_layers = 3

def print_to_file_and_terminal(message, file):
    print(message)
    print(message, file=file)

def similarity_major_greedy(t):
    code = random.randint(0, 1000)
    with open(f'1_results_tables/greedy-{t}-{str(code)}.log', 'w') as f:
        n_list = [random.randint(0,4), random.randint(0,4), random.randint(0,4)]
        k_list = [random.randint(1,14), random.randint(1,14), random.randint(1,14)]
        for j in range(t):
            print_to_file_and_terminal("Iteration: " + str(j+1), f)
            print_to_file_and_terminal("n_list: " + str(n_list), f)
            print_to_file_and_terminal("k_list: " + str(k_list), f)
            
            train_accuracy, test_accuracy, train_accuracy_amm, test_accuracy_amm, train_mse, test_mse = run_experiment_mask('n', 'm', n_list[:], k_list[:], None)
            print_to_file_and_terminal(f'-- Train Accuracy -- {float(train_accuracy)}', f)
            print_to_file_and_terminal(f'-- Test Accuracy -- {float(test_accuracy)}', f)
            print_to_file_and_terminal(f'-- Train Accuracy AMM -- {float(train_accuracy_amm)}', f)
            print_to_file_and_terminal(f'-- Test Accuracy AMM -- {float(test_accuracy_amm)}', f)
            print_to_file_and_terminal(f'-- Train MSE --', f)
            for i, mse in enumerate(train_mse):
                print_to_file_and_terminal(f'Layer {i}: {mse:.4f}', f)
            print_to_file_and_terminal(f'-- Test MSE --', f)
            for i, mse in enumerate(test_mse):
                print_to_file_and_terminal(f'Layer {i}: {mse:.4f}', f)
                
            train_mse = train_mse[:3]
            test_mse = test_mse[:3]
            
            max_index = train_mse.index(max(train_mse))
            min_index = test_mse.index(min(test_mse))
            
            k_list[min_index] = max(1, k_list[min_index]-1) 
            if k_list[min_index] == 1:
               n_list[min_index] = max(0, n_list[min_index]-1) 
               k_list[min_index] = 14 
           
            k_list[max_index] = min(14, k_list[max_index]+1)
            if k_list[max_index] == 14:
                n_list[max_index] = min(4, n_list[max_index]+1) 
                k_list[max_index] = 1 

# pass in t from the command line 
argsparse = argparse.ArgumentParser()
argsparse.add_argument('--episodes', '-e', type=int, default=10)
argsparse.add_argument('--gpu', '-g', type=int, default=0)
args = argsparse.parse_args()

set_gpu(args.gpu)

similarity_major_greedy(args.episodes)
            
