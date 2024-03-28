import argparse
from main import main

n_layers = 4
def print_to_file_and_terminal(message, file):
    print(message)
    print(message, file=file)

def similarity_major_greedy(t):
    with open(f'1_results_tables/greedy-{t}.log', 'w') as f:
        n_list = [2]*n_layers
        k_list = [64]*n_layers 
        for j in range(t):
            print_to_file_and_terminal("Iteration: " + str(j+1), f)
            print_to_file_and_terminal("n_list: " + str(n_list), f)
            print_to_file_and_terminal("k_list: " + str(k_list), f)
            similarities = main(n_list, k_list)
            print_to_file_and_terminal("Similarities: " + str(similarities), f)
            max_index = similarities.index(max(similarities))
            min_index = similarities.index(min(similarities))
           
            k_list[max_index] = max(1, n_list[max_index]//2) 
            if k_list[max_index] == 16:
               n_list[max_index] = max(1, n_list[max_index]//2) 
               k_list[max_index] = 8192 
           
            k_list[min_index] *= 2
            if k_list[min_index] == 8192:
                n_list[min_index] *= 2
                k_list[min_index] = 16 
# pass in t from the command line 
argsparse = argparse.ArgumentParser()
argsparse.add_argument('t', type=int)
args = argsparse.parse_args()
similarity_major_greedy(args.t)
            
