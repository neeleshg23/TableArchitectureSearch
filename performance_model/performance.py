import pandas as pd
import argparse

def get_flops_conv2d(data, n_list, k_list):
    nn_conv1 = data.loc[data['name'] == 'conv1', 'flops'].item()
    nn_conv2 = data.loc[data['name'] == 'conv2', 'flops'].item()
    nn_conv3 = data.loc[data['name'] == 'conv3', 'flops'].item()
    
    flops_conv1 = data.loc[data['name'] == 'conv1', 'H'].item() * data.loc[data['name'] == 'conv1', 'W'].item() * pow(2, n_list[0]) * (k_list[0] + data.loc[data['name'] == 'conv1', 'out_channels'].item() * n_list[0])
    flops_conv2 = data.loc[data['name'] == 'conv2', 'H'].item() * data.loc[data['name'] == 'conv2', 'W'].item() * pow(2, n_list[1]) * (k_list[1] + data.loc[data['name'] == 'conv2', 'out_channels'].item() * n_list[1])
    flops_conv3 = data.loc[data['name'] == 'conv3', 'H'].item() * data.loc[data['name'] == 'conv3', 'W'].item() * pow(2, n_list[2]) * (k_list[2] + data.loc[data['name'] == 'conv3', 'out_channels'].item() * n_list[2])
    return nn_conv1+nn_conv2+nn_conv3, flops_conv1 + flops_conv2 + flops_conv3

def get_storage_conv2d(data, n_list, k_list):
    nn_conv1 = data.loc[data['name'] == 'conv1', 'parameters'].item()
    nn_conv2 = data.loc[data['name'] == 'conv2', 'parameters'].item()
    nn_conv3 = data.loc[data['name'] == 'conv3', 'parameters'].item()
    
    storage_conv1 = data.loc[data['name'] == 'conv1', 'H'].item() * data.loc[data['name'] == 'conv1', 'W'].item() * pow(2, n_list[0]) + data.loc[data['name'] == 'conv1', 'out_channels'].item() * pow(2, n_list[0]) * pow(2, k_list[0])
    storage_conv2 = data.loc[data['name'] == 'conv2', 'H'].item() * data.loc[data['name'] == 'conv2', 'W'].item() * pow(2, n_list[1]) + data.loc[data['name'] == 'conv2', 'out_channels'].item() * pow(2, n_list[1]) * pow(2, k_list[1])
    storage_conv3 = data.loc[data['name'] == 'conv3', 'H'].item() * data.loc[data['name'] == 'conv3', 'W'].item() * pow(2, n_list[2]) + data.loc[data['name'] == 'conv3', 'out_channels'].item() * pow(2, n_list[2]) * pow(2, k_list[2])
    return nn_conv1 + nn_conv2 + nn_conv3, storage_conv1 + storage_conv2 + storage_conv3

data = [
    ['conv1', 'Conv2d', '(1,28,28)', '(12,28,28)', 5, 2, 1, 1, 12, 28, 28, 312, 244608],
    ['pool', 'MaxPool2d', '(12,14,14)', '(12,7,7)', None, None, None, None, None, None, None, 0, 3136],
    ['conv2', 'Conv2d', '(12,14,14)', '(16,14,14)', 5, 2, 1, 12, 16, 14, 14, 4816, 943936],
    ['conv3', 'Conv2d', '(16,7,7)', '(16,7,7)', 3, 1, 1, 16, 16, 7, 7, 2320, 113680],
    ['drop', 'Dropout2d', '(16,7,7)', '(16,7,7)', None, None, None, None, None, None, None, 0, 0],
    ['fc', 'Linear', '(784)', '(10)', None, None, None, 784, 10, None, None, 7850, 7840]
]
columns = ['name', 'type', 'in_shape', 'out_shape', 'K', 'P', 'S', 'in_channels', 'out_channels', 'H', 'W', 'parameters', 'flops']
df = pd.DataFrame(data, columns=columns)
print(df)

if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--subspace", "-s", type=str, default='0,0,0')
    argsparse.add_argument("--kprototype", "-k", type=str, default='1,1,1')
    args = argsparse.parse_args()
    n_list = [int(x) for x in args.subspace.split(',')]
    k_list = [int(x) for x in args.kprototype.split(',')]
    og_flops, flops = get_flops_conv2d(df, n_list, k_list)
    og_storage, storage = get_storage_conv2d(df, n_list, k_list)
    
    print(f"NN FLOPS: {og_flops}")
    print(f"NN Storage: {og_storage}")
    
    print(f"FLOPS: {flops}")
    print(f"Storage: {storage}")