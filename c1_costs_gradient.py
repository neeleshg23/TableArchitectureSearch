import numpy as np
import pandas as pd

import math
import random

def get_flops_conv2d(data, n_list, k_list):
    nn_conv1 = data.loc[data['name'] == 'conv1', 'flops'].item()
    nn_conv2 = data.loc[data['name'] == 'conv2', 'flops'].item()
    nn_conv3 = data.loc[data['name'] == 'conv3', 'flops'].item()
    
    flops_conv1 = data.loc[data['name'] == 'conv1', 'H'].item() * data.loc[data['name'] == 'conv1', 'W'].item() * pow(2, n_list[0]) * (k_list[0] + data.loc[data['name'] == 'conv1', 'out_channels'].item() * n_list[0])
    flops_conv2 = data.loc[data['name'] == 'conv2', 'H'].item() * data.loc[data['name'] == 'conv2', 'W'].item() * pow(2, n_list[1]) * (k_list[1] + data.loc[data['name'] == 'conv2', 'out_channels'].item() * n_list[1])
    flops_conv3 = data.loc[data['name'] == 'conv3', 'H'].item() * data.loc[data['name'] == 'conv3', 'W'].item() * pow(2, n_list[2]) * (k_list[2] + data.loc[data['name'] == 'conv3', 'out_channels'].item() * n_list[2])
    
    # print("Flops conv1", flops_conv1)
    # print("Flops conv2", flops_conv2)
    # print("Flops conv3", flops_conv3)
    
    return flops_conv1 + flops_conv2 + flops_conv3

def get_storage_conv2d(df, n_list, k_list):
    storage_conv1 = df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * (2 ** n_list[0]) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * (2 ** (n_list[0] + k_list[0]))
    storage_conv2 = df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * (2 ** n_list[1]) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * (2 ** (n_list[1] + k_list[1]))
    storage_conv3 = df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * (2 ** n_list[2]) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * (2 ** (n_list[2] + k_list[2]))
    return storage_conv1 + storage_conv2 + storage_conv3

def calculate_gradients(params, df):
    S1, S2, S3, K1, K2, K3 = params
    gradients = np.zeros(6)
    
    # Gradients for storage cost
    gradients[0] = df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * math.log2(K1) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * K1 
    gradients[1] = df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * math.log2(K2) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * K2
    gradients[2] = df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * math.log2(K3) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * K3 
    gradients[3] = (df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * S1) / (K1 * math.log(2)) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * S1 
    gradients[4] = (df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * S2) / (K2 * math.log(2)) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * S2 
    gradients[5] = (df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * S3) / (K3 * math.log(2)) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * S3
    
    # Gradients for arithmetic operations
    gradients[0] += df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * (math.log2(K1) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * (1 / (S1 * math.log(2))))
    gradients[1] += df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * (math.log2(K2) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * (1 / (S2 * math.log(2))))
    gradients[2] += df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * (math.log2(K3) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * (1 / (S3 * math.log(2))))
    gradients[3] += (df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * S1) / (K1 * math.log(2))
    gradients[4] += (df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * S2) / (K2 * math.log(2))
    gradients[5] += (df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * S3) / (K3 * math.log(2))
    
    return gradients

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

def calculate_data_gradients(params, X, y):
    # Compute the predicted values using the current parameters
    y_pred = np.zeros_like(y)
    for i in range(len(X)):
        S1, S2, S3, K1, K2, K3 = params
        y_pred[i] = get_flops_conv2d(df, [S1, S2, S3], [K1, K2, K3]) + get_storage_conv2d(df, [S1, S2, S3], [K1, K2, K3])
    
    # Compute the mean squared error loss
    mse = np.mean((y - y_pred) ** 2)
    
    # Compute the gradients of the loss with respect to the parameters
    gradients = np.zeros(6)
    for i in range(len(X)):
        S1, S2, S3, K1, K2, K3 = params
        dS1 = 2 * (y_pred[i] - y[i]) * (df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * math.log2(K1) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * K1 )
        dS2 = 2 * (y_pred[i] - y[i]) * (df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * math.log2(K2) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * K2 )
        dS3 = 2 * (y_pred[i] - y[i]) * (df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * math.log2(K3) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * K3 )
        dK1 = 2 * (y_pred[i] - y[i]) * ((df.loc[df['name'] == 'conv1', 'H'].item() * df.loc[df['name'] == 'conv1', 'W'].item() * S1) / (K1 * math.log(2)) + df.loc[df['name'] == 'conv1', 'out_channels'].item() * S1 )
        dK2 = 2 * (y_pred[i] - y[i]) * ((df.loc[df['name'] == 'conv2', 'H'].item() * df.loc[df['name'] == 'conv2', 'W'].item() * S2) / (K2 * math.log(2)) + df.loc[df['name'] == 'conv2', 'out_channels'].item() * S2 )
        dK3 = 2 * (y_pred[i] - y[i]) * ((df.loc[df['name'] == 'conv3', 'H'].item() * df.loc[df['name'] == 'conv3', 'W'].item() * S3) / (K3 * math.log(2)) + df.loc[df['name'] == 'conv3', 'out_channels'].item() * S3 )
        gradients += np.array([dS1, dS2, dS3, dK1, dK2, dK3])
    
    gradients /= len(X)
    return gradients

# Load the dataset from the CSV file
data = pd.read_csv("hyperparameters.csv")

# Preprocess the data
X = data[['S1', 'S2', 'S3', 'K1', 'K2', 'K3']]
y = data['FLOPs'] + data['Storage']  # Assuming you want to optimize FLOPs + Storage

learning_rate = 0.01
num_iterations = 500 

# parameter initialization, to be random
params = np.array([random.randint(1,4), random.randint(1,4), random.randint(1,4), random.randint(1,14), random.randint(1,14), random.randint(1,14)])

print("Initial parameters:", params)

# Perform gradient descent
for i in range(num_iterations):
    # Calculate the gradients
    gradients = calculate_gradients(params, df)
    data_gradients = calculate_data_gradients(params, X, y)
    
    gradients += data_gradients
    
    # Normalize the gradients
    gradient_norm = np.linalg.norm(gradients)
    if gradient_norm > 0:
        gradients = gradients / gradient_norm
    
    # Update the parameters
    params = params - learning_rate * gradients
    
    # Evaluate the updated parameters
    storage_cost = get_storage_conv2d(df, [params[0], params[1], params[2]], [params[3], params[4], params[5]])
    arithmetic_ops = get_flops_conv2d(df, [params[0], params[1], params[2]], [params[3], params[4], params[5]])
    print(f"Iteration {i+1}: Storage Cost = {storage_cost}, Arithmetic Ops = {arithmetic_ops}")
    
# Retrieve the optimized parameters
optimized_params = params
print("Optimized parameters:", optimized_params)

# round the final parameters to the nearest integer
optimized_params = np.round(optimized_params)
print("Optimized parameters (rounded):", optimized_params) 

optimized_params[0] = np.clip(optimized_params[0], 0, 4)
optimized_params[1] = np.clip(optimized_params[1], 0, 4)
optimized_params[2] = np.clip(optimized_params[2], 0, 4)
optimized_params[3] = np.clip(optimized_params[3], 1, 14)
optimized_params[4] = np.clip(optimized_params[4], 1, 14)
optimized_params[5] = np.clip(optimized_params[5], 1, 14)

optimized_params = np.round(optimized_params)
print("Optimized parameters (rounded and projected):", optimized_params) 
