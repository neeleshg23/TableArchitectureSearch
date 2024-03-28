import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.dlpack as dlpack
import cupy as cp
import numpy as np

from .amm.pq_amm_cnn import PQ_AMM_CNN
from .amm.vq_amm import PQMatmul
from .amm.kmeans import get_gpu

def im2col(input_data, kernel_size, stride, pad):
    cp.cuda.Device(get_gpu()).use()
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - kernel_size) // stride + 1
    out_w = (W + 2*pad - kernel_size) // stride + 1
    
    
    img = cp.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    img = cp.asarray(img)
    # img = img.to_gpu(3)
    col = cp.zeros((N, C, kernel_size, kernel_size, out_h, out_w))
    
    for y in range(kernel_size):
        y_max = y + stride*out_h
        for x in range(kernel_size):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

class SimCNN_AMM:
    num_layers = 4
    def __init__(self, state_dict, n_list, k_list):
        self.n = n_list
        self.k = k_list
        
        # Extract weights and biases from the state_dict 
        self.conv1_weight = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv1.weight']))
        self.conv1_bias = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv1.bias']))
        self.conv2_weight = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv2.weight']))
        self.conv2_bias = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv2.bias']))
        self.conv3_weight = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv3.weight']))
        self.conv3_bias = cp.fromDlpack(dlpack.to_dlpack(state_dict['conv3.bias']))
        self.fc_weight = cp.fromDlpack(dlpack.to_dlpack(state_dict['fc.weight']))
        self.fc_bias = cp.fromDlpack(dlpack.to_dlpack(state_dict['fc.bias'])) 
        
        self.amm_estimators = []*self.num_layers
    
    def conv2d(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        out = cp.dot(col, col_W) + b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
    
    def conv2d_amm(self, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)
        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])
        
        est = PQ_AMM_CNN(self.n.pop(0), self.k.pop(0))  
        est.fit(col_matrix_2d, col_W)

        est.reset_for_new_task()
        est.set_B(col_W)
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out, est
    
    def conv2d_eval(self, est, x, W, b, stride=1, pad=0):
        FN, C, FH, FW = W.shape
        N, C, H, Wid = x.shape
        out_h = int(1 + (H + 2*pad - FH) / stride)
        out_w = int(1 + (Wid + 2*pad - FW) / stride)

        col = im2col(x, FH, stride, pad)
        col_W = W.reshape(FN, -1).T

        col_matrix_2d = col.reshape(-1, col.shape[-1])
        
        est.reset_enc()
        conv_result = est.predict_cnn(col_matrix_2d, col_W)
        
        output = conv_result.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        out = output + b.reshape(1, -1, 1, 1)
        return out
    
    def relu(self, x):
        return cp.maximum(0, x)
    
    def linear_amm(self, input_data, weights, bias):
        input_data = torch.from_dlpack(input_data.toDlpack()).float()
        est = PQMatmul(self.n.pop(0), self.k.pop(0))
        est.fit(input_data, weights)
        est.reset_for_new_task()
        est.set_B(weights)
        res = est.predict(input_data, weights) + bias
        return res, est
    
    def linear_eval(self, est, input_data, weights, bias):
        est.reset_enc()
        weights, bias = cp.asarray(weights), cp.asarray(bias) 
        res = est.predict(input_data, weights) + bias
        return res
    
    def dropout(self, input, p):
        binary_value = cp.random.rand(*input.shape) > p
        res = input * binary_value
        res /= p
        return res
    
    
    # All forward methods expect torch.Tensor in and return torch.Tensor out
    
    def forward_test(self, x):
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.conv2d(x, self.conv1_weight, self.conv1_bias, stride=1, pad=2)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        x = self.conv2d(x, self.conv2_weight, self.conv2_bias, stride=1, pad=2)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        x = self.conv2d(x, self.conv3_weight, self.conv3_bias, stride=1, pad=1)
        x = self.relu(x)
        x = x.reshape(-1, 8 * 8 * 16)
        x = cp.dot(x, self.fc_weight.T) + self.fc_bias
        x = torch.from_dlpack(x.toDlpack()).float() 
        x = F.log_softmax(x, dim=1)
        return x
    
    def forward_train(self, x):
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x, est = self.conv2d_amm(x, self.conv1_weight, self.conv1_bias, stride=1, pad=2)
        self.amm_estimators.append(est)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        x, est = self.conv2d_amm(x, self.conv2_weight, self.conv2_bias, stride=1, pad=2)
        self.amm_estimators.append(est)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        x, est = self.conv2d_amm(x, self.conv3_weight, self.conv3_bias, stride=1, pad=1)
        self.amm_estimators.append(est)
        x = self.relu(x)
        # ADD DROP OUT HERE TO PREVENT OVERFITTING
        x = self.dropout(x, 0.2)
        x = x.reshape(-1, 8 * 8 * 16)
        x, est = self.linear_amm(x, self.fc_weight.T, self.fc_bias)
        self.amm_estimators.append(est)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.log_softmax(x, dim=1)
        return x
   
    def forward_eval(self, x):
        intermediate = []
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.conv2d_eval(self.amm_estimators.pop(0), x, self.conv1_weight, self.conv1_bias, stride=1, pad=2)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        intermediate.append(cp.asnumpy(x))
        x = self.conv2d_eval(self.amm_estimators.pop(0), x, self.conv2_weight, self.conv2_bias, stride=1, pad=2)
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.max_pool2d(x, 2)
        x = cp.from_dlpack(dlpack.to_dlpack(x))
        x = self.relu(x)
        intermediate.append(cp.asnumpy(x))
        x = self.conv2d_eval(self.amm_estimators.pop(0), x, self.conv3_weight, self.conv3_bias, stride=1, pad=1)
        x = self.relu(x)
        intermediate.append(cp.asnumpy(x))
        x = x.reshape(-1, 8 * 8 * 16)
        x = self.linear_eval(self.amm_estimators.pop(0), x, self.fc_weight.T, self.fc_bias)
        intermediate.append(cp.asnumpy(x))
        x = torch.from_dlpack(x.toDlpack()).float()
        x = F.log_softmax(x, dim=1)
        return x, intermediate 