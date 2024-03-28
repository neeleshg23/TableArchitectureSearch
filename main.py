import torch
import torch.optim as optim
import cupy as cp 

from data_loader import get_data
from train import train, test
from metrics import layer_cossim 

from models.simcnn import SimCNN
from models_amm.simcnn_amm import SimCNN_AMM

from models_amm.amm.kmeans import set_gpu, get_gpu

def split(data_loader):
    all_data, all_targets = [], []
    for batch_idx, (data, target) in enumerate(data_loader):
        all_data.append(data)
        all_targets.append(target)
    all_data_tensor = torch.cat(all_data, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    return all_data_tensor, all_targets_tensor


def train_simcnn():
    root = "/data/neelesh/CV_Datasets"
    train_loader, val_loader, test_loader = get_data(root, 'c10')
    device = torch.device(f'cuda:{get_gpu()}' if torch.cuda.is_available() else 'cpu')
    model = SimCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epoch_nums = []
    training_loss = []
    validation_loss = []
    
    epochs = 50
    best_dev_loss = 1000
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        dev_loss = test(model, device, val_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(dev_loss)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), '0_results/simcnn_best.pth')
            
    
    print("Best Validation Loss: ", best_dev_loss)
    print("--Testing--")
    
    best_model = SimCNN().to(device)
    best_model.load_state_dict(torch.load('0_results/simcnn_best.pth')) 
    best_model.eval()
    test_loss = test(best_model, device, test_loader)
    print("Test Loss: ", test_loss)

n_train = 500 
n_test = 500

def main(n_values, k_values):
    set_gpu(1)
    root = "/data/neelesh/CV_Datasets"
    train_loader, val_loader, test_loader = get_data(root, 'c10')
    device = torch.device(f'cuda:{get_gpu()}' if torch.cuda.is_available() else 'cpu')
    model = SimCNN().to(device)
    model.load_state_dict(torch.load('0_results_networks/simcnn_best.pth')) 
    model.eval()
    
    train_data, train_target = split(train_loader)
    train_data, train_target = train_data[:n_train].to(device), train_target[:n_train].to(device)
    
    model_amm = SimCNN_AMM(model.state_dict(), n_values[:], k_values[:]) # send in shallow copies 
    
    out_amm = model_amm.forward_train(train_data) 
    
    # TEST AMM MODEL
    print("-- STARTING AMM MODEL TESTING --")
    
    test_data, test_target = split(test_loader)
    test_data, test_target = test_data[:n_test].to(device), test_target[:n_test].to(device)
    
    out, intermediate = model(test_data)
    out_amm, intermediate_amm = model_amm.forward_eval(test_data)
    
    layerwise_cosine_similarity = layer_cossim(intermediate, intermediate_amm) 

    print("Layerwise Cosine Similarity:", layerwise_cosine_similarity)
    
    # Get MSE between the final output of the two models
    mse = torch.nn.functional.mse_loss(out, out_amm)
    print("MSE between final output of the two models:", mse.item())
    
    # take a snapshot of results and save them to /1_results_tables/simcnn-n1-n2-n3-n4-k1-k2-k3-k4.txt
    path = '1_results_tables/simcnn-' + "-".join(map(str, n_values)) + "-" + "-".join(map(str, k_values)) + ".txt"
    with open(path, 'w') as f:
        f.write(f"Layerwise Cosine Similarity: {layerwise_cosine_similarity}\n")
        f.write(f"MSE between final output of the two models: {mse.item()}\n")

    return layerwise_cosine_similarity

# main([1, 1, 1, 1], [16, 16, 16, 16]) 