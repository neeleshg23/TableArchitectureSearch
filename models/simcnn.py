import torch.nn as nn
import torch.nn.functional as F

class SimCNN(nn.Module):
    # Constructor
    def __init__(self, num_classes=10):
        super(SimCNN, self).__init__()
        
        # Our images are RGB, so input channels = 3. We'll apply 12 filters in the first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=2)
        
        # We'll apply max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # A second convolutional layer takes 12 input channels, and generates 16 outputs
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
        
        # A third convolutional layer takes 16 input channels, and generates 16 outputs
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # A drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # Our 32x32 image tensors will be pooled twice with a kernel size of 2. 32/2/2 is 8.
        # So our feature tensors are now 8 x 8, and we've generated 16 of them
        # We need to flatten these and feed them to a fully-connected layer
        # to map them to the probability for each class
        self.fc = nn.Linear(in_features=8 * 8 * 16, out_features=num_classes)

    def forward(self, x):
        intermediate = []
        # Use a relu activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x)))
        intermediate.append(x.detach().cpu().numpy())
        # Use a relu activation function after layer 2 (convolution 2 and pool)
        x = F.relu(self.pool(self.conv2(x)))
        intermediate.append(x.detach().cpu().numpy()) 
        
        # Use a relu activation function after layer 3 (convolution 3)
        x = F.relu(self.conv3(x))
        intermediate.append(x.detach().cpu().numpy())
        # Select some features to drop after the 3rd convolution to prevent overfitting
        x = F.relu(self.drop(x))
        
        # Only drop the features if this is a training pass
        x = F.dropout(x, training=self.training)
        
        # Flatten
        x = x.view(-1, 8 * 8 * 16)
        # Feed to fully-connected layer to predict class
        x = self.fc(x)
        intermediate.append(x.detach().cpu().numpy())
        # Return log_softmax tensor 
        return F.log_softmax(x, dim=1), intermediate
