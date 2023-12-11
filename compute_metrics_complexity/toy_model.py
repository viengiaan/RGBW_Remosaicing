import torch
import torch.nn as nn

# Toy HDR Model
class ToyHDRModel(nn.Module):

    def __init__(self):
        super(ToyHDRModel, self).__init__()
      
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16*3, 16, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, X):
        
        x1 = X[:,0,0:3,:,:]
        x2 = X[:,1,0:3,:,:]
        x3 = X[:,2,0:3,:,:]
        F1 = self.relu(self.conv1(x1))
        F2 = self.relu(self.conv1(x2))
        F3 = self.relu(self.conv1(x3))
        F_cat = torch.cat((F1, F2, F3), 1)
        F_mid = self.conv2(F_cat)
        F_out = self.conv3(F_mid)
        HDR_out = self.relu(F_out)
        return HDR_out
