import torch
import torch.nn as nn
from modelUtils import (create_cnn, CNNBlock,
                        create_fcs, cnn_output_size)

class CNNDriver(nn.Module):
    def __init__(self, input_dims = (3,66,200), norm="batch"):
        super().__init__()
        self.input_dims = input_dims

        cnn_architecture = [
            (5,24,2,0),
            (5,36,2,0),
            (5,48,2,0),
            (3,64,1,0),
            (3,64,1,0),
        ]

        self.cnn = create_cnn(cnn_architecture, self.input_dims[0],
                norm=norm,
        )
        cnn_out = cnn_output_size(self.input_dims,cnn_architecture)
        out_size = cnn_out[0]*cnn_out[1]*cnn_out[2]
        self.fcs = create_fcs(out_size,[1164,100,50,10])
        self.output_layer = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        x = self.output_layer(x).squeeze()
        return x

class DeeperCNNDriver(nn.Module):
    def __init__(self, input_dims = (3,455,256), norm="batch"):
        super().__init__()
        self.input_dims = input_dims

        cnn_architecture = [
            (5,24,1,2),
            (5,24,2,0),
            (5,36,1,2),
            (5,36,2,0),
            (5,48,1,2),
            (5,48,2,0),
            (3,64,1,1),
            (3,64,1,0),
            (3,64,1,0),
        ]

        self.cnn = create_cnn(cnn_architecture, self.input_dims[0],
                norm=norm)
        cnn_out = cnn_output_size(self.input_dims,cnn_architecture)
        out_size = cnn_out[0]*cnn_out[1]*cnn_out[2]
        self.fcs = create_fcs(out_size,[1164,100,50,10])
        self.output_layer = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        x = self.output_layer(x).squeeze()
        return x

        


