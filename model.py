import torch
import torch.nn as nn
from modelUtils import (create_cnn, create_cnn3D, CNNBlock,
                        create_fcs, cnn_output_size,
                        cnn3D_output_size)

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

class CNNDriver3D(nn.Module):
    def __init__(self, input_dims = (3,5,66,200), norm="batch",
            lookback=5):
        super().__init__()
        self.input_dims = input_dims

        cnn_architecture = [
            ((1,5,5),24,(1,2,2),0),
            ((1,5,5),36,(1,2,2),0),
            ((1,5,5),48,(1,2,2),0),
            ((1,3,3),64,(1,1,1),0),
            ((1,3,3),64,(1,1,1),0),
        ]

        self.cnn = create_cnn3D(cnn_architecture, self.input_dims[0],
                norm=norm,
        )
        cnn_out = cnn3D_output_size(self.input_dims,cnn_architecture)
        out_size = lookback*cnn_out[0]*cnn_out[1]*cnn_out[2]
        self.fcs = create_fcs(out_size,[1164,100,50,10])
        self.output_layer = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        x = self.output_layer(x).squeeze()
        return x

class DeeperCNNDriver3D(nn.Module):
    def __init__(self, input_dims = (3,5,455,256), norm="batch",
            lookback=5):
        super().__init__()
        self.input_dims = input_dims

        cnn_architecture = [
            ((1,5,5),24,(1,1,1),(0,2,2)),
            ((1,5,5),24,(1,2,2),0),
            ((1,5,5),36,(1,1,1),(0,2,2)),
            ((1,5,5),36,(1,2,2),0),
            ((1,5,5),48,(1,1,1),(0,2,2)),
            ((1,5,5),48,(1,2,2),0),
            ((1,3,3),64,(1,1,1),(0,1,1)),
            ((1,3,3),64,(1,1,1),0),
            ((1,3,3),64,(1,1,1),0),
        ]

        self.cnn = create_cnn3D(cnn_architecture, self.input_dims[0],
                norm=norm)
        cnn_out = cnn3D_output_size(self.input_dims,cnn_architecture)
        out_size = lookback*cnn_out[0]*cnn_out[1]*cnn_out[2]
        self.fcs = create_fcs(out_size,[1164,100,50,10])
        self.output_layer = nn.Linear(10,1)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fcs(x)
        x = self.output_layer(x).squeeze()
        return x

#########################################################################
#                           LSTM MODELS                                 #
#########################################################################

class LSTMDriver(nn.Module):
    def __init__(self, input_dims=(3,455,356), norm="batch",
            hidden_size=256):
        super(LSTMDriver, self).__init__()

        cnn_architecture = [
            (5,24,2,0),
            (5,36,2,0),
            (5,48,2,0),
            (3,64,1,0),
            (3,64,1,0),
        ]

        self.cnn = create_cnn(cnn_architecture, input_dims[0],
                norm=norm,
        )
        cnn_out = cnn_output_size(input_dims,cnn_architecture)
        out_size = cnn_out[0]*cnn_out[1]*cnn_out[2]

        self.fc1 = nn.Linear(out_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fcs = create_fcs(hidden_size,[hidden_size,100,50,10])
        self.output_layer = nn.Linear(10,1)

    def forward(self, x, h_prev, c_prev):
        batch_size = x.shape[0]

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.reshape((batch_size, -1))

        fc1_out = self.fc1(cnn_output).unsqueeze(0)
        
        out, (h_out, c_out) = self.lstm(fc1_out, (h_prev,c_prev))

        x = out.reshape(batch_size, -1)

        x = self.fcs(x)

        x = self.output_layer(x).squeeze()

        return x, h_out, c_out

class LSTMDriver3D(nn.Module):
    def __init__(self, input_dims=(3,5,455,356), norm="batch",
            lookback=5, hidden_size=256):
        super(LSTMDriver, self).__init__()
        cnn_architecture = [
            ((1,5,5),24,(1,2,2),0),
            ((1,5,5),36,(1,2,2),0),
            ((1,5,5),48,(1,2,2),0),
            ((1,3,3),64,(1,1,1),0),
            ((1,3,3),64,(1,1,1),0),
        ]
        self.input_dims=input_dims
        
        self.cnn = create_cnn3D(cnn_architecture, self.input_dims[0],
                norm=norm)
        cnn_out = cnn3D_output_size(self.input_dims,cnn_architecture)
        out_size = lookback*cnn_out[0]*cnn_out[1]*cnn_out[2]

        self.lstm = nn.LSTM(out_size, hidden_size, batch_first=True)

        self.fcs = create_fcs(hidden_size,[1164,100,50,10])
        self.output_layer = nn.Linear(10,1)

    def forward(self, x, h_prev, c_prev):
        batch_size = x.shape[0]
        print(batch_size)

        cnn_output = self.cnn(x)
        cnn_output = cnn_output.reshape((batch_size, -1)).unsqueeze(0)
        
        h_out, c_out = self.lstm(cnn_output, (h_prev,c_prev))

        x = h_out.reshape(batch_size, -1)
        print(x.shape)

        x = self.fcs(x)

        x = self.output_layer(x)

        return x, h_out, c_out




