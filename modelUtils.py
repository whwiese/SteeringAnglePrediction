import torch
import torch.nn as nn

"""
CREATE CNN ARCHITECTURE FORMAT: 
(K,C,S,P) - ConvNet with Kernel Size=K, Output Filters=C, Stride=S, Padding=P
"M" - 2x2 Max Pooling Layer with Stride = 2
[(K,C,S,P),(K,C,S,P),N] - Tuples signify CovNets with same format as above,
N signifies number of times to repeat sequence of conv layers
"""

def create_cnn(architecture, in_channels=3, norm=None):
    layers = []

    for layer in architecture:
        if type(layer) == tuple:
            layers += [CNNBlock(in_channels,layer[1],layer[0],layer[2],layer[3], norm=norm)]
            in_channels = layer[1]
        elif type(layer) == str:
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        elif type(layer) == list:
            for _ in range(layer[-1]):
                for conv in layer[:-1]:
                    layers += [CNNBlock(in_channels,conv[1],conv[0],conv[2],conv[3], norm=norm)]
                    in_channels = conv[1]

    return nn.Sequential(*layers)

"""
creates a seuqential network of fully connected layers
with optional dropout and batchnorm

"""
def create_fcs(input_size, output_sizes, dropout=0.0, batch_norm=False):   
    """
    creates a seuqential network of fully connected layers
    with optional dropout and batchnorm

    NOTE: This will apply ReLU (and batch norm and dropout if enabled)
    after every layer including the last. The final fully connected
    layer in a network should be created separately.

    inputs:
        input_size (int): input size of first fc layer
        output_sizes (list of ints): output sizes of each layer
        dropout (float): Dropout parameter. 0 turns dropout off.
        batch_norm (boolean): set tp true to get batch norm layers
            after each fully connected layer except the last one.
    """
    fcs =[]

    for i, output_size in enumerate(output_sizes):
        
        if batch_norm:
            fcs += [nn.Linear(input_size,output_size,bias=False)]
            fcs += [nn.BatchNorm1d(output_size)]
        else:
            fcs += [nn.Linear(input_size,output_size)]
        
        fcs += [nn.ReLU()]

        if dropout:
            fcs += [nn.Dropout(dropout)]

        input_size = output_size

    return nn.Sequential(*fcs)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1, padding=0, norm=None):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size,  bias=False,
                stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.norm = norm

    def forward(self,x):
        x = self.conv(x)
        if self.norm == "batch":
            x = self.batchnorm(x)
        x = self.ReLU(x)
        return x


def cnn_output_size(input_size, architecture):
    """
    calculates the output size of a cnn
    created with create_cnn based on its
    architecture

    inputs:
        input_size (tuple): input_channels, input_x, input_y
        architecture (list): see create_cnn for format details
    returns:
        output_size(tuple): output_channels, output_x, output_y
    """
    onput_channels, output_x, output_y = input_size

    for layer in architecture:
        if type(layer) == tuple:
            output_x = (output_x-layer[0]+2*layer[3])//layer[2]+1
            output_y = (output_y-layer[0]+2*layer[3])//layer[2]+1
            output_channels = layer[1]
        elif type(layer) == str:
            output_x = output_x//2
            output_y = output_y//2
        elif type(layer) == list:
            for _ in range(layer[-1]):
                for conv in layer[:-1]:
                    output_x = (output_x-conv[0]+2*conv[3])//conv[2]+1
                    output_y = (output_y-conv[0]+2*conv[3])//conv[2]+1
                    output_channels = conv[1]

    return (output_channels, output_x, output_y)

