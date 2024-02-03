import torch.nn as nn

class SobelNet(nn.Module):
    """"
    Create the neural network for mimicking the output of the filter.
    Obtains parameters like number layers from config file.
    """

    def __init__(self, in_channel=1, num_layers=1):
        super(SobelNet, self).__init__()
        out_channels = 1
        self.relu = nn.ReLU()

        self.conv_layers = nn.ModuleList()
        if num_layers == 1:
            self.conv_layers.append(nn.Conv2d(in_channel, out_channels, kernel_size=3, padding=1))
        else:
            self.conv_layers.append(nn.Conv2d(in_channel, 32, kernel_size=3, padding=1))
            self.conv_layers.append(self.relu)
            
            for _ in range(num_layers - 2):
                self.conv_layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
                self.conv_layers.append(self.relu)
            
            self.conv_layers.append(nn.Conv2d(32, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x
    
