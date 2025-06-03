import reparam as rep
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1) 
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.conv1x1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn3(self.conv3x3(x)) + self.bn1(self.conv1x1(x)) + self.bn(x))
        return x
        

class RepVGG(nn.Module):
    def __init__(self, layer_list, num_channels, num_classes):
        super(RepVGG, self).__init__()
        
        self.base = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=3, stride=1), 
                             nn.BatchNorm2d(num_channels),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=num_channels, out_channels=2*num_channels, kernel_size=3, stride=1), 
                             nn.BatchNorm2d(2*num_channels),
                             nn.ReLU() 
                            )

        self.hidden_layers = self._make_layer(layer_list, 2*num_channels)

        last_num_channels = num_channels*2**(len(layer_list)+1)
        self.last_conv = nn.Sequential(nn.Conv2d(in_channels=last_num_channels, out_channels=last_num_channels, kernel_size=1, stride=1), 
                             nn.BatchNorm2d(num_channels*2**(len(layer_list)+1)),
                             nn.ReLU()
                             )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1) # Global average poolong
        self.output_layers = nn.Linear(last_num_channels, num_classes)
 
    def _make_layer(self, layer_list, num_channels):
        layers = []
        for num_layers in layer_list:
            for j in range(num_layers):
                block = ResBlock(num_channels)
                layers.append(block)
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=2*num_channels, kernel_size=3, padding=1, stride=2))
            num_channels = 2*num_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x)  
        x = self.hidden_layers(x)
        x = self.last_conv(x)
        x = self.gap(x).view(x.size(0),-1)
        x = self.output_layers(x)
        return x


def repvgg(layer_list=None, num_channels=16, num_classes=10, mode=None, param=None):   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RepVGG(layer_list=layer_list, num_channels=num_channels, num_classes=num_classes)

    if param is not None:           
        model.load_state_dict(torch.load(param, map_location=torch.device(device)))
    else:
        print("Initial random parameters")

    if mode == 'inference':           
        model.eval()
        model = rep.inference(model, layer_list)

    return model.to(device)