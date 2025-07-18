'''
Unet Architecture written in Pytorch 
May 2024 
'''
__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 


#initial commit 
#note to cite article tutorial where Rosnel found the code

#Replacing lines 1-14
import torch
import torchvision.transforms.functional
from torch import nn
from pathlib import Path
from utility.settings import DEVICE_COMPUTE_PLATFORM, LEARNING_RATE

#3x3 Convolution Layers
#"Each step in the contraction path and expansive path have two convolutional layers followed by ReLU activations."
#Replacing lines 17-21
    #inputs = Input(input_size)
    #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    #conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # First 3x3 convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second 3x3 convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)
    
class FinalConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 1x1 convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.Sigmoid()  

    def forward(self, x: torch.Tensor):
        x = self.conv(x)  
        return x
    
#down-sample 
#("each step in the contracting path down-samples the feautre map w 2x2 max pooling layer")
#Replacing lines 
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)
    
#Up-sample
#each step in the expansive path up-samples the feature map with a 2x2 up-convolution
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)
    
#Crop and Concatenate the Feature Map
#"At every step in the expansive path the corresponding feature map from the contracting path concatenated with the current feature map."
#Replacing lines 
class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        return x

#U-net
class UNet(nn.Module):
    def __init__(self, out_channels: int):

        super().__init__()

        self.dropout=nn.Dropout(0.5)
        down_conv_sizes = [(1, 64), (64, 128), (128, 256), (256, 512)]
        
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in down_conv_sizes])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024)
 
        upsample_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]        
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in upsample_sizes])
       

        
        up_conv_sizes = [(1024, 512), (512, 256), (256, 128), (128, 64)]
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in up_conv_sizes])
        
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        
        # Final 1x1 convolution layer to produce the output
        self.final_conv = (nn.Conv2d(64, out_channels, kernel_size=1)) #FinalConvolution(64, out_channels)

    def forward(self, x: torch.Tensor):
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        #^^there's a 20% chance that a random selection of neuorns will be forced to be 0 to reinforce the training of other neurons
       
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two 3x3 convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        # Two 3x3 convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x)
        x=self.dropout(x)
        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two 3x3 convolutional layers
            x = self.up_conv[i](x)

        # Final 1x1 convolution layer
        #replacing our lines 61-62
            #conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
            # model = Model(inputs = inputs, outputs = conv10)
        out = (self.final_conv(x))
        return out

# Function to load model
def load_model_weights(model_path:str|Path|UNet, device:str = DEVICE_COMPUTE_PLATFORM, num_class:int=1)->UNet:
    
    if isinstance(model_path, UNet):
        model = model_path
        return model
    
    else:
        model = UNet(out_channels=num_class)
        if isinstance(model_path, str|Path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except:
                Warning('Could not load provided state_dict. Model will be initialized with random weights.')
                
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        return model
    
def initialize_model_training(pretrained_weights:str|Path|UNet, num_class:int=1)->None:
    if pretrained_weights:
        unet_model = load_model_weights(pretrained_weights, DEVICE_COMPUTE_PLATFORM)
    else:
        unet_model = UNet(out_channels=num_class)

    print(f"Compute platform is: {DEVICE_COMPUTE_PLATFORM}")
        
    unet_model.to(DEVICE_COMPUTE_PLATFORM)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
    return unet_model, optimizer