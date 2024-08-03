import torch
import torchvision.models
from collections import namedtuple
import inspect
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LPIPS_5Layer(torch.nn.Module):
    def __init__(self, feature_extract_net = "vgg", use_dropout=True, requires_grad=False):
        super(LPIPS_5Layer, self).__init__()
        if feature_extract_net == "vgg":
            self.feature_net = VGG16_5Slice()
        else:
            raise NotImplementedError("Feature extraction network not implemented")

        self.scaling_layer = ScalingLayer()

        self.one_x_one_conv1 = OnexOneConv(64, use_dropout=use_dropout) # relu1_2
        self.one_x_one_conv2 = OnexOneConv(128, use_dropout=use_dropout) # relu2_2
        self.one_x_one_conv3 = OnexOneConv(256, use_dropout=use_dropout) # relu3_3
        self.one_x_one_conv4 = OnexOneConv(512, use_dropout=use_dropout) # relu4_3
        self.one_x_one_conv5 = OnexOneConv(512, use_dropout=use_dropout) # relu5_3
        self.one_x_one_convs = torch.nn.ModuleList([self.one_x_one_conv1, self.one_x_one_conv2, self.one_x_one_conv3, self.one_x_one_conv4, self.one_x_one_conv5])
        
        if not requires_grad:
            model_path = "./weights/v0.1/vgg16.pth" #change to the one suitable
            self.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, image1, image2, input_in_neg1_to_1=False):
        # Normalize images to 0-1
        if input_in_neg1_to_1:
            image1 = (image1 + 1) / 2
            image2 = (image2 + 1) / 2
        # Normalize images to imagenet mean and std
        normed_image1, normed_image2 = self.scaling_layer(image1), self.scaling_layer(image2)

        pretrained_net_out1, pretrained_net_out2 = self.feature_net(normed_image1), self.feature_net(normed_image2)
        
        # Compute LPIPS
        normalized_out0, normalized_out1, diffs, layer_final_difference = [], [], []
        ########################
        
        # Compute Square of Difference for each 5 layer output
        for idx in range(5):
            #integral part of LPIPS: Normalize the output on the channel layer 
            #this way it focus more on the image pattern rather than the intensity
            normalized_out0.append(torch.nn.functional.normalize(pretrained_net_out1[idx]), dim=1) #normalize on the channel
            normalized_out1.append(torch.nn.functional.normalize(pretrained_net_out2[idx]), dim=1) #normalize on the channel
            diffs.append((normalized_out0[idx] - normalized_out1[idx]) ** 2)
            
            #1x1 conv
            onexone_res = self.one_x_one_convs[idx](diffs[idx])

            layer_final_difference.append(onexone_res.mean(dim=[2, 3], keepdim=True))
        
        # Final LPIPS
        LPIPS_value = 0
        for idx in range(5):
            LPIPS_value += layer_final_difference[idx]
        return LPIPS_value

        



class VGG16_5Slice(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16_5Slice, self).__init__()
        # Load pretrained vgg model from torchvision
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        # Freeze vgg model
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, X):
        # Return output of vgg features
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out
    
class OnexOneConv(torch.nn.Module):
    def __init__(self, chn_in, use_dropout=False):
        super(OnexOneConv, self).__init__()
        chn_out = 1
        layers = [torch.nn.Dropout(p = 0.5)] if (use_dropout) else []
        layers += [torch.nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False), ]
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.model(x)
        return out
    
class ScalingLayer(torch.nn.module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        # Imagnet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.229, .224, .225])[None, :, None, None])
    
    def forward(self, inp):
        return (inp - self.shift) / self.scale