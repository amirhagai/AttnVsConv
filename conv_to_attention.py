import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import numpy as np
# import torchvision.transforms as T
# from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt
from PIL import Image

class AttentionFromConv(nn.Module):
    def __init__(self, im_height, im_width, conv_layer, dim_head = 1):
        super().__init__()
        
        dim=conv_layer.kernel_size[0] * conv_layer.kernel_size[1] * conv_layer.in_channels
        heads=conv_layer.out_channels
        c_in=conv_layer.in_channels
        c_out=conv_layer.out_channels
        patch_height_in=conv_layer.kernel_size[0]
        patch_width_in=conv_layer.kernel_size[1]

        
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)

        # self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # self.to_k = nn.Linear(dim, inner_dim, bias = False)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)        
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias = False)

        d_k = dim
        # orthogonal_matrix = torch.linalg.qr(torch.randn(d_k, d_k)).Q  # QR decomposition for orthonormal matrix
        # orthogonal_matrix = orthogonal_matrix.repeat(heads, 1)  # Repeat for each head
        # orthogonal_matrix = torch.eye(d_k)
        self.to_q.weight.data = torch.eye(d_k).to(conv_layer.weight.device)
        self.to_k.weight.data = torch.eye(d_k).to(conv_layer.weight.device)

        self.to_v.weight = torch.nn.Parameter(conv_layer.weight.reshape(conv_layer.out_channels, -1).clone().detach())

        self.to_out.weight = torch.nn.Parameter(torch.eye(inner_dim).to(conv_layer.weight.device))

        def unfold_function(in_channels, kernel_size_0, kernel_size_1, padding, stride):
            def unfold(image):
                unfolded = F.unfold(image, kernel_size=(kernel_size_0, kernel_size_1), dilation=1, padding=padding, stride=stride)
                patch_dim = in_channels * kernel_size_0 * kernel_size_1
                patches = unfolded.view(image.size(0), patch_dim, -1)
                patches = patches.permute(0, 2, 1)  # reshaping to have each patch as a row vector
                return patches
            return unfold
        
        self.rearange_input = unfold_function(conv_layer.in_channels,
                                              patch_height_in,
                                              patch_width_in,
                                              conv_layer.padding,
                                              conv_layer.stride
                                              )
        
        patch_height_out = int((im_height + 2 * conv_layer.padding[0] - patch_height_in) / conv_layer.stride[0]) + 1
        patch_width_out = int((im_width + 2 * conv_layer.padding[1] - patch_width_in) / conv_layer.stride[1]) + 1
        self.rearange_out = Rearrange("b (p1 p2) c-> b c (p1) (p2)",p1 = patch_height_out, p2 = patch_width_out, c=c_out)
        # self.conv = conv_layer


    def forward(self, x, weights=False, i=0):

        x = self.rearange_input(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        if weights:
          return self.rearange_out(self.to_out(out)), attn
        out = self.to_out(out) # out == out_conv[:, :, :, 0].permute(0, 2, 1)
        return self.rearange_out(out)



def test_conv_to_attn():
    w = 512
    h = 512
    stride = 16
    c_in_at_conv = 13
    channels_conv = 14
    patch_height_in_at_conv = 16
    patch_width_in_at_conv = 32

    conv_layer = torch.nn.Conv2d(in_channels=c_in_at_conv,
                                out_channels=channels_conv,
                                kernel_size=(patch_height_in_at_conv,
                                            patch_width_in_at_conv),
                                stride=stride, bias=False)

    att = AttentionFromConv(
                im_height=h,
                im_width=w,
                conv_layer=conv_layer,
            )
    max_arr = []
    min_arr = []
    mean_arr = []
    size = 5
    for i in range(200):
        im = torch.rand(size, c_in_at_conv, w, h)

        out, weights = att(im, weights=True)
        weights = weights[0].detach().numpy()[:, :, None]
        weights = np.repeat(weights, 3, axis=2)
        weights = (weights * 255).astype(np.uint8)
        Image.fromarray(weights).save(f'/code/attn_weights/attn_weights_{i}.png')
        con = conv_layer(im)

        div = out / con

        max_arr.append(div.abs().max().item())
        min_arr.append(div.min().item())
        mean_arr.append(div.mean().item())
        print(i)
        print(f"(div > 0.98).sum()  / div.shape - {(div > 0.98).sum() / div.numel()}")
        print(max_arr[-1], min_arr[-1], mean_arr[-1], end="\n\n\n")

    max_arr = np.array(max_arr)
    min_arr = np.array(min_arr)
    mean_arr = np.array(mean_arr)
    print(f"max - {np.mean(max_arr)}, {np.std(max_arr)}, min - {np.mean(min_arr)}, {np.std(min_arr)}, mean - {np.mean(mean_arr)}, {np.std(mean_arr)}")


def format_attr_path(attr_path):
    parts = attr_path.split('.')
    formatted_parts = []
    for part in parts:
        if '[' in part:
            name, index = part[:-1].split('[')
            formatted_parts.append(f"{name}[{index}]")
        elif part.isdigit():
            # This part is a numeric index following a dot, convert it
            formatted_parts[-1] = f"{formatted_parts[-1]}[{part}]"
        else:
            formatted_parts.append(part)
    return '.'.join(formatted_parts)


def get_nested_attr(obj, attr_path):
    current = obj
    parts = attr_path.split('.')
    for part in parts:
        if '[' in part and ']' in part:  # Checks for the presence of brackets indicating an index
            name, index = part[:-1].split('[')
            current = getattr(current, name)  # Access the attribute up to the bracket
            current = current[int(index)]  # Use the index inside the brackets
        else:
            current = getattr(current, part)  # Regular attribute access
    return current
    
    

def main():
    test_conv_to_attn()


if __name__ == "__main__":
    main()