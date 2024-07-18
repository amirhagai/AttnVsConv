import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F


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

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias = False)

        d_k = dim
        orthogonal_matrix = torch.linalg.qr(torch.randn(d_k, d_k)).Q  # QR decomposition for orthonormal matrix
        orthogonal_matrix = orthogonal_matrix.repeat(heads, 1)  # Repeat for each head
        self.to_q.weight.data = orthogonal_matrix
        self.to_k.weight.data = orthogonal_matrix

        self.to_v.weight = torch.nn.Parameter(conv_layer.weight.reshape(conv_layer.out_channels, -1).clone().detach())

        self.to_out.weight = torch.nn.Parameter(torch.eye(inner_dim))

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
        
        patch_height_out = int((im_height - patch_height_in) / conv_layer.stride[0]) + 1
        patch_width_out = int((im_width - patch_width_in) / conv_layer.stride[1]) + 1
        self.rearange_out = Rearrange("b (p1 p2) c-> b c (p1) (p2)",p1 = patch_height_out, p2 = patch_width_out, c=c_out)
        self.conv = conv_layer


    def forward(self, x, weights=False):

        x = self.rearange_input(x)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)

        if weights:
          attn_weights = self.attend(dots)
          return self.to_out(out), attn_weights
        out = self.to_out(out) # out == out_conv[:, :, :, 0].permute(0, 2, 1)
        return self.rearange_out(out)



def test():
    w = 256
    h = 256
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

    for _ in range(100):
        
        im = torch.rand(2, c_in_at_conv, w, h)

        out = att(im)       
        con = conv_layer(im)
        
        div = out / con
        print(div.abs().max(), div.min(), div.mean())









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

def set_layer_by_name(obj, original_name, new_layer):
    parts = original_name.split('.')
    for i in range(len(parts)):
        if '[' in parts[i] and ']' in parts[i]:
            name, index = parts[i][:-1].split('[')
            if i == len(parts) - 1:
                getattr(obj, name)[int(index)] = new_layer
            else:
                obj = getattr(obj, name)[int(index)]
        else:
            if i == len(parts) - 1:
                setattr(obj, parts[i], new_layer)
            else:
                obj = getattr(obj, parts[i])





def get_all_convs_from_model(model):
  handles = []
  def add_hooks(model, parent_name="", layer_info=None):
      if layer_info is None:
          layer_info = {}

      def forward_hook(module, input, output, name=""):
          layer_info[name] = {"input": input[0].shape, "output": output.shape, "layer" : module, "w" : input[0].shape[2], "h" : input[0].shape[3]}

      for name, layer in model.named_children():
          # Create a path that corresponds to how you would access the attribute in the model
          if parent_name:
              path = f"{parent_name}.{name}"
          else:
              path = name

          # Register hooks only on Conv layers
          if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
              # Here we pass 'path' to the forward hook so it knows under what key to store the shape info
              handle = layer.register_forward_hook(lambda module, input, output, path=path: forward_hook(module, input, output, name=path))
              handles.append(handle)
          # Recurse into child layers
          add_hooks(layer, path, layer_info)

      return layer_info

  layer_info = add_hooks(model)
  dummy_input = torch.randn(2, 3, 224, 224)
  with torch.no_grad():
      _ = model(dummy_input)
  for handle in handles:
      handle.remove()
  return layer_info


def getresnet():
  model_cls = getattr(models, 'resnet50')
  model = model_cls(pretrained=True)
  return model


def replace_layer(model, layer_name_from_hook):
    layer_info = get_all_convs_from_model(model)
    set_layer_by_name(model,
                      layer_name_from_hook,
                      AttentionFromConv(im_height=layer_info[layer_name_from_hook]['h'],
                                        im_width=layer_info[layer_name_from_hook]['w'],
                                        conv_layer=layer_info[layer_name_from_hook]['layer']))
    return model

def main():
    model = getresnet()
    model.eval()
    test()
    # from PIL import Image
    
    # im = Image.open("/data/imagenet/1.jpeg")

    # List to store hook handles
    # handles = []
    # names = []

    # Add hooks
    # add_hooks(model, handles)
    
    replace_layer(model, 'layer4.0.conv3')

    # layer_info = get_all_convs_from_model(model)
    # set_layer_by_name(model, 'layer4.0.conv3', AttentionFromConv(im_height=layer_info['layer4.0.conv3']['h'], im_width=layer_info['layer4.0.conv3']['w'], conv_layer=layer_info['layer4.0.conv3']['layer']))
    # Dummy input for the model
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(dummy_input)


if __name__ == "__main__":
    main()