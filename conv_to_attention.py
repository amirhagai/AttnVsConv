import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import copy
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


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
        
        patch_height_out = int((im_height + 2 * conv_layer.padding[0] - patch_height_in) / conv_layer.stride[0]) + 1
        patch_width_out = int((im_width + 2 * conv_layer.padding[1] - patch_width_in) / conv_layer.stride[1]) + 1
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
    max_arr = []
    min_arr = []
    mean_arr = []
    for _ in range(200):

        im = torch.rand(2, c_in_at_conv, w, h)

        out = att(im)
        con = conv_layer(im)

        div = out / con

        max_arr.append(div.abs().max().item())
        min_arr.append(div.min().item())
        mean_arr.append(div.mean().item())
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






def get_grads(model, image, target, loss_fn, layer_name=""):
    """
    Computes the gradient of the specified layer with respect to the loss.

    Args:
    - model: The neural network model.
    - image: The input image tensor.
    - target: The target labels tensor.
    - loss_fn: The loss function to use.
    - layer_name: The name of the layer to compute the gradient for.

    Returns:
    - grad: The gradient of the layer with respect to the loss.
    """
    model.zero_grad()
    image.requires_grad = True
    output = model(image)
    loss = loss_fn(output, target)
    loss.backward()

    grads = {}
    for name, module in model.named_modules():
        if layer_name and name != layer_name:
            continue
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
              grads[name] = module.weight.grad
        else:
            continue
            # break

    return grads

def get_nested_attr(obj, attr_path):
    """get the model and string as return by x = get_grads; x.keys() and returns the 
    relevant layer"""
    current = obj
    try:
        for part in attr_path.split('.'):
            if '[' in part and ']' in part:
                # Extract the base name and the index
                base, index = part[:-1].split('[')
                current = getattr(current, base)[int(index)]
            else:
                current = getattr(current, part)
    except AttributeError as e:
        return None, f"AttributeError: {str(e)}"
    return current, None


def inject_noise_to_layer_output(model, layer_name, noise_level):
    """
    Injects noise into a specified layer of the model.

    Args:
    - model: The neural network model.
    - layer_name: The name of the layer where noise should be injected.
    - noise_level: The standard deviation of the Gaussian noise to inject.

    Returns:
    - injected_model: The model with noise injected.
    - handle: The handle of the registered forward hook.
    """
    # Create a copy of the model to inject noise into
    injected_model = copy.deepcopy(model)

    # Locate the layer by name
    def find_layer_by_name(model, layer_name):
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f'Layer {layer_name} not found in the model.')

    layer_to_inject = find_layer_by_name(injected_model, layer_name)

    # Define the forward hook
    def forward_with_noise(module, input, output):
        noise = torch.randn_like(output) * noise_level
        return output + noise

    # Register the forward hook
    handle = layer_to_inject.register_forward_hook(forward_with_noise)

    return injected_model, handle

def inject_noise_to_weights(model, layer_name, noise_level):
    """
    Injects noise into the weights of a specific layer of the model and returns both the modified model and the original model.

    Args:
        model (torch.nn.Module): The model with which to work.
        layer_name (str): The name of the layer to inject noise into.
        noise_level (float): The standard deviation of the Gaussian noise to be injected.

    Returns:
        injected_model (torch.nn.Module): The model with noise injected into the weights.
        original_model (torch.nn.Module): The original model.
    """
    # Create a copy of the model to inject noise into
    injected_model = copy.deepcopy(model)

    # Locate the layer by name
    def find_layer_by_name(model, layer_name):
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f'Layer {layer_name} not found in the model.')

    # Get the layer to inject noise into
    layer_to_inject = find_layer_by_name(injected_model, layer_name)

    # Add noise to the chosen layer's weights
    with torch.no_grad():
        if hasattr(layer_to_inject, 'weight') and layer_to_inject.weight is not None:
            # print(layer_to_inject)
            noise = torch.normal(0, noise_level, size=layer_to_inject.weight.shape).to(layer_to_inject.weight.device)
            # print(torch.norm(noise))
            layer_to_inject.weight += noise
        else:
            raise ValueError(f'Layer {layer_name} does not have weights.')

    return injected_model, model


def grads_usage_example():
    model = getresnet()
    # Example usage
    image = torch.randn(1, 3, 224, 224)  # Replace with your preprocessed image
    target = torch.tensor([0])  # Replace with your target label
    loss_fn = F.cross_entropy
    # layer_name = 'layer1.0.conv1'

    model.train()
    grads = get_grads(model, image, target, loss_fn, "")
    convs = list(grads.keys())
    print(convs, end="\n\n\n")
    z = 0
    for c in convs:
        z += 1
        layer = get_nested_attr(model, c)
        layer_name = str(layer)
        print(layer)
    print(z)


def inject_noise_to_layer_output_usage_example():
    
    original_model = getresnet()
    # Example usage
    layer_name = 'layer1.0.conv1'
    noise_level = 0.9

    # Create a model with injected noise
    injected_model, handle = inject_noise_to_layer_output(original_model, layer_name, noise_level)

    # Assuming you have an image tensor
    image = torch.randn(1, 3, 224, 224)  # Replace with your preprocessed image

    # Set both models to evaluation mode
    original_model.eval()
    injected_model.eval()

    # Get prediction from the original model
    with torch.no_grad():
        original_output = original_model(image)

    # Get prediction from the model with injected noise
    with torch.no_grad():
        injected_output = injected_model(image)

    # Calculate the difference in predictions
    pred_diff = (injected_output - original_output).abs()
    print(torch.norm(pred_diff))

    # Clean up by removing the hook
    handle.remove()


def inject_noise_to_weights_usage_example():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    sample_image = T.ToPILImage()(torch.rand(3, 224, 224))
    image = preprocess(T.ToTensor()(sample_image).unsqueeze(0))

    # Inject noise into the 'layer4.0.conv1' layer's weights with a noise level of 0.1
    injected_model, original_model = inject_noise_to_weights(model, 'layer4.0.conv1', noise_level=0.1)

    # Get predictions from both models
    with torch.no_grad():
        original_output = original_model(image)
        injected_output = injected_model(image)

    # Calculate the difference in predictions
    pred_diff = (injected_output - original_output).abs()
    print(torch.norm(pred_diff))

    inp = original_model.layer4[0].conv1.weight.data - injected_model.layer4[0].conv1.weight.data
    print(torch.norm(inp))
    
    
def main():
    model = getresnet()
    model.eval()
    # test()
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

    # grads_usage_example()
    # inject_noise_to_layer_output_usage_example()    
    # inject_noise_to_weights_usage_example()

    main()