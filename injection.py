import torch
import torch.nn as nn
import copy
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import torch.nn.functional as F



def getresnet():
  model_cls = getattr(models, 'resnet50')
  model = model_cls(pretrained=True)
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
    size = 5
    image = torch.randn(size, 3, 224, 224)  # Replace with your preprocessed image
    target = torch.tensor([0] * size)  # Replace with your target label
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
    
    
if __name__ == '__main__':
    inject_noise_to_weights_usage_example()
    inject_noise_to_layer_output_usage_example()
    grads_usage_example()
    