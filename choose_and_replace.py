import torch
import torchvision.models as models
from conv_to_attention import AttentionFromConv
import torch.nn as nn
from injection import (
    get_grads,
    inject_noise_to_layer_output,
    inject_noise_to_weights
    )


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




def get_all_convs_from_model(model, input_shape):
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
  dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
  with torch.no_grad():
      _ = model(dummy_input)
  for handle in handles:
      handle.remove()
  return layer_info


def getresnet():
  model_cls = getattr(models, 'resnet50')
  model = model_cls(pretrained=True)
  return model


def replace_layer(model, layer_name_from_hook, input_shape, scale=1):
    layer_info = get_all_convs_from_model(model, input_shape)
    set_layer_by_name(model,
                      layer_name_from_hook,
                      AttentionFromConv(im_height=layer_info[layer_name_from_hook]['h'],
                                        im_width=layer_info[layer_name_from_hook]['w'],
                                        conv_layer=layer_info[layer_name_from_hook]['layer'], 
                                        scale=scale))
    return model




def choose_and_replace(model, data_loader, critirion, input_shape):
    
    
    layer_info = get_all_convs_from_model(model, input_shape)
    names = layer_info.keys()
    
    if critirion == 'grads':
        grads_arr = []
        for x, y in data_loader:
            grads_values = get_grads(model, image=x, target=y)
            grads_arr.append(grads_values)
        return grads_arr
            
    elif critirion == 'inject noise to weights':
        diffs = {}
        for key in names:
            diffs[key] = 0
            injected_model, model = inject_noise_to_weights(model,
                                                            key,
                                                            noise_level=0.1
                                                            )
            
            for x, y in data_loader:
                diffs[key] = diffs[key] + (injected_model(x) - model(x)).abs()
        return diffs 
                
        
    elif critirion == 'inject noise to layer output':
        diffs = {}
        for key in names:
            diffs[key] = 0
            injected_model, handle = inject_noise_to_layer_output(model,
                                                                  key,
                                                                  noise_level=0.1
                                                                  )
            for x, y in data_loader:
                diffs[key] = diffs[key] + (injected_model(x) - model(x)).abs()
            handle.remove()
        return diffs
    
    else:
        raise 'no such critirion'
        


def main():

    model = getresnet()
    model.eval()
    
    print(f'before - {model.layer4[0].conv3}')
    replace_layer(model, 'layer4.0.conv3', input_shape=(1, 3, 224, 224))

    # layer_info = get_all_convs_from_model(model)
    # set_layer_by_name(model, 'layer4.0.conv3', AttentionFromConv(im_height=layer_info['layer4.0.conv3']['h'], im_width=layer_info['layer4.0.conv3']['w'], conv_layer=layer_info['layer4.0.conv3']['layer']))
    # Dummy input for the model
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model(dummy_input)
        
    print(f'after - {model.layer4[0].conv3}')




if __name__ == "__main__":

    main()