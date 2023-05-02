from kandinsky2.model.unet import Upsample
import torch
import torch.nn as nn


fp32_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, 
                nn.Softmax, nn.Softmax2d, nn.Softmin,
                nn.CrossEntropyLoss,
                nn.LayerNorm, nn.Embedding, nn.GroupNorm, Upsample)

def monkey_patch_bf16(module):
    if isinstance(module, fp32_modules):
        #print(f'Skipping {type(module)} because it is not compatible with bfloat16')
        module.forward = convert_to_fp32(module.forward)
        module = module.to(torch.float32)
    else:
        children = list(module.children())
        
        if len(children) > 0:
            for child in module.children():
                monkey_patch(child)
        else:
            module.forward = convert_to_bfloat16(module.forward)
            module = module.to(torch.bfloat16)
    return module


def convert_to_bfloat16(func):
    def wrapper(*args, **kwargs):
        args = [arg.to(torch.bfloat16) if isinstance(arg, torch.Tensor) and arg.dtype != torch.bfloat16 else arg for arg in args]
        kwargs = {key: (value.to(torch.bfloat16) if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16 else value) for key, value in kwargs.items()}
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and result.dtype == torch.float32:
            result = result.to(torch.bfloat16)
        return result
    return wrapper


def convert_to_fp32(func):
    def wrapper(*args, **kwargs):
        args = [arg.to(torch.float32) if isinstance(arg, torch.Tensor) and arg.is_floating_point() and arg.dtype != torch.float32 else arg for arg in args]
        kwargs = {key: (value.to(torch.float32) if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32 else value) for key, value in kwargs.items()}
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and result.is_floating_point() and result.dtype != torch.bfloat16:
            result = result.to(torch.float32)
        return result
    return wrapper