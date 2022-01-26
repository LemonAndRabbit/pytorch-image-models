import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

def pruning_handler(model: torch.nn.Module, pruning_type, pruning_arg, name=''):
    # TODO: more custimized pruning dependant on and more pruning type
    model = copy.copy(model)
    if pruning_type == 'l1_unstructed':
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=pruning_arg)
            # if isinstance(module, torch.nn.Linear):
            #     prune.l1_unstructured(module, name='weight', amount=pruning_arg)
    elif pruning_type == 'l1_structed':
        for name, module in model.named_modules():
            # and ('layer2' in name or 'layer3' in name or 'layer4' in name) and 
            if isinstance(module, torch.nn.Conv2d) and name=='conv1':
                print('Pruning triggerd: '+ name)
                prune.ln_structured(module, name='weight', amount=pruning_arg, dim=0, n=1)
            # if isinstance(module, torch.nn.Linear):
            #     prune.ln_structured(module, name='weight', amount=pruning_arg, dim=0, n=1)
    else:
        raise NotImplementedError('pruning_type='+pruning_type+'not supported yet')
    return model

def prune_solidator(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and name=='conv1':
            prune.remove(module, name='weight')
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, name='weight')
    
def prune_protector(model: torch.nn.Module):
    def get_protector_hook(weight_mask):
        def protector_hook(m, input):
            m.weight = nn.parameter.Parameter(m.weight*weight_mask)

        return protector_hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for k, hook in module._forward_pre_hooks.items():
                if hook.__name__ == 'protector_hook':
                    del module._forward_pre_hooks[k]
            print('protected pruning: ' + name)
            module.register_forward_pre_hook(get_protector_hook(module.weight_mask))

def remove_prune_protector(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for k, hook in module._forward_pre_hooks.items():
                if hook.__name__ == 'protector_hook':
                    hook(module, 0)
                    del module._forward_pre_hooks[k]