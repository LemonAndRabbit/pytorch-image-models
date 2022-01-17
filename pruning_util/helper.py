import torch
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
    
    
