# A Project on Sparsity Visualize and Analysis
> Zhifan Ye @USTC

## Python Script

for simple pruning and fusion, please use command:
```
python pruned_validate.py [dataset path] --model [model name] -b [batch size] --fuse [bool] --pruning [pruning type] --pruning-arg [args of pruning method]
```

## Toolset

in folder [sparsity_util](sparsity_util/)

1. [draw.py](sparsity_util/draw.py) for drawing 2D & 3D Sparsity Map

2. [gen_sparsit_info.py](sparsity_util/gen_sparsity_info.py) to store and load sparsity

3. [sparsity_info.py](sparsity_util/sparsity_info.py) defines a class to operate on sparsity info

in folder [pruning_util]

+ helper.py: support for L1 Structed and Unstructed Pruning, can be tuned to support more complxed tuning method

## Testcase

in folder [timm/models](timm/models/), 5 testcases are available

+ deit_small_patch16_224
+ levit_128
+ mlp_mixer_b16_224
+ mobilenetv3_large_100
+ resnet18