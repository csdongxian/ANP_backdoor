# Adversarial Neuron Pruning Purifies Backdoored Deep Models

Backdoor Defense @ NeurIPS 2021 "[Adversarial Neuron Pruning Purifies Backdoored Deep Models](https://arxiv.org/abs/2110.14430)" by [Dongxian Wu](https://csdongxian.github.io/) and [Yisen Wang](https://yisenwang.github.io/). 

## News

11/08/2021 - Our checkpoints and recipe have been released.

10/31/2021 - Our code has be released.

10/28/2021 - Our [paper](https://arxiv.org/abs/2110.14430) and [slide](https://drive.google.com/file/d/112FRYd9OfaElQJU_VCOUtVja8z7msOgO/view?usp=sharing) have be released.

10/26/2021 - Our code and paper will be released soon.

## What ANP Does

ANP can easily repair backdoored deep models using limited clean data and limited computational resources. Only **500 clean images** from CIFAR-10 and **2000 iterations** are used in the displayed example.

<img src="https://github.com/csdongxian/ANP_backdoor/blob/main/_plot/ANP_overall.png" width="60%" height="60%">

## Requirements
This code is implemented in PyTorch, and we have tested the code under the following environment settings:

- python = 3.7.3
- torch = 1.8.0
- torchvision = 0.9.0 


## A Quick Start - How to use it

For a detailed introduction, please refer to our [recipe](https://github.com/csdongxian/ANP_backdoor/blob/main/recipe.md).

#### Step 1: Train a backdoored DNN
By default, we train a backdoored resnet-18 under badnets with 5% poison rate and class 0 as target label, 

```
python train_backdoor_cifar.py --output-dir './save'
```

We save trained backdoored model and the trigger info as `./save/last_model.th` and `./save/trigger_info.th`. Some checkpoints have been released in [Google drive](https://drive.google.com/drive/folders/1voFOKUyyprzvF3cLQf2N5OOCqsxmNLnn?usp=sharing) or [Baidu drive](https://pan.baidu.com/s/1oHIh5kmgq5iISaF1MRImUg) (pwd: bmrb).


#### Step 2: Optimize masks under neuron perturbations

We optimize the mask for each neuron under neuron perturbations, and save mask values in './save/mask_values.txt' . By default, we only use 500 clean data to optimize.

```
python optimize_mask_cifar.py --output-dir './save' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th'
```

#### Step 3: Prune neurons to defend

You can prune neurons by threshold,

```
python prune_neuron_cifar.py --output-dir './save' --mask-file './save/mask_values.txt' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th'
```

## Citing this work

If you use our code, please consider cite our work

```bibtex
@inproceedings{wu2021adversarial,
    title={Adversarial Neuron Pruning Purifies Backdoored Deep Models},
    author={Dongxian Wu and Yisen Wang},
    booktitle={NeurIPS},
    year={2021}
}
```

If there is any problem, be free to open an issue or contact: wudx16@gmail.com.

## Useful Links

[1] Mode Connectivity Repair (MCR) defense: https://github.com/IBM/model-sanitization/tree/master/backdoor

[2] Input-aware Backdoor (IAB) attack: https://github.com/VinAIResearch/input-aware-backdoor-attack-release
