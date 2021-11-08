# Recipe

![recipe](https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/285/bento-box_1f371.png)

## Step 1. Train a backdoored model

- To train a BadNets [1] ResNet-18 ([checkpoint](https://drive.google.com/drive/folders/1SXcaYhw3CNbNCLAKig9GfwI4TF_vPFIV?usp=sharing) used in our paper), run,
```
python train_backdoor_cifar.py --poison-type badnets --poison-rate 0.05 --poison-target 0 --output-dir './save'
```


- To train a Blend [2] ResNet-18 ([checkpoint](https://drive.google.com/drive/folders/1SXcaYhw3CNbNCLAKig9GfwI4TF_vPFIV?usp=sharing)), run,
```
python train_backdoor_cifar.py --poison-type blend --poison-rate 0.05 --trigger-alpha 0.2 --poison-target 0 --output-dir './save'
```

- To train a Clean-label Backdoor (CLB) [3] ResNet-18 ([checkpoint](https://drive.google.com/drive/folders/1SXcaYhw3CNbNCLAKig9GfwI4TF_vPFIV?usp=sharing)), we first leverage adversarial perturbations to generate the poisoned training set using [open-source code](https://github.com/MadryLab/label-consistent-backdoor-code) following Turner et al.[3]. Specifically, 80% samples from the target class are poisoned with a "checkerboard" trigger at the four corners and Linf eps=16/255 (The [training set]() used in our paper). Then, save the dataset in ```./clb-data``` and run,

```
python train_backdoor_cifar.py --poison-type clean-label --clb-dir './clb-data' --poison-target 0 --output-dir './save'
```

- To train and repair a Input-aware Backdoor [4] ResNet-18, refer to our code [here]().

## Step 2. Optimize the masks of all neurons

- We optimize the masks on 1% of CIFAR-10 training data (0.01). We set eps=0.4 and alpha=0.2 in ANP by default. Then run,

```
python optimize_mask_cifar.py --val-frac 0.01 --anp-eps 0.4 --anp-alpha 0.2 --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th' --output-dir './save'
```


## Step 3. Prune neurons

- Neurons are pruned based on their mask values. We stop pruning until reaching a predefined threshold, 

```
python prune_neuron_cifar.py --pruning-by threshold --pruning-step 0.05 --pruning-max 0.95 --mask-file './save/mask_values.txt' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th' --output-dir './save'
```

- or a predefined number,

```
python prune_neuron_cifar.py --pruning-by number --pruning-step 20 --pruning-max 1000 --mask-file './save/mask_values.txt' --checkpoints './save/last_model.th' --trigger-info' './save/trigger_info.th' --output-dir './save'
```

## Step 4. Visualizattion

If steps above are conducted correctly, we can visualize the results. The expected visualization is as follows,

<img src="https://github.com/csdongxian/ANP_backdoor/blob/main/_plot/ANP_recipe.png" width="60%" height="60%">

## Reference

[1] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. BadNets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access, 2019.

[2] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, Dawn Song. Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. arXiv Preprint arXiv:1712.05526, 2017.  

[3] Alexander Turner, Dimitris Tsipras, and Aleksander Madry. Label-Consistent Backdoor Attacks. arXiv Preprint arXiv:1912.02771, 2019.

[4] Tuan Anh Nguyen and Anh Tran. Input-Aware Dynamic Backdoor Attack. In NeurIPS, 2020.