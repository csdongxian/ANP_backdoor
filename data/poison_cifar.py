import os
import numpy as np
from copy import deepcopy
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from PIL import Image


def split_dataset(dataset, val_frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    :param val_frac: the fraction of validation set.
    :param perm: A predefined permutation for sampling. If perm is None, generate one.
    :return: A training set + a validation set
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_val = int(val_frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_val:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_val:]].tolist()

    # generate the test set
    val_set = deepcopy(dataset)
    val_set.data = val_set.data[perm[:nb_val]]
    val_set.targets = np.array(val_set.targets)[perm[:nb_val]].tolist()
    return train_set, val_set


def generate_trigger(trigger_type):
    if trigger_type == 'checkerboard_1corner':  # checkerboard at the right bottom corner
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8) + 122
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for h in trigger_region:
            for w in trigger_region:
                pattern[30 + h, 30 + w, 0] = trigger_value[h+1][w+1]
                mask[30 + h, 30 + w, 0] = 1
    elif trigger_type == 'checkerboard_4corner':  # checkerboard at four corners
        pattern = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        mask = np.zeros(shape=(32, 32, 1), dtype=np.uint8)
        trigger_value = [[0, 0, 255], [0, 255, 0], [255, 0, 255]]
        trigger_region = [-1, 0, 1]
        for center in [1, 30]:
            for h in trigger_region:
                for w in trigger_region:
                    pattern[center + h, 30 + w, 0] = trigger_value[h + 1][w + 1]
                    pattern[center + h, 1 + w, 0] = trigger_value[h + 1][- w - 2]
                    mask[center + h, 30 + w, 0] = 1
                    mask[center + h, 1 + w, 0] = 1
    elif trigger_type == 'gaussian_noise':
        pattern = np.array(Image.open('./data/cifar_gaussian_noise.png'))
        mask = np.ones(shape=(32, 32, 1), dtype=np.uint8)
    else:
        raise ValueError(
            'Please choose valid poison method: [checkerboard_1corner | checkerboard_4corner | gaussian_noise]')
    return pattern, mask


def add_trigger_cifar(data_set, trigger_type, poison_rate, poison_target, trigger_alpha=1.0):
    """
    A simple implementation for backdoor attacks which only supports Badnets and Blend.
    :param clean_set: The original clean data.
    :param poison_type: Please choose on from [checkerboard_1corner | checkerboard_4corner | gaussian_noise].
    :param poison_rate: The injection rate of backdoor attacks.
    :param poison_target: The target label for backdoor attacks.
    :param trigger_alpha: The transparency of the backdoor trigger.
    :return: A poisoned dataset, and a dict that contains the trigger information.
    """
    pattern, mask = generate_trigger(trigger_type=trigger_type)
    poison_cand = [i for i in range(len(data_set.targets)) if data_set.targets[i] != poison_target]
    poison_set = deepcopy(data_set)
    poison_num = int(poison_rate * len(poison_cand))
    choices = np.random.choice(poison_cand, poison_num, replace=False)

    for idx in choices:
        orig = poison_set.data[idx]
        poison_set.data[idx] = np.clip(
            (1 - mask) * orig + mask * ((1 - trigger_alpha) * orig + trigger_alpha * pattern), 0, 255
        ).astype(np.uint8)
        poison_set.targets[idx] = poison_target
    trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                    'trigger_alpha': trigger_alpha, 'poison_target': np.array([poison_target]),
                    'data_index': choices}
    return poison_set, trigger_info


def add_predefined_trigger_cifar(data_set, trigger_info, exclude_target=True):
    """
    Poisoning dataset using a predefined trigger.
    This can be easily extended to various attacks as long as they provide trigger information for every sample.
    :param data_set: The original clean dataset.
    :param trigger_info: The information for predefined trigger.
    :param exclude_target: Whether to exclude samples that belongs to the target label.
    :return: A poisoned dataset
    """
    if trigger_info is None:
        return data_set
    poison_set = deepcopy(data_set)

    pattern = trigger_info['trigger_pattern']
    mask = trigger_info['trigger_mask']
    alpha = trigger_info['trigger_alpha']
    poison_target = trigger_info['poison_target']
    poison_set.data = \
        ((1 - mask) * poison_set.data + mask * ((1 - alpha) * poison_set.data + alpha * pattern)).astype(np.uint8)
    if poison_target.size == 1:
        poison_target = np.repeat(poison_target, len(poison_set.targets), axis=0)
    poison_set.targets = poison_target

    if exclude_target:
        no_target_idx = (poison_target != data_set.targets)
        poison_set.data = poison_set.data[no_target_idx, :, :, :]
        poison_set.targets = list(poison_set.targets[no_target_idx])
    return poison_set


class CIFAR10CLB(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(CIFAR10CLB, self).__init__()
        if train:
            self.data = np.load(os.path.join(root, 'train_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'train_labels.npy')).astype(np.long)
        else:
            self.data = np.load(os.path.join(root, 'test_images.npy')).astype(np.uint8)
            self.targets = np.load(os.path.join(root, 'test_labels.npy')).astype(np.long)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    clean_set = CIFAR10(root='../../data')
    poison_set, _ = add_trigger_cifar(data_set=clean_set, trigger_type='checkerboard_1corner', poison_rate=1.0, poison_target=0)
    import matplotlib.pyplot as plt
    print(poison_set.__getitem__(0))
    x, y = poison_set.__getitem__(0)
    plt.imshow(x)
    plt.show()


