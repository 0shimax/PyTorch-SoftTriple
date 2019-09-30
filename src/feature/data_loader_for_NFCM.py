import torchvision
import torch
from torch.utils.data import Dataset
import numpy as np
import random


# ラベル	クラス
# 0	T-シャツ/トップ (T-shirt/top)
# 1	ズボン (Trouser)
# 2	プルオーバー (Pullover)
# 3	ドレス (Dress)
# 4	コート (Coat)
# 5	サンダル (Sandal)
# 6	シャツ (Shirt)
# 7	スニーカー (Sneaker)
# 8	バッグ (Bag)
# 9	アンクルブーツ (Ankle boot)

near_cat_dict = {0:[0,1,5,6,7,9], 1:[0,1,2,5,6,7,9], 2:[1,2,5,7], 3:[3,4,8],
                 4:[1,4,6,8,9], 5:[0,1,5,6], 6:[1,5,6,7,9], 7:[0,1,6,7],
                 8:[3,4,8,9], 9:[1,4,6,8,9]}
# far_cat = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}


class FMNISTDataset(Dataset):
    def __init__(self, n_class=10, train=True):
        super().__init__()
        self.n_class = n_class
        self.train = train
        self.n_relation = self.n_class**2
        self.fashion_mnist_data = torchvision.datasets.FashionMNIST(
            './fashion-mnist',
            transform=torchvision.transforms.ToTensor(),
            train=train,
            download=True)
        self.labels = [fnmn[1] for fnmn in torchvision.datasets.FashionMNIST('./fashion-mnist')]
        self.labels = np.array(self.labels, dtype=np.int32)

    def __len__(self):
        return len(self.fashion_mnist_data)

    def __getitem__(self, idx):
        image, cat = self.fashion_mnist_data[idx]
        return image, cat


def loader(dataset, batch_size, shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)
    return loader
