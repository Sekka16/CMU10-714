from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.images = self.images.reshape(self.images.shape[0], 28, 28, 1)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # 在parse_minst得到的self.images的维度是 num_of_images * 784
        # 为了与后续的Transformations以及RandomCrop相配合，需要reshape成`H*W*C`的形状
        img = self.images[index]
        img = self.apply_transforms(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION