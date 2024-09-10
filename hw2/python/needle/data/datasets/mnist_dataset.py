from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import struct
import gzip

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as image, gzip.open(label_filename, "rb") as label:
      image_magic, image_number, image_rows, image_cols = struct.unpack(">IIII", image.read(16))
      image_pixels = struct.unpack("B" * (image_number * image_rows * image_cols), image.read())
      image_pixels_array = np.array(image_pixels, dtype=np.float32).reshape(image_number, image_rows * image_cols) / 255.0
        
      label_magic, label_number = struct.unpack(">II", label.read(8))
      label_values = struct.unpack("B" * label_number, label.read())
      label_values_array = np.array(label_values, dtype=np.uint8)

    return image_pixels_array, label_values_array
    ### END YOUR CODE

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