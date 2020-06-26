from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)

 # downsize image res to 256, crop the center 224, and grayscale
PREPROCESS_TRANSFORM = transforms.Compose([
    # TODO: change this to `transforms.Resize((244,244))`. It is like this to conform with old checkpoint
    transforms.Resize((244,244)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# preprocess and rotate and flip images for more data
DATA_AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    # I would like to add random rotations, but they keep zeroing out the tensor...
    # transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# process the image in the same way as our training data
def process_image(image_path):
    image = Image.open(image_path)
    
    transformed_image = PREPROCESS_TRANSFORM(image)
    array = np.array(transformed_image)
    
    return array
