from PIL import Image
import numpy as np
from torchvision import transforms

# process the image in the same way as our training data
def process_image(image_path):
    image = Image.open(image_path)

    # define the image transform
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                        [0.229, 0.224, 0.225])])
    
    transformed_image = transform(image)
    array = np.array(transformed_image)
    
    return array
