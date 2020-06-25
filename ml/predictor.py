import numpy as np
import torch

from torch import nn
from torchvision import datasets, models
from ml.util import process

# class to load the model and process images
class Predictor:
    # function to load model
    def load_model(self, checkpoint_path, num_classes):
        # use resnet34 for transfer learning
        model = models.resnet34(pretrained=True)
        num_input_filters = model.fc.in_features

        # define the fully-connected layer to be a linear transformation from num_input_filters to num_classes
        model.fc = nn.Linear(num_input_filters, num_classes)

        # load weights and class mappings from checkpoint, and use cpu
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.class_to_idx = checkpoint['class_to_idx']
    
        return model

    def __init__(self, checkpoint_path, num_classes = 196):
        # load model and use cpu
        self._model = self.load_model(checkpoint_path, num_classes).cpu()

    '''
    function to predict the class of an image

    @params
    image_path = path to the image
    topk = how many of the top-predicted classes do you want returned

    @outputs
    pred_confs = np array of confidence values of predictions
    pred_classes = corresponding np aray of predicted class indexes
    '''
    def predict(self, image_path, topk=5):
        image = process.process_image(image_path)

        # convert image to float tensor and use cpu
        img_tensor = torch.from_numpy(image).type(torch.FloatTensor)
        img_tensor.cpu()

        # add a dimension to image to comply with (B x C x W x H) input of model
        img_tensor = img_tensor.unsqueeze_(0)

        # set model to evaluation mode and forward the image with gradients off
        self._model.eval()
        with torch.no_grad():
            output = self._model.forward(img_tensor)
 
        pred_confs = output.topk(topk)[0]
        pred_classes = output.topk(topk)[1]
        
        pred_confs = np.array(pred_confs)[0]
        pred_classes = np.array(pred_classes)[0]

        return pred_confs, pred_classes
