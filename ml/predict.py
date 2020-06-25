import sys

from ml import predictor
from ml.util import get_classes

checkpoint_path = 'checkpoint.pth'
train_path = 'stanford-car-dataset-by-classes/car_data/train'

if (len(sys.argv) != 2):
    print('Expected 1 argument: image_path')
    sys.exit()
image_path = sys.argv[1]

classes, class_to_idx = get_classes.get_classes('stanford-car-dataset-by-classes/car_data/train')
num_classes = len(classes)
print(f'{num_classes} classes')

predictor = predictor.Predictor(checkpoint_path, num_classes)

# optionally, pass in topk=?
pred_confs, pred_classes = predictor.predict(image_path)

print('Predicted Confidences')
print(pred_confs)
print('Predicted Classes')
print(pred_classes)
print('Top Predicted Class')
print(classes[pred_classes[0]])
