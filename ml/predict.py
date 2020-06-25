import sys
import json

# allow imports from root
sys.path.append('..')

from ml import predictor

checkpoint_path = 'model/checkpoint.pth'
classes_path = 'model/classes.json'

if (len(sys.argv) != 2):
    print('Expected 1 argument: image_path')
    sys.exit()
image_path = sys.argv[1]

with open(classes_path, 'r') as fin:
    classes = json.load(fin)

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
