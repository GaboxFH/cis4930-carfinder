import os
import json

from ml import predictor
from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage

class Index(View):
    test = False
    def __init__(self):
        self._template = 'index.html'
        self._checkpoint_path = 'ml/model/checkpoint.pth'

        with open('ml/model/classes.json', 'r') as fin:
            self._classes = json.load(fin)

        self._predictor = predictor.Predictor(self._checkpoint_path, len(self._classes))

    def get(self, request):
        return render(request, self._template)

    def predict_image(self, request):
        file_obj=request.FILES['filePath']
        fs=FileSystemStorage()
        
        file_path = fs.save(file_obj.name, file_obj)
        file_path = fs.url(file_path)

        file_name = os.path.basename(file_path)
        full_file_path = os.path.join('finder', 'static', 'media', file_name)
        pred_confs, pred_classes = self._predictor.predict(full_file_path)

        predicted_class = self._classes[pred_classes[0]]
        test = True
        context={
            'filePathName': file_path,
            'predictedLabel': predicted_class,
            'test': test
            # 'pred_confs': pred_confs,
            # 'pred_classes': pred_classes
        }
        
        return render(request,'upload.html',context)
