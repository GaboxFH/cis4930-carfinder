from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage



class Index(View):
    template = 'index.html'

    def get(self, request):
        return render(request, self.template)


    def predictImage(request):
        fileObj=request.FILES['filePath']
        fs=FileSystemStorage()
        
        filePathName = fs.save(fileObj.name,fileObj)
        filePathName = fs.url(filePathName)
    
        context={'filePathName':filePathName}
        return render(request,'upload.html',context) 


