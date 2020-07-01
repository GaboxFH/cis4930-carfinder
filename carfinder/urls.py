"""carfinder URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from finder import views
from finder import forms
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import include, url, handler404
# from django.views.generic.simple import direct_to_template
from django.views.generic.base import TemplateView

index = views.Index()

urlpatterns = [
    path('admin/', admin.site.urls),
    # url(r'^$', TemplateView.as_view(template_name='index.html'),
    #     name='home'),
    path('', views.Index.as_view(), name='index'),
    path('upload/', forms.upload, name='upload'),
    path('upload/predictImage', index.predict_image, name="predictImage"),
    path('list/',index.list_of_cars, name='listOfCars')
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# cmd to run the hadler python manage.py collectstatic
handler404 = 'finder.views.error_404'
