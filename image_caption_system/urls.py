from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process', views.process, name='process'),
    path('upload', views.upload, name='upload'),
    # path('upload', views.upload, name='upload'),
    path('<int:image_id>/', views.detail, name='details'),
    path('history', views.history, name='history'),
    path('<int:image_id>/delete', views.delete, name='delete'),
]