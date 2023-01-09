from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train-cnn', views.traincnn, name='train-cnn'),
    path('train-auc', views.trainauc, name='train-auc'),
    path('pred-cnn', views.predcnn, name='pred-cnn'),
    path('pred-auc', views.predauc, name='pred-auc'),
    path('training-cnn', views.trainingcnn, name='training-cnn'),
    path('training-auc', views.trainingauc, name='training-auc'),
    path('predict-image', views.predictimage, name='predict-image'),
    path('upload-image', views.uploadimage, name='upload-image'),
]