from django.urls import path
from . import views

urlpatterns = [
    path('train-model/', views.train_model, name='train_model'),
    path('predict/', views.predict, name='predict'),
    path('models/', views.list_models, name='list_models'),
    path('models/<int:model_id>/', views.get_model, name='get_model'),
]
