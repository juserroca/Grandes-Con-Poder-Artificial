from django.contrib import admin
from .models import TrainedModel, Prediction


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'analysis_type', 'accuracy', 'created_at']
    list_filter = ['model_type', 'analysis_type', 'created_at']
    search_fields = ['name', 'model_type']
    readonly_fields = ['created_at']
    ordering = ['-created_at']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'model', 'created_at']
    list_filter = ['model', 'created_at']
    search_fields = ['model__name']
    readonly_fields = ['created_at']
    ordering = ['-created_at']
