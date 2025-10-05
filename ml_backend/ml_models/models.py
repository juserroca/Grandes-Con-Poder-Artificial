from django.db import models
import json


class TrainedModel(models.Model):
    """Modelo para almacenar información de modelos entrenados"""
    
    MODEL_CHOICES = [
        ('random-forest', 'Random Forest'),
        ('linear-regression', 'Linear Regression'),
        ('neural-network', 'Neural Network'),
        ('svm', 'Support Vector Machine'),
        ('gradient-boosting', 'Gradient Boosting'),
    ]
    
    ANALYSIS_TYPE_CHOICES = [
        ('own-data', 'Own Data'),
        ('app-data', 'App Data'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_CHOICES)
    analysis_type = models.CharField(max_length=10, choices=ANALYSIS_TYPE_CHOICES)
    hyperparameters = models.JSONField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    mae = models.FloatField()
    r2_score = models.FloatField()
    training_time = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    
    # Para datos propios
    target_variable = models.CharField(max_length=100, null=True, blank=True)
    input_variables = models.JSONField(null=True, blank=True)
    file_name = models.CharField(max_length=255, null=True, blank=True)
    
    # Para datos del App
    dataset_name = models.CharField(max_length=255, null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type}) - {self.accuracy:.3f}"


class Prediction(models.Model):
    """Modelo para almacenar predicciones realizadas"""
    
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField()
    prediction_result = models.JSONField()
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction {self.id} - {self.model.name}"
