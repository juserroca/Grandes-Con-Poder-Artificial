from rest_framework import serializers
from .models import TrainedModel, Prediction


class TrainedModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainedModel
        fields = '__all__'


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'


class TrainingRequestSerializer(serializers.Serializer):
    """Serializer para la petición de entrenamiento"""
    analysis_type = serializers.ChoiceField(choices=['own-data', 'app-data'])
    model = serializers.CharField(max_length=20)
    hyperparameters = serializers.DictField()
    
    # Para datos propios
    target_variable = serializers.CharField(required=False, allow_blank=True)
    input_variables = serializers.ListField(required=False, allow_empty=True)
    csv_data = serializers.ListField(required=False, allow_empty=True)
    csv_columns = serializers.ListField(required=False, allow_empty=True)
    column_types = serializers.DictField(required=False)  # Nuevo: tipos de datos
    file_name = serializers.CharField(required=False, allow_blank=True)
    
    # Para datos del App
    dataset_name = serializers.CharField(required=False, allow_blank=True)


class PredictionRequestSerializer(serializers.Serializer):
    """Serializer para la petición de predicción"""
    model_id = serializers.IntegerField()
    input_data = serializers.DictField()
