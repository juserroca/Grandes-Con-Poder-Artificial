#!/usr/bin/env python
"""
Script de prueba para verificar la conversión de tipos de datos
"""
import os
import sys
import django
import pandas as pd

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_backend.settings')
django.setup()

from ml_models.views import convert_data_types

def test_data_conversion():
    """Probar la conversión de tipos de datos"""
    print("🧪 Iniciando prueba de conversión de tipos de datos")
    
    # Crear datos de prueba
    test_data = [
        {'name': 'Juan', 'age': '25', 'salary': '50000.50', 'active': 'true', 'category': 'A'},
        {'name': 'María', 'age': '30', 'salary': '60000.75', 'active': 'false', 'category': 'B'},
        {'name': 'Pedro', 'age': '35', 'salary': '70000.00', 'active': '1', 'category': 'A'},
        {'name': 'Ana', 'age': '28', 'salary': '55000.25', 'active': '0', 'category': 'C'},
    ]
    
    columns = ['name', 'age', 'salary', 'active', 'category']
    df = pd.DataFrame(test_data, columns=columns)
    
    print(f"📊 DataFrame original:")
    print(f"   Shape: {df.shape}")
    print(f"   Tipos: {dict(df.dtypes)}")
    print(f"   Datos:\n{df}")
    
    # Definir tipos de conversión
    column_types = {
        'age': 'int',
        'salary': 'float', 
        'active': 'boolean',
        'category': 'categorical',
        'name': 'string'
    }
    
    print(f"\n🔧 Aplicando conversiones: {column_types}")
    
    # Aplicar conversiones
    df_converted = convert_data_types(df, column_types)
    
    print(f"\n✅ DataFrame después de conversión:")
    print(f"   Shape: {df_converted.shape}")
    print(f"   Tipos: {dict(df_converted.dtypes)}")
    print(f"   Datos:\n{df_converted}")
    
    # Verificar tipos específicos
    print(f"\n🔍 Verificaciones:")
    print(f"   age es int64: {df_converted['age'].dtype == 'Int64'}")
    print(f"   salary es float64: {df_converted['salary'].dtype == 'float64'}")
    print(f"   active es object (boolean): {df_converted['active'].dtype == 'object'}")
    print(f"   category es category: {df_converted['category'].dtype.name == 'category'}")
    print(f"   name es string: {df_converted['name'].dtype == 'string'}")
    
    print(f"\n✅ Prueba completada exitosamente!")

if __name__ == '__main__':
    test_data_conversion()
