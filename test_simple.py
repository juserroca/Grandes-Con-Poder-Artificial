#!/usr/bin/env python
"""
Script simple para probar la conversión de tipos de datos
"""
import pandas as pd

def convert_data_types(df, column_types):
    """
    Convierte los tipos de datos de las columnas seleccionadas
    """
    print(f"🔄 Iniciando conversión de tipos de datos para {len(column_types)} columnas")
    
    for column, data_type in column_types.items():
        if column not in df.columns:
            print(f"⚠️ Columna '{column}' no encontrada en el DataFrame")
            continue
            
        try:
            print(f"🔄 Convirtiendo columna '{column}' a tipo '{data_type}'")
            
            if data_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                print(f"✅ Columna '{column}' convertida a entero")
                
            elif data_type == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
                print(f"✅ Columna '{column}' convertida a flotante")
                
            elif data_type == 'boolean':
                bool_map = {
                    'true': True, 'false': False, '1': True, '0': False,
                    'yes': True, 'no': False, 'si': True, 'no': False
                }
                df[column] = df[column].astype(str).str.lower().map(bool_map).fillna(df[column])
                print(f"✅ Columna '{column}' convertida a booleano")
                
            elif data_type == 'categorical':
                df[column] = df[column].astype('category')
                print(f"✅ Columna '{column}' convertida a categórico")
                
            elif data_type == 'string':
                df[column] = df[column].astype('string')
                print(f"✅ Columna '{column}' convertida a string")
                
        except Exception as e:
            print(f"❌ Error convirtiendo columna '{column}': {e}")
            continue
    
    return df

def test_conversion():
    """Probar la conversión de tipos de datos"""
    print("🧪 Iniciando prueba de conversión de tipos de datos")
    
    # Crear datos de prueba
    test_data = [
        {'name': 'Juan', 'age': '25', 'salary': '50000.50', 'active': 'true', 'category': 'A'},
        {'name': 'María', 'age': '30', 'salary': '60000.75', 'active': 'false', 'category': 'B'},
        {'name': 'Pedro', 'age': '35', 'salary': '70000.00', 'active': '1', 'category': 'A'},
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
    
    print(f"\n✅ Prueba completada exitosamente!")

if __name__ == '__main__':
    test_conversion()
