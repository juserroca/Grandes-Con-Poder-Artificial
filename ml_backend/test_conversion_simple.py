#!/usr/bin/env python
"""
Script para probar la conversión de tipos de datos
"""
import pandas as pd

def convert_data_types(df, column_types):
    """
    Convierte los tipos de datos de las columnas seleccionadas
    """
    print(f"Iniciando conversion de tipos de datos para {len(column_types)} columnas")
    
    for column, data_type in column_types.items():
        if column not in df.columns:
            print(f"Advertencia: Columna '{column}' no encontrada en el DataFrame")
            continue
            
        try:
            print(f"Convirtiendo columna '{column}' a tipo '{data_type}'")
            
            if data_type == 'int':
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                print(f"OK: Columna '{column}' convertida a entero")
                
            elif data_type == 'float':
                df[column] = pd.to_numeric(df[column], errors='coerce')
                print(f"OK: Columna '{column}' convertida a flotante")
                
            elif data_type == 'boolean':
                bool_map = {
                    'true': True, 'false': False, '1': True, '0': False,
                    'yes': True, 'no': False, 'si': True, 'no': False
                }
                df[column] = df[column].astype(str).str.lower().map(bool_map).fillna(df[column])
                print(f"OK: Columna '{column}' convertida a booleano")
                
            elif data_type == 'string':
                df[column] = df[column].astype('string')
                print(f"OK: Columna '{column}' convertida a string")
                
        except Exception as e:
            print(f"Error convirtiendo columna '{column}': {e}")
            continue
    
    return df

def test_conversion():
    """Probar la conversión de tipos de datos"""
    print("Iniciando prueba de conversion de tipos de datos")
    
    # Simular datos que vienen del request.data
    csv_data = [
        {'name': 'Juan', 'age': '25', 'salary': '50000.50', 'active': 'true', 'category': 'A'},
        {'name': 'Maria', 'age': '30', 'salary': '60000.75', 'active': 'false', 'category': 'B'},
        {'name': 'Pedro', 'age': '35', 'salary': '70000.00', 'active': '1', 'category': 'A'},
    ]
    
    csv_columns = ['name', 'age', 'salary', 'active', 'category']
    
    # Crear DataFrame desde csv_data
    df = pd.DataFrame(csv_data, columns=csv_columns)
    
    print(f"DataFrame creado desde csv_data:")
    print(f"   Shape: {df.shape}")
    print(f"   Tipos originales: {dict(df.dtypes)}")
    print(f"   Datos:")
    print(df)
    
    # Simular columnTypes del request
    column_types = {
        'age': 'int',
        'salary': 'float', 
        'active': 'boolean',
        'category': 'string',
        'name': 'string'
    }
    
    # Filtrar solo las columnas que se van a usar (simulando variables seleccionadas)
    target_variable = 'age'
    input_variables = ['salary', 'active', 'category']
    selected_columns = [target_variable] + input_variables
    relevant_types = {k: v for k, v in column_types.items() if k in selected_columns}
    
    print(f"\nVariables seleccionadas: {selected_columns}")
    print(f"Tipos relevantes: {relevant_types}")
    
    # Aplicar conversiones solo a las columnas seleccionadas
    df_converted = convert_data_types(df, relevant_types)
    
    print(f"\nDataFrame despues de conversion:")
    print(f"   Shape: {df_converted.shape}")
    print(f"   Tipos finales: {dict(df_converted.dtypes)}")
    print(f"   Datos:")
    print(df_converted)
    
    print(f"\nPrueba completada exitosamente!")

if __name__ == '__main__':
    test_conversion()
