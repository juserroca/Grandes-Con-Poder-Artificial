/**
 * Configuración de la API para conectar con el backend Django
 */

const API_BASE_URL = 'http://localhost:8000/api';

export interface TrainingRequest {
  analysis_type: 'own-data' | 'app-data';
  model: string;
  hyperparameters: Record<string, any>;
  target_variable?: string;
  input_variables?: string[];
  csv_data?: any[];
  csv_columns?: string[];
  file_name?: string;
  dataset_name?: string;
}

export interface TrainingResponse {
  model_id: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  mae: number;
  r2_score: number;
  training_time: string;
  plots: {
    confusion_matrix?: string;
    feature_importance?: string;
    learning_curve?: string;
  };
}

export interface PredictionRequest {
  model_id: number;
  input_data: Record<string, any>;
}

export interface PredictionResponse {
  prediction_id: number;
  prediction: any;
  model_name: string;
  model_accuracy: number;
}

export interface TrainedModel {
  id: number;
  name: string;
  model_type: string;
  analysis_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  mae: number;
  r2_score: number;
  training_time: string;
  created_at: string;
  target_variable?: string;
  input_variables?: string[];
  file_name?: string;
  dataset_name?: string;
}

/**
 * Cliente API para interactuar con el backend Django
 */
export class MLAPIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Entrena un modelo de machine learning
   */
  async trainModel(data: TrainingRequest): Promise<TrainingResponse> {
    const response = await fetch(`${this.baseURL}/train-model/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.details || error.error || 'Error en el entrenamiento');
    }

    return response.json();
  }

  /**
   * Realiza una predicción con un modelo entrenado
   */
  async predict(data: PredictionRequest): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseURL}/predict/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.details || error.error || 'Error en la predicción');
    }

    return response.json();
  }

  /**
   * Obtiene la lista de todos los modelos entrenados
   */
  async listModels(): Promise<TrainedModel[]> {
    const response = await fetch(`${this.baseURL}/models/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.details || error.error || 'Error obteniendo modelos');
    }

    return response.json();
  }

  /**
   * Obtiene detalles de un modelo específico
   */
  async getModel(modelId: number): Promise<TrainedModel> {
    const response = await fetch(`${this.baseURL}/models/${modelId}/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.details || error.error || 'Error obteniendo modelo');
    }

    return response.json();
  }

  /**
   * Verifica si el servidor está disponible
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/models/`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000), // 5 segundos timeout
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Instancia por defecto del cliente API
export const mlAPI = new MLAPIClient();

// Función de utilidad para mostrar gráficos base64
export const displayPlot = (base64Data: string, containerId: string) => {
  const container = document.getElementById(containerId);
  if (container && base64Data) {
    container.innerHTML = `<img src="data:image/png;base64,${base64Data}" alt="Gráfico" style="max-width: 100%; height: auto;" />`;
  }
};

// Función de utilidad para formatear métricas
export const formatMetric = (value: number, decimals: number = 3): string => {
  return value.toFixed(decimals);
};

// Función de utilidad para formatear tiempo de entrenamiento
export const formatTrainingTime = (timeString: string): string => {
  return timeString;
};
