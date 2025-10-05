import { useState, useCallback } from 'react';
import { mlAPI, TrainingRequest, PredictionRequest, TrainedModel, TrainingResponse, PredictionResponse } from '../lib/api';

export interface UseMLAPIState {
  loading: boolean;
  error: string | null;
  models: TrainedModel[];
  currentModel: TrainedModel | null;
}

export interface UseMLAPIActions {
  trainModel: (data: TrainingRequest) => Promise<TrainingResponse | null>;
  predict: (data: PredictionRequest) => Promise<PredictionResponse | null>;
  listModels: () => Promise<void>;
  getModel: (modelId: number) => Promise<void>;
  clearError: () => void;
  checkHealth: () => Promise<boolean>;
}

export function useMLAPI(): UseMLAPIState & UseMLAPIActions {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<TrainedModel[]>([]);
  const [currentModel, setCurrentModel] = useState<TrainedModel | null>(null);

  const trainModel = useCallback(async (data: TrainingRequest): Promise<TrainingResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await mlAPI.trainModel(data);

      // Actualizar la lista de modelos después del entrenamiento
      await listModels();

      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error desconocido en el entrenamiento';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const predict = useCallback(async (data: PredictionRequest): Promise<PredictionResponse | null> => {
    setLoading(true);
    setError(null);

    try {
      const response = await mlAPI.predict(data);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error desconocido en la predicción';
      setError(errorMessage);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const listModels = useCallback(async (): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      const response = await mlAPI.listModels();
      setModels(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error obteniendo modelos';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const getModel = useCallback(async (modelId: number): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      const response = await mlAPI.getModel(modelId);
      setCurrentModel(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Error obteniendo modelo';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      return await mlAPI.healthCheck();
    } catch {
      return false;
    }
  }, []);

  return {
    loading,
    error,
    models,
    currentModel,
    trainModel,
    predict,
    listModels,
    getModel,
    clearError,
    checkHealth,
  };
}
