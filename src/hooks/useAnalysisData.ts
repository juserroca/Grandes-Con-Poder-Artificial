import { useState, useCallback } from 'react';

export interface AnalysisData {
  analysisType: string;
  model: string;
  hyperparameters: Record<string, any>;
  // Para datos propios
  csvData?: any[];
  csvColumns?: string[];
  columnTypes?: Record<string, string>; // Nuevo: tipos de datos de las columnas
  targetVariable?: string;
  inputVariables?: string[];
  fileName?: string;
  // Para datos del App
  datasetName?: string;
}

// Variable global para mantener los datos en memoria durante la sesión
let globalAnalysisData: AnalysisData | null = null;

export const useAnalysisData = () => {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(globalAnalysisData);

  const setAnalysisDataGlobal = useCallback((data: AnalysisData) => {
    globalAnalysisData = data;
    setAnalysisData(data);
  }, []);

  const clearAnalysisData = useCallback(() => {
    globalAnalysisData = null;
    setAnalysisData(null);
  }, []);

  return {
    analysisData,
    setAnalysisData: setAnalysisDataGlobal,
    clearAnalysisData
  };
};
