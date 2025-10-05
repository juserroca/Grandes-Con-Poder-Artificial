import { useState, useEffect } from "react";
import { Navigation } from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Brain, Download, PlayCircle, CheckCircle, Database, FileText, Settings } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAnalysisData } from "@/hooks/useAnalysisData";

const Training = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [analysisConfig, setAnalysisConfig] = useState<any>(null);
  const [trainingResults, setTrainingResults] = useState<any>(null);
  const { toast } = useToast();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { analysisData } = useAnalysisData();

  // Cargar configuración del análisis desde memoria global o localStorage
  useEffect(() => {
    // Priorizar datos de memoria global (contiene todos los datos del CSV)
    if (analysisData) {
      setAnalysisConfig(analysisData);
    } else {
      // Fallback a localStorage si no hay datos en memoria
      const savedConfig = localStorage.getItem('analysisConfig');
      if (savedConfig) {
        try {
          const config = JSON.parse(savedConfig);
          setAnalysisConfig(config);
        } catch (error) {
          console.error('Error parsing analysis config:', error);
          toast({
            title: "Error de configuración",
            description: "No se pudo cargar la configuración del análisis anterior",
            variant: "destructive",
          });
        }
      } else {
        // Si no hay configuración guardada, mostrar mensaje de error
        toast({
          title: "Sin configuración",
          description: "No se encontró configuración del análisis anterior. Regresa al paso de análisis.",
          variant: "destructive",
        });
      }
    }

    // Cargar resultados del entrenamiento si existen
    const savedResults = localStorage.getItem('trainingResults');
    if (savedResults) {
      try {
        const results = JSON.parse(savedResults);
        setTrainingResults(results);
      } catch (error) {
        console.error('Error parsing training results:', error);
      }
    }
  }, [analysisData, toast]);

  const getModelName = (modelValue: string) => {
    const models: Record<string, string> = {
      "random-forest": "Random Forest",
      "linear-regression": "Regresión Lineal",
      "neural-network": "Red Neuronal",
      "svm": "Máquinas de Vectores de Soporte",
      "gradient-boosting": "Gradient Boosting"
    };
    return models[modelValue] || modelValue;
  };

  const startTraining = async () => {
    if (!analysisConfig) {
      toast({
        title: "Sin configuración",
        description: "No hay configuración de análisis disponible",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setProgress(0);

    try {
      // Preparar datos para enviar a Django
      const trainingData = {
        analysis_type: analysisConfig.analysisType,
        model: analysisConfig.model,
        hyperparameters: analysisConfig.hyperparameters,
        // Para datos propios
        target_variable: analysisConfig.targetVariable,
        input_variables: analysisConfig.inputVariables,
        csv_data: analysisConfig.csvData,
        csv_columns: analysisConfig.csvColumns,
        column_types: analysisConfig.columnTypes, // Incluir tipos de datos
        file_name: analysisConfig.fileName,
        // Para datos del App
        dataset_name: analysisConfig.datasetName
      };

      // Simular progreso mientras se envía la petición
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Enviar petición a Django
      const response = await fetch('http://localhost:8000/api/train-model/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData)
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const result = await response.json();

      // Completar progreso
      clearInterval(progressInterval);
      setProgress(100);

      // Guardar resultados
      localStorage.setItem('trainingResults', JSON.stringify(result));
      setTrainingResults(result);

      setIsTraining(false);

      toast({
        title: "Entrenamiento completado",
        description: `Modelo entrenado con accuracy: ${result.accuracy}%`,
      });

    } catch (error) {
      console.error('Error en el entrenamiento:', error);
      setIsTraining(false);
      setProgress(0);

      toast({
        title: "Error en el entrenamiento",
        description: error instanceof Error ? error.message : "Error desconocido",
        variant: "destructive",
      });
    }
  };

  const handleDownloadModel = () => {
    toast({
      title: "Descarga iniciada",
      description: "El modelo se está descargando...",
    });
  };

  const handleContinue = () => {
    navigate("/prediction");
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="pt-24 pb-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-3">Entrenamiento del Modelo</h1>
            <p className="text-muted-foreground text-lg">
              Configura y entrena tu modelo de machine learning
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Configuration Card */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-primary" />
                Configuración del Modelo
              </h3>

              {analysisConfig ? (
                <div className="space-y-6">
                  {/* Información del análisis */}
                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      {analysisConfig.analysisType === "app-data" ? (
                        <Database className="w-4 h-4 text-primary" />
                      ) : (
                        <FileText className="w-4 h-4 text-primary" />
                      )}
                      <span className="text-sm font-medium">
                        {analysisConfig.analysisType === "app-data"
                          ? "Análisis con Datos del App"
                          : "Análisis con Datos Propios"
                        }
                      </span>
                    </div>

                    {analysisConfig.analysisType === "app-data" ? (
                      <div className="p-3 bg-secondary/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Dataset</p>
                        <p className="font-medium">{analysisConfig.datasetName}</p>
                      </div>
                    ) : (
                      <div className="p-3 bg-secondary/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Archivo</p>
                        <p className="font-medium">{analysisConfig.fileName}</p>
                      </div>
                    )}
                  </div>

                  {/* Modelo seleccionado */}
                  <div className="space-y-2">
                    <Label>Modelo Seleccionado</Label>
                    <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{getModelName(analysisConfig.model)}</span>
                        <Badge variant="secondary">Configurado</Badge>
                      </div>
                    </div>
                  </div>

                  {/* Variables (solo para datos propios) */}
                  {analysisConfig.analysisType !== "app-data" && (
                    <div className="space-y-2">
                      <Label>Variables Configuradas</Label>
                      <div className="space-y-2">
                        <div className="p-3 bg-secondary/50 rounded-lg">
                          <p className="text-sm text-muted-foreground mb-1">Variable Objetivo</p>
                          <p className="font-medium">{analysisConfig.targetVariable}</p>
                        </div>
                        <div className="p-3 bg-secondary/50 rounded-lg">
                          <p className="text-sm text-muted-foreground mb-1">Variables de Entrada</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {analysisConfig.inputVariables.map((variable: string) => (
                              <Badge key={variable} variant="outline" className="text-xs">
                                {variable}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Tipos de Datos (solo para datos propios) */}
                  {analysisConfig.analysisType !== "app-data" && analysisConfig.columnTypes && (
                    <div className="space-y-2">
                      <Label>Tipos de Datos Configurados</Label>
                      <div className="space-y-2">
                        {/* Variable Objetivo */}
                        {analysisConfig.targetVariable && analysisConfig.columnTypes[analysisConfig.targetVariable] && (
                          <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <span className="text-sm font-medium text-green-700">{analysisConfig.targetVariable}</span>
                                <Badge variant="outline" className="text-xs bg-green-100 text-green-700">
                                  Objetivo
                                </Badge>
                              </div>
                              <Badge variant="secondary" className="text-xs">
                                {analysisConfig.columnTypes[analysisConfig.targetVariable]}
                              </Badge>
                            </div>
                          </div>
                        )}

                        {/* Variables de Entrada */}
                        {analysisConfig.inputVariables?.map((column) =>
                          analysisConfig.columnTypes[column] ? (
                            <div key={column} className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                  <span className="text-sm font-medium text-blue-700">{column}</span>
                                  <Badge variant="outline" className="text-xs bg-blue-100 text-blue-700">
                                    Entrada
                                  </Badge>
                                </div>
                                <Badge variant="secondary" className="text-xs">
                                  {analysisConfig.columnTypes[column]}
                                </Badge>
                              </div>
                            </div>
                          ) : null
                        )}
                      </div>
                    </div>
                  )}

                  {/* Hiperparámetros */}
                  <div className="space-y-2">
                    <Label>Hiperparámetros Configurados</Label>
                    <div className="space-y-2">
                      {Object.entries(analysisConfig.hyperparameters)
                        .filter(([key, value]) => value !== undefined && value !== null && value !== "")
                        .map(([key, value]) => (
                          <div key={key} className="p-3 bg-secondary/50 rounded-lg">
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium capitalize">
                                {key.replace(/_/g, ' ')}
                              </span>
                              <span className="text-sm text-muted-foreground">{value}</span>
                            </div>
                          </div>
                        ))}
                      {Object.entries(analysisConfig.hyperparameters)
                        .filter(([key, value]) => value !== undefined && value !== null && value !== "").length === 0 && (
                          <div className="p-3 bg-secondary/30 rounded-lg text-center">
                            <p className="text-sm text-muted-foreground">No hay hiperparámetros configurados</p>
                          </div>
                        )}
                    </div>
                  </div>

                  <Button
                    className="w-full"
                    size="lg"
                    onClick={startTraining}
                    disabled={isTraining}
                  >
                    {isTraining ? (
                      <>
                        <Brain className="w-4 h-4 mr-2 animate-pulse" />
                        Entrenando...
                      </>
                    ) : (
                      <>
                        <PlayCircle className="w-4 h-4 mr-2" />
                        Iniciar Entrenamiento
                      </>
                    )}
                  </Button>
                </div>
              ) : (
                <div className="flex items-center justify-center py-8">
                  <div className="text-center">
                    <Settings className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-muted-foreground">Cargando configuración...</p>
                  </div>
                </div>
              )}
            </Card>

            {/* Results Card */}
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-6">Resultados del Entrenamiento</h3>

              {isTraining && (
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Progreso</span>
                    <span className="text-sm text-muted-foreground">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              <div className="space-y-4">
                <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                  <h4 className="text-sm font-semibold mb-3">Métricas de Precisión</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Accuracy</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? `${(trainingResults.accuracy * 100).toFixed(2)}%` : "-"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Precision</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? `${(trainingResults.precision * 100).toFixed(2)}%` : "-"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Recall</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? `${(trainingResults.recall * 100).toFixed(2)}%` : "-"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">F1-Score</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? `${(trainingResults.f1_score * 100).toFixed(2)}%` : "-"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-secondary/50 rounded-lg border border-border">
                  <h4 className="text-sm font-semibold mb-3">Información del Entrenamiento</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Tiempo</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? trainingResults.training_time : "-"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Error (MAE)</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? trainingResults.mae.toFixed(4) : "-"}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">R² Score</span>
                      <span className="text-sm font-medium">
                        {trainingResults ? trainingResults.r2_score.toFixed(4) : "-"}
                      </span>
                    </div>
                  </div>
                </div>

                {trainingResults && (
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={handleDownloadModel}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Descargar Modelo
                  </Button>
                )}
              </div>
            </Card>
          </div>

          {/* Visualization Card */}
          <Card className="p-6 mt-6">
            <h3 className="text-lg font-semibold mb-4">Visualizaciones del Modelo</h3>
            {trainingResults && trainingResults.plots ? (
              <div className="space-y-6">
                {trainingResults.plots.confusion_matrix && (
                  <div>
                    <h4 className="text-md font-medium mb-3">Matriz de Confusión</h4>
                    <div className="h-64 flex items-center justify-center bg-secondary/30 rounded-lg">
                      <img
                        src={`data:image/png;base64,${trainingResults.plots.confusion_matrix}`}
                        alt="Matriz de Confusión"
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  </div>
                )}

                {trainingResults.plots.feature_importance && (
                  <div>
                    <h4 className="text-md font-medium mb-3">Importancia de Variables</h4>
                    <div className="h-64 flex items-center justify-center bg-secondary/30 rounded-lg">
                      <img
                        src={`data:image/png;base64,${trainingResults.plots.feature_importance}`}
                        alt="Importancia de Variables"
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  </div>
                )}

                {trainingResults.plots.learning_curve && (
                  <div>
                    <h4 className="text-md font-medium mb-3">Curva de Aprendizaje</h4>
                    <div className="h-64 flex items-center justify-center bg-secondary/30 rounded-lg">
                      <img
                        src={`data:image/png;base64,${trainingResults.plots.learning_curve}`}
                        alt="Curva de Aprendizaje"
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-80 flex items-center justify-center bg-secondary/30 rounded-lg">
                <p className="text-muted-foreground">
                  {trainingResults
                    ? "No hay gráficos disponibles"
                    : "Inicia el entrenamiento para ver resultados"
                  }
                </p>
              </div>
            )}
          </Card>

          {trainingResults && (
            <div className="mt-8 flex justify-end">
              <Button size="lg" onClick={handleContinue}>
                Continuar a Predicción
              </Button>
            </div>
          )}
        </div>
      </div >
    </div >
  );
};

export default Training;
