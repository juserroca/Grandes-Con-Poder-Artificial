import { useState, useEffect } from "react";
import { Navigation } from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FileUp, FormInput, Download, TrendingUp, Settings } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Prediction = () => {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [analysisConfig, setAnalysisConfig] = useState<any>(null);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const { toast } = useToast();

  // Cargar configuración del análisis anterior
  useEffect(() => {
    const savedConfig = localStorage.getItem('analysisConfig');
    if (savedConfig) {
      try {
        const config = JSON.parse(savedConfig);
        setAnalysisConfig(config);

        // Inicializar formData con las variables de entrada
        if (config.analysisType === "own-data" && config.inputVariables) {
          const initialFormData: Record<string, string> = {};
          config.inputVariables.forEach((variable: string) => {
            initialFormData[variable] = "";
          });
          setFormData(initialFormData);
        }
      } catch (error) {
        console.error('Error parsing analysis config:', error);
        toast({
          title: "Error de configuración",
          description: "No se pudo cargar la configuración del análisis anterior",
          variant: "destructive",
        });
      }
    } else {
      toast({
        title: "Sin configuración",
        description: "No se encontró configuración del análisis anterior. Regresa al paso de análisis.",
        variant: "destructive",
      });
    }
  }, [toast]);

  const handleInputChange = (variable: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [variable]: value
    }));
  };

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setPrediction(null);
    setPredictionResult(null);

    try {
      // Preparar datos para el backend
      const inputData: Record<string, number> = {};

      // Convertir los valores del formulario a números
      Object.entries(formData).forEach(([key, value]) => {
        if (value && value.trim() !== '') {
          inputData[key] = parseFloat(value);
        }
      });

      // Verificar que todos los campos requeridos estén llenos
      const requiredFields = analysisConfig?.inputVariables || [];
      const missingFields = requiredFields.filter((field: string) => !inputData[field]);

      if (missingFields.length > 0) {
        toast({
          title: "Campos requeridos",
          description: `Por favor completa: ${missingFields.join(', ')}`,
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }

      console.log('Enviando datos de predicción:', {
        model_id: 1, // Usar el modelo entrenado
        input_data: inputData
      });

      // Llamar al endpoint de predicción
      const response = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: 1, // ID del modelo entrenado
          input_data: inputData
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Error en la predicción');
      }

      const result = await response.json();
      console.log('Resultado de la predicción:', result);

      // Procesar resultado
      if (result.prediction !== undefined) {
        const predictionValue = result.prediction;
        const predictionText = predictionValue > 0.5
          ? `Es exoplaneta (${(predictionValue * 100).toFixed(1)}%)`
          : `No es exoplaneta (${((1 - predictionValue) * 100).toFixed(1)}%)`;

        setPrediction(predictionText);
        setPredictionResult(result);

        toast({
          title: "Predicción completada",
          description: "Los resultados están listos",
        });
      } else {
        throw new Error('Formato de respuesta inválido');
      }

    } catch (error) {
      console.error('Error en la predicción:', error);
      toast({
        title: "Error en la predicción",
        description: error instanceof Error ? error.message : "Error desconocido",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      toast({
        title: "Procesando archivo",
        description: `${file.name} cargado correctamente`,
      });
      setTimeout(() => {
        setPrediction("Predicciones masivas completadas");
      }, 1500);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="pt-24 pb-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-3">Predicción de Datos</h1>
            <p className="text-muted-foreground text-lg">
              Realiza predicciones individuales o masivas con tu modelo entrenado
            </p>
          </div>

          <Tabs defaultValue="form" className="space-y-6">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="form">
                <FormInput className="w-4 h-4 mr-2" />
                Formulario Individual
              </TabsTrigger>
              <TabsTrigger value="batch">
                <FileUp className="w-4 h-4 mr-2" />
                Predicción Masiva
              </TabsTrigger>
            </TabsList>

            <TabsContent value="form" className="space-y-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-6">Ingresa los Datos</h3>

                {analysisConfig ? (
                  <form onSubmit={handleFormSubmit} className="space-y-6">
                    {analysisConfig.analysisType === "own-data" && analysisConfig.inputVariables ? (
                      <div className="space-y-4">
                        <div className="p-4 bg-secondary/50 rounded-lg mb-6">
                          <h4 className="font-semibold mb-2">Variables de Entrada Configuradas</h4>
                          <p className="text-sm text-muted-foreground">
                            Ingresa los valores para las variables que seleccionaste en el análisis anterior
                          </p>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                          {analysisConfig.inputVariables.map((variable: string) => (
                            <div key={variable} className="space-y-2">
                              <Label htmlFor={variable}>{variable}</Label>
                              <Input
                                id={variable}
                                placeholder={`Ingresa valor para ${variable}`}
                                type="number"
                                step="0.01"
                                value={formData[variable] || ""}
                                onChange={(e) => handleInputChange(variable, e.target.value)}
                                required
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="p-4 bg-secondary/50 rounded-lg mb-6">
                          <h4 className="font-semibold mb-2">Variables de Entrada del Dataset</h4>
                          <p className="text-sm text-muted-foreground">
                            Ingresa los valores para las variables del dataset de la aplicación
                          </p>
                        </div>

                        <div className="grid md:grid-cols-2 gap-6">
                          <div className="space-y-2">
                            <Label htmlFor="koi_period">Período Orbital (días)</Label>
                            <Input id="koi_period" placeholder="Ingresa período orbital" type="number" step="0.01" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="koi_impact">Impacto</Label>
                            <Input id="koi_impact" placeholder="Ingresa impacto" type="number" step="0.01" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="koi_duration">Duración (horas)</Label>
                            <Input id="koi_duration" placeholder="Ingresa duración" type="number" step="0.01" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="koi_depth">Profundidad</Label>
                            <Input id="koi_depth" placeholder="Ingresa profundidad" type="number" step="0.01" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="koi_prad">Radio Planetario</Label>
                            <Input id="koi_prad" placeholder="Ingresa radio planetario" type="number" step="0.01" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="koi_teq">Temperatura de Equilibrio</Label>
                            <Input id="koi_teq" placeholder="Ingresa temperatura" type="number" step="0.01" />
                          </div>
                        </div>
                      </div>
                    )}

                    <Button type="submit" size="lg" className="w-full" disabled={isLoading}>
                      <TrendingUp className="w-4 h-4 mr-2" />
                      {isLoading ? "Realizando Predicción..." : "Realizar Predicción"}
                    </Button>
                  </form>
                ) : (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <Settings className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                      <p className="text-muted-foreground">Cargando configuración...</p>
                    </div>
                  </div>
                )}
              </Card>

              {prediction && (
                <Card className="p-6 bg-primary/5 border-primary/20">
                  <h3 className="text-lg font-semibold mb-4">Resultado de la Predicción</h3>
                  <div className="p-6 bg-background rounded-lg border border-border">
                    <p className="text-3xl font-bold text-primary mb-2">{prediction}</p>
                    <p className="text-sm text-muted-foreground mb-4">
                      Predicción generada con modelo entrenado
                    </p>

                    {predictionResult && (
                      <div className="mt-4 p-4 bg-secondary/50 rounded-lg">
                        <h4 className="font-semibold mb-2">Detalles Técnicos</h4>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Probabilidad:</span>
                            <span className="ml-2 font-mono">
                              {((predictionResult.prediction || 0) * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Accuracy:</span>
                            <span className="ml-2 font-mono">{predictionResult.model_accuracy * 100}%</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="batch" className="space-y-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-6">Predicción Masiva desde CSV</h3>

                <div className="space-y-6">
                  <div className="p-6 bg-secondary/30 rounded-lg border-2 border-dashed border-border">
                    <div className="text-center">
                      <FileUp className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                      <h4 className="font-semibold mb-2">Sube tu archivo CSV</h4>
                      <p className="text-sm text-muted-foreground mb-4">
                        El archivo debe contener las mismas variables que usaste en el entrenamiento
                      </p>

                      <Label htmlFor="batch-upload" className="cursor-pointer">
                        <Button variant="outline">
                          Seleccionar Archivo CSV
                        </Button>
                        <Input
                          id="batch-upload"
                          type="file"
                          accept=".csv"
                          onChange={handleBatchUpload}
                          className="hidden"
                        />
                      </Label>
                    </div>
                  </div>

                  <Card className="p-4 bg-accent/10 border-accent/20">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-accent/20 rounded-full flex items-center justify-center flex-shrink-0">
                        <Download className="w-4 h-4 text-accent" />
                      </div>
                      <div>
                        <p className="font-medium text-sm mb-1">Descargar Template</p>
                        <p className="text-xs text-muted-foreground mb-3">
                          Usa nuestra plantilla para asegurar el formato correcto
                        </p>
                        <Button variant="outline" size="sm">
                          Descargar Template CSV
                        </Button>
                      </div>
                    </div>
                  </Card>

                  <div className="space-y-3">
                    <h4 className="font-semibold text-sm">Requisitos del archivo:</h4>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start">
                        <span className="text-primary mr-2">•</span>
                        <span>Formato CSV con codificación UTF-8</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary mr-2">•</span>
                        <span>Primera fila con nombres de variables</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary mr-2">•</span>
                        <span>Columnas deben coincidir con variables del modelo</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-primary mr-2">•</span>
                        <span>Valores numéricos sin caracteres especiales</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </Card>

              {prediction && (
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Resultados de Predicción Masiva</h3>

                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 bg-secondary rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Total procesados</p>
                      <p className="text-2xl font-bold">1,245</p>
                    </div>
                    <div className="p-4 bg-success/10 rounded-lg border border-success/20">
                      <p className="text-sm text-muted-foreground mb-1">Exitosos</p>
                      <p className="text-2xl font-bold text-success">1,242</p>
                    </div>
                    <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/20">
                      <p className="text-sm text-muted-foreground mb-1">Con errores</p>
                      <p className="text-2xl font-bold text-destructive">3</p>
                    </div>
                  </div>

                  <Button className="w-full">
                    <Download className="w-4 h-4 mr-2" />
                    Descargar Resultados
                  </Button>
                </Card>
              )}
            </TabsContent>
          </Tabs>

          {/* Historical Results Section */}
          <Card className="p-6 mt-6">
            <h3 className="text-lg font-semibold mb-6">Historial de Predicciones</h3>

            <div className="space-y-3">
              {[1, 2, 3].map((_, i) => (
                <div key={i} className="flex items-center justify-between p-4 bg-secondary/30 rounded-lg hover:bg-secondary/50 transition-colors">
                  <div>
                    <p className="font-medium text-sm">Predicción #{1234 - i}</p>
                    <p className="text-xs text-muted-foreground">Hace {i + 1} hora(s)</p>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-sm">Resultado: 89.{3 - i}%</p>
                    <Button variant="ghost" size="sm" className="h-7 mt-1">
                      Ver detalles
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Prediction;
