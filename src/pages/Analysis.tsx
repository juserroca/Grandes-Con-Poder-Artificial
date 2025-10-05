import { useState } from "react";
import { Navigation } from "@/components/Navigation";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Upload, FileText, BarChart3, Settings, Brain, Cpu } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAnalysisData } from "@/hooks/useAnalysisData";

const Analysis = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<any[]>([]);
  const [csvColumns, setCsvColumns] = useState<string[]>([]);
  const [columnTypes, setColumnTypes] = useState<Record<string, string>>({});
  const [targetVariable, setTargetVariable] = useState<string>("");
  const [inputVariables, setInputVariables] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const { toast } = useToast();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const analysisType = searchParams.get("type");
  const { setAnalysisData } = useAnalysisData();

  const mlModels = [
    { value: "random-forest", label: "Random Forest", description: "Ideal para clasificación y regresión con datos complejos" },
    { value: "linear-regression", label: "Regresión Lineal", description: "Perfecto para relaciones lineales simples" },
    { value: "neural-network", label: "Red Neuronal", description: "Potente para patrones no lineales complejos" },
    { value: "svm", label: "Máquinas de Vectores de Soporte", description: "Excelente para clasificación con márgenes claros" },
    { value: "gradient-boosting", label: "Gradient Boosting", description: "Alto rendimiento en competencias de ML" },
  ];

  const dataTypes = [
    { value: "int", label: "Entero (int)", description: "Números enteros: 1, 2, 3, -5" },
    { value: "float", label: "Decimal (float)", description: "Números decimales: 1.5, 3.14, -2.7" },
    { value: "string", label: "Texto (string)", description: "Cadenas de texto: 'Hola', 'Categoría A'" },
    { value: "boolean", label: "Booleano (boolean)", description: "Verdadero/Falso: true, false, 1, 0" },
    { value: "categorical", label: "Categórico", description: "Categorías limitadas: 'A', 'B', 'C'" },
    { value: "datetime", label: "Fecha/Hora", description: "Fechas y horas: '2023-01-01', '10:30'" },
  ];

  const getModelHyperparameters = (model: string) => {
    const params: Record<string, any> = {};
    switch (model) {
      case "random-forest":
        params.n_estimators = 100;
        params.max_depth = 10;
        params.min_samples_split = 2;
        break;
      case "neural-network":
        params.hidden_layers = 2;
        params.learning_rate = 0.001;
        break;
      case "svm":
        params.C = 1.0;
        params.kernel = "rbf";
        params.gamma = "scale";
        break;
      case "gradient-boosting":
        params.n_estimators = 100;
        params.learning_rate = 0.1;
        params.max_depth = 3;
        break;
      default:
        params.alpha = 0.01;
    }
    return params;
  };

  const parseCSV = (csvText: string) => {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return { data: [], columns: [] };

    const columns = lines[0].split(',').map(col => col.trim().replace(/"/g, ''));
    const data = lines.slice(1).map(line => {
      const values = line.split(',').map(val => val.trim().replace(/"/g, ''));
      const row: any = {};
      columns.forEach((col, index) => {
        row[col] = values[index] || '';
      });
      return row;
    });

    return { data, columns };
  };

  const detectColumnType = (columnName: string, data: any[]) => {
    // Tomar una muestra de los primeros 100 valores no vacíos
    const sampleValues = data
      .map(row => row[columnName])
      .filter(val => val !== '' && val !== null && val !== undefined)
      .slice(0, 100);

    if (sampleValues.length === 0) return 'string';

    // Detectar tipo de dato
    let isNumeric = true;
    let isInteger = true;
    let isBoolean = true;
    let isDate = true;

    for (const value of sampleValues) {
      const strValue = String(value).toLowerCase();

      // Verificar si es numérico
      if (isNaN(Number(value))) {
        isNumeric = false;
      }

      // Verificar si es entero
      if (isNumeric && !Number.isInteger(Number(value))) {
        isInteger = false;
      }

      // Verificar si es booleano
      if (!['true', 'false', '1', '0', 'yes', 'no', 'si', 'no'].includes(strValue)) {
        isBoolean = false;
      }

      // Verificar si es fecha (patrones básicos)
      const datePatterns = [
        /^\d{4}-\d{2}-\d{2}$/, // YYYY-MM-DD
        /^\d{2}\/\d{2}\/\d{4}$/, // MM/DD/YYYY
        /^\d{2}-\d{2}-\d{4}$/, // DD-MM-YYYY
        /^\d{4}\/\d{2}\/\d{2}$/, // YYYY/MM/DD
      ];

      if (!datePatterns.some(pattern => pattern.test(String(value)))) {
        isDate = false;
      }
    }

    // Determinar tipo basado en las verificaciones
    if (isBoolean && sampleValues.length > 0) {
      return 'boolean';
    } else if (isDate && sampleValues.length > 0) {
      return 'datetime';
    } else if (isInteger && isNumeric) {
      return 'int';
    } else if (isNumeric) {
      return 'float';
    } else {
      // Verificar si es categórico (valores únicos limitados)
      const uniqueValues = [...new Set(sampleValues)];
      if (uniqueValues.length <= 10 && uniqueValues.length < sampleValues.length * 0.5) {
        return 'categorical';
      }
      return 'string';
    }
  };

  const handleModelSelect = (model: string) => {
    setSelectedModel(model);
    setHyperparameters(getModelHyperparameters(model));
    toast({
      title: "Modelo seleccionado",
      description: `Has elegido ${mlModels.find(m => m.value === model)?.label}`,
    });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        setSelectedFile(file);

        const reader = new FileReader();
        reader.onload = (event) => {
          const csvText = event.target?.result as string;
          const { data, columns } = parseCSV(csvText);

          setCsvData(data);
          setCsvColumns(columns);
          setTargetVariable("");
          setInputVariables([]);

          // Inicializar tipos de datos (se detectarán cuando se seleccionen las variables)
          setColumnTypes({});

          toast({
            title: "Archivo cargado",
            description: `${file.name} con ${columns.length} columnas y ${data.length} filas está listo para análisis`,
          });
        };
        reader.readAsText(file);
      } else {
        toast({
          title: "Formato no válido",
          description: "Por favor selecciona un archivo CSV",
          variant: "destructive",
        });
      }
    }
  };

  const handleTargetVariableChange = (variable: string) => {
    setTargetVariable(variable);
    // Remover la variable objetivo de las variables de entrada si estaba seleccionada
    setInputVariables(prev => prev.filter(v => v !== variable));

    // Detectar tipo de dato automáticamente para la variable objetivo
    if (csvData.length > 0) {
      const detectedType = detectColumnType(variable, csvData);
      setColumnTypes(prev => ({
        ...prev,
        [variable]: detectedType
      }));
    }
  };

  const handleInputVariableToggle = (variable: string, checked: boolean) => {
    if (checked) {
      setInputVariables(prev => [...prev, variable]);

      // Detectar tipo de dato automáticamente para la variable de entrada
      if (csvData.length > 0) {
        const detectedType = detectColumnType(variable, csvData);
        setColumnTypes(prev => ({
          ...prev,
          [variable]: detectedType
        }));
      }
    } else {
      setInputVariables(prev => prev.filter(v => v !== variable));

      // Remover tipo de dato si se deselecciona la variable
      setColumnTypes(prev => {
        const newTypes = { ...prev };
        delete newTypes[variable];
        return newTypes;
      });
    }
  };

  const handleColumnTypeChange = (column: string, newType: string) => {
    setColumnTypes(prev => ({
      ...prev,
      [column]: newType
    }));
  };

  const handleAnalyze = () => {
    if (analysisType === "app-data") {
      if (!selectedModel) {
        toast({
          title: "Sin modelo seleccionado",
          description: "Por favor selecciona un modelo de ML primero",
          variant: "destructive",
        });
        return;
      }

      // Guardar configuración en memoria global
      const analysisData = {
        analysisType: "app-data",
        model: selectedModel,
        hyperparameters: hyperparameters,
        datasetName: "NASA Kepler Dataset"
      };

      setAnalysisData(analysisData as any);

      // Guardar configuración básica en localStorage
      localStorage.setItem('analysisConfig', JSON.stringify(analysisData));

      toast({
        title: "Configuración completada",
        description: `Modelo ${mlModels.find(m => m.value === selectedModel)?.label} configurado con hiperparámetros`,
      });

      setTimeout(() => {
        navigate("/training");
      }, 2000);
    } else {
      if (!selectedFile) {
        toast({
          title: "Sin archivo",
          description: "Por favor carga un archivo CSV primero",
          variant: "destructive",
        });
        return;
      }

      if (!targetVariable) {
        toast({
          title: "Variable objetivo requerida",
          description: "Por favor selecciona la variable objetivo",
          variant: "destructive",
        });
        return;
      }

      if (inputVariables.length === 0) {
        toast({
          title: "Variables de entrada requeridas",
          description: "Por favor selecciona al menos una variable de entrada",
          variant: "destructive",
        });
        return;
      }

      if (!selectedModel) {
        toast({
          title: "Modelo requerido",
          description: "Por favor selecciona un modelo de ML",
          variant: "destructive",
        });
        return;
      }

      // Guardar configuración en memoria global (evita QuotaExceededError del localStorage)
      const analysisData = {
        analysisType: "own-data",
        model: selectedModel,
        hyperparameters: hyperparameters,
        targetVariable: targetVariable,
        inputVariables: inputVariables,
        fileName: selectedFile?.name || "archivo.csv",
        csvColumns: csvColumns,
        columnTypes: columnTypes, // Incluir tipos de datos
        csvData: csvData // Mantenemos todos los datos en memoria
      };

      setAnalysisData(analysisData as any);

      // Guardar solo la configuración básica en localStorage (sin datos)
      const basicConfig = {
        analysisType: "own-data",
        model: selectedModel,
        hyperparameters: hyperparameters,
        targetVariable: targetVariable,
        inputVariables: inputVariables,
        fileName: selectedFile?.name || "archivo.csv",
        csvColumns: csvColumns,
        dataSize: csvData.length
      };

      localStorage.setItem('analysisConfig', JSON.stringify(basicConfig));

      toast({
        title: "Análisis iniciado",
        description: `Procesando datos con modelo ${mlModels.find(m => m.value === selectedModel)?.label}...`,
      });

      setTimeout(() => {
        navigate("/training");
      }, 2000);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      <div className="pt-24 pb-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-4xl font-bold mb-3">
              {analysisType === "app-data" ? "Análisis con Datos del App" : "Análisis de Datos"}
            </h1>
            <p className="text-muted-foreground text-lg">
              {analysisType === "app-data"
                ? "Selecciona el modelo de ML y configura los hiperparámetros para el entrenamiento"
                : "Carga y prepara tus datos para el entrenamiento del modelo"
              }
            </p>
          </div>

          <Tabs defaultValue={analysisType === "app-data" ? "model" : "upload"} className="space-y-6">
            <TabsList className={`grid w-full ${analysisType === "app-data" ? "grid-cols-2" : "grid-cols-4"}`}>
              {analysisType === "app-data" ? (
                <>
                  <TabsTrigger value="model">
                    <Brain className="w-4 h-4 mr-2" />
                    Seleccionar Modelo
                  </TabsTrigger>
                  <TabsTrigger value="hyperparams">
                    <Cpu className="w-4 h-4 mr-2" />
                    Hiperparámetros
                  </TabsTrigger>
                </>
              ) : (
                <>
                  <TabsTrigger value="upload">
                    <Upload className="w-4 h-4 mr-2" />
                    Cargar Datos
                  </TabsTrigger>
                  <TabsTrigger value="types">
                    <FileText className="w-4 h-4 mr-2" />
                    Tipos de Datos
                  </TabsTrigger>
                  <TabsTrigger value="configure">
                    <Settings className="w-4 h-4 mr-2" />
                    Configurar
                  </TabsTrigger>
                  <TabsTrigger value="preview">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Vista Previa
                  </TabsTrigger>
                </>
              )}
            </TabsList>

            {analysisType === "app-data" && (
              <TabsContent value="model" className="space-y-6">
                <Card className="p-8">
                  <h3 className="text-xl font-semibold mb-6">Selecciona tu Modelo de Machine Learning</h3>
                  <div className="grid gap-4">
                    {mlModels.map((model) => (
                      <div
                        key={model.value}
                        className={`p-4 border rounded-lg cursor-pointer transition-all hover:border-primary/50 ${selectedModel === model.value
                          ? "border-primary bg-primary/5"
                          : "border-border hover:bg-secondary/50"
                          }`}
                        onClick={() => handleModelSelect(model.value)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold text-lg mb-2">{model.label}</h4>
                            <p className="text-muted-foreground">{model.description}</p>
                          </div>
                          <div className="ml-4">
                            <div className={`w-4 h-4 rounded-full border-2 ${selectedModel === model.value
                              ? "bg-primary border-primary"
                              : "border-border"
                              }`}>
                              {selectedModel === model.value && (
                                <div className="w-full h-full rounded-full bg-primary scale-50"></div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </TabsContent>
            )}

            {analysisType === "app-data" && (
              <TabsContent value="hyperparams" className="space-y-6">
                <Card className="p-8">
                  <h3 className="text-xl font-semibold mb-6">Configuración de Hiperparámetros</h3>
                  {selectedModel ? (
                    <div className="space-y-6">
                      <div className="p-4 bg-secondary/50 rounded-lg">
                        <p className="text-sm text-muted-foreground mb-2">Modelo seleccionado:</p>
                        <p className="font-semibold">{mlModels.find(m => m.value === selectedModel)?.label}</p>
                      </div>

                      {selectedModel === "random-forest" && (
                        <div className="space-y-4">
                          <div>
                            <Label className="text-base font-medium">Número de Estimadores</Label>
                            <div className="mt-2">
                              <Slider
                                value={[hyperparameters.n_estimators || 100]}
                                onValueChange={(value) => setHyperparameters(prev => ({ ...prev, n_estimators: value[0] }))}
                                max={500}
                                min={10}
                                step={10}
                                className="w-full"
                              />
                              <p className="text-sm text-muted-foreground mt-1">
                                Valor: {hyperparameters.n_estimators || 100}
                              </p>
                            </div>
                          </div>

                          <div>
                            <Label className="text-base font-medium">Profundidad Máxima</Label>
                            <div className="mt-2">
                              <Slider
                                value={[hyperparameters.max_depth || 10]}
                                onValueChange={(value) => setHyperparameters(prev => ({ ...prev, max_depth: value[0] }))}
                                max={50}
                                min={1}
                                step={1}
                                className="w-full"
                              />
                              <p className="text-sm text-muted-foreground mt-1">
                                Valor: {hyperparameters.max_depth || 10}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {selectedModel === "neural-network" && (
                        <div className="space-y-4">
                          <div>
                            <Label className="text-base font-medium">Capas Ocultas</Label>
                            <div className="mt-2">
                              <Slider
                                value={[hyperparameters.hidden_layers || 2]}
                                onValueChange={(value) => setHyperparameters(prev => ({ ...prev, hidden_layers: value[0] }))}
                                max={10}
                                min={1}
                                step={1}
                                className="w-full"
                              />
                              <p className="text-sm text-muted-foreground mt-1">
                                Valor: {hyperparameters.hidden_layers || 2}
                              </p>
                            </div>
                          </div>

                          <div>
                            <Label className="text-base font-medium">Tasa de Aprendizaje</Label>
                            <div className="mt-2">
                              <Slider
                                value={[hyperparameters.learning_rate || 0.001]}
                                onValueChange={(value) => setHyperparameters(prev => ({ ...prev, learning_rate: value[0] }))}
                                max={0.1}
                                min={0.0001}
                                step={0.0001}
                                className="w-full"
                              />
                              <p className="text-sm text-muted-foreground mt-1">
                                Valor: {hyperparameters.learning_rate || 0.001}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {selectedModel === "svm" && (
                        <div className="space-y-4">
                          <div>
                            <Label className="text-base font-medium">Parámetro C</Label>
                            <div className="mt-2">
                              <Slider
                                value={[hyperparameters.C || 1.0]}
                                onValueChange={(value) => setHyperparameters(prev => ({ ...prev, C: value[0] }))}
                                max={10}
                                min={0.1}
                                step={0.1}
                                className="w-full"
                              />
                              <p className="text-sm text-muted-foreground mt-1">
                                Valor: {hyperparameters.C || 1.0}
                              </p>
                            </div>
                          </div>

                          <div>
                            <Label className="text-base font-medium">Kernel</Label>
                            <Select
                              value={hyperparameters.kernel || "rbf"}
                              onValueChange={(value) => setHyperparameters(prev => ({ ...prev, kernel: value }))}
                            >
                              <SelectTrigger className="w-full">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="rbf">RBF</SelectItem>
                                <SelectItem value="linear">Linear</SelectItem>
                                <SelectItem value="poly">Polynomial</SelectItem>
                                <SelectItem value="sigmoid">Sigmoid</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <Brain className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">Primero selecciona un modelo en la pestaña anterior</p>
                    </div>
                  )}
                </Card>
              </TabsContent>
            )}

            {analysisType !== "app-data" && (
              <TabsContent value="upload" className="space-y-6">
                <Card className="p-8">
                  <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-border rounded-lg hover:border-primary/50 transition-colors">
                    <Upload className="w-16 h-16 text-muted-foreground mb-4" />
                    <h3 className="text-xl font-semibold mb-2">Arrastra tu archivo CSV aquí</h3>
                    <p className="text-muted-foreground mb-6">o haz clic para seleccionar</p>

                    <div>
                      <Input
                        id="file-upload"
                        type="file"
                        accept=".csv"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                      <Button
                        variant="outline"
                        className="relative cursor-pointer"
                        onClick={() => document.getElementById('file-upload')?.click()}
                      >
                        <FileText className="w-4 h-4 mr-2" />
                        Seleccionar Archivo
                      </Button>
                    </div>

                    {selectedFile && (
                      <div className="mt-6 p-4 bg-success/10 rounded-lg border border-success/20">
                        <p className="text-sm font-medium text-success">
                          ✓ {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                        </p>
                      </div>
                    )}
                  </div>
                </Card>

                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Enfoque MOSEL</h3>
                  <div className="space-y-4 text-sm">
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-primary font-semibold text-xs">1</span>
                      </div>
                      <div>
                        <p className="font-medium mb-1">Análisis de Módulos</p>
                        <p className="text-muted-foreground">
                          Los datos se organizan en módulos. Si no están adaptados, la plataforma los ajustará automáticamente.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-primary font-semibold text-xs">2</span>
                      </div>
                      <div>
                        <p className="font-medium mb-1">Variables Representativas</p>
                        <p className="text-muted-foreground">
                          Selecciona las variables objetivo y datos relevantes. Personaliza colores y visualizaciones.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-primary/10 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                        <span className="text-primary font-semibold text-xs">3</span>
                      </div>
                      <div>
                        <p className="font-medium mb-1">Validación</p>
                        <p className="text-muted-foreground">
                          El sistema valida la integridad de los datos y sugiere correcciones si es necesario.
                        </p>
                      </div>
                    </div>
                  </div>
                </Card>
              </TabsContent>
            )}

            {analysisType !== "app-data" && (
              <TabsContent value="types" className="space-y-6">
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-6">Configuración de Tipos de Datos</h3>

                  {csvColumns.length > 0 && (targetVariable || inputVariables.length > 0) ? (
                    <div className="space-y-4">
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <p className="text-sm text-blue-800 mb-2">
                          <strong>💡 Consejo:</strong> Configura los tipos de datos solo para las variables que usarás en el entrenamiento. Los tipos se detectan automáticamente, pero puedes ajustarlos manualmente.
                        </p>
                      </div>

                      {/* Variable Objetivo */}
                      {targetVariable && (
                        <div className="space-y-2">
                          <h4 className="font-semibold text-lg text-green-700">Variable Objetivo</h4>
                          <div className="p-4 border-2 border-green-200 rounded-lg bg-green-50">
                            <div className="flex items-center justify-between mb-3">
                              <div>
                                <h5 className="font-semibold text-lg">{targetVariable}</h5>
                                <p className="text-sm text-gray-600">
                                  Muestra: {csvData.length > 0 ? String(csvData[0][targetVariable] || 'N/A').substring(0, 50) : 'N/A'}
                                </p>
                              </div>
                              <div className="text-right">
                                <Label className="text-sm font-medium">Tipo de Dato</Label>
                                <Select
                                  value={columnTypes[targetVariable] || 'string'}
                                  onValueChange={(value) => handleColumnTypeChange(targetVariable, value)}
                                >
                                  <SelectTrigger className="w-48">
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {dataTypes.map((type) => (
                                      <SelectItem key={type.value} value={type.value}>
                                        <div>
                                          <div className="font-medium">{type.label}</div>
                                          <div className="text-xs text-gray-500">{type.description}</div>
                                        </div>
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>

                            {/* Estadísticas de la variable objetivo */}
                            <div className="grid grid-cols-3 gap-4 text-sm">
                              <div>
                                <span className="text-gray-600">Valores únicos:</span>
                                <span className="ml-2 font-medium">
                                  {[...new Set(csvData.map(row => row[targetVariable]))].length}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">Valores nulos:</span>
                                <span className="ml-2 font-medium">
                                  {csvData.filter(row => !row[targetVariable] || row[targetVariable] === '').length}
                                </span>
                              </div>
                              <div>
                                <span className="text-gray-600">Tipo detectado:</span>
                                <span className="ml-2 font-medium text-blue-600">
                                  {dataTypes.find(t => t.value === (columnTypes[targetVariable] || 'string'))?.label}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Variables de Entrada */}
                      {inputVariables.length > 0 && (
                        <div className="space-y-2">
                          <h4 className="font-semibold text-lg text-blue-700">Variables de Entrada</h4>
                          <div className="space-y-3">
                            {inputVariables.map((column) => (
                              <div key={column} className="p-4 border rounded-lg bg-gray-50">
                                <div className="flex items-center justify-between mb-3">
                                  <div>
                                    <h5 className="font-semibold text-lg">{column}</h5>
                                    <p className="text-sm text-gray-600">
                                      Muestra: {csvData.length > 0 ? String(csvData[0][column] || 'N/A').substring(0, 50) : 'N/A'}
                                    </p>
                                  </div>
                                  <div className="text-right">
                                    <Label className="text-sm font-medium">Tipo de Dato</Label>
                                    <Select
                                      value={columnTypes[column] || 'string'}
                                      onValueChange={(value) => handleColumnTypeChange(column, value)}
                                    >
                                      <SelectTrigger className="w-48">
                                        <SelectValue />
                                      </SelectTrigger>
                                      <SelectContent>
                                        {dataTypes.map((type) => (
                                          <SelectItem key={type.value} value={type.value}>
                                            <div>
                                              <div className="font-medium">{type.label}</div>
                                              <div className="text-xs text-gray-500">{type.description}</div>
                                            </div>
                                          </SelectItem>
                                        ))}
                                      </SelectContent>
                                    </Select>
                                  </div>
                                </div>

                                {/* Estadísticas de la variable de entrada */}
                                <div className="grid grid-cols-3 gap-4 text-sm">
                                  <div>
                                    <span className="text-gray-600">Valores únicos:</span>
                                    <span className="ml-2 font-medium">
                                      {[...new Set(csvData.map(row => row[column]))].length}
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-gray-600">Valores nulos:</span>
                                    <span className="ml-2 font-medium">
                                      {csvData.filter(row => !row[column] || row[column] === '').length}
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-gray-600">Tipo detectado:</span>
                                    <span className="ml-2 font-medium text-blue-600">
                                      {dataTypes.find(t => t.value === (columnTypes[column] || 'string'))?.label}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : csvColumns.length > 0 ? (
                    <div className="text-center py-8">
                      <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground mb-2">Primero selecciona las variables en la pestaña "Configurar"</p>
                      <p className="text-sm text-gray-500">Las variables objetivo y de entrada aparecerán aquí para configurar sus tipos de datos</p>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">Primero carga un archivo CSV para configurar los tipos de datos</p>
                    </div>
                  )}
                </Card>
              </TabsContent>
            )}

            {analysisType !== "app-data" && (
              <TabsContent value="configure" className="space-y-6">
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-6">Configuración del Análisis</h3>

                  {csvColumns.length > 0 ? (
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <Label>Variable Objetivo</Label>
                        <Select value={targetVariable} onValueChange={handleTargetVariableChange}>
                          <SelectTrigger>
                            <SelectValue placeholder="Selecciona la variable a predecir" />
                          </SelectTrigger>
                          <SelectContent>
                            {csvColumns.map((column) => (
                              <SelectItem key={column} value={column}>
                                {column}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Variables de Entrada</Label>
                        <div className="max-h-48 overflow-y-auto border rounded-md p-3 space-y-2">
                          {csvColumns
                            .filter(column => column !== targetVariable)
                            .map((column) => (
                              <div key={column} className="flex items-center space-x-2">
                                <Checkbox
                                  id={column}
                                  checked={inputVariables.includes(column)}
                                  onCheckedChange={(checked) =>
                                    handleInputVariableToggle(column, checked as boolean)
                                  }
                                />
                                <Label htmlFor={column} className="text-sm font-normal cursor-pointer">
                                  {column}
                                </Label>
                              </div>
                            ))}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Variables seleccionadas: {inputVariables.length}
                        </p>
                      </div>

                      <div className="space-y-4">
                        <h4 className="text-lg font-semibold">Selección de Modelo ML</h4>
                        <div className="grid gap-3">
                          {mlModels.map((model) => (
                            <div
                              key={model.value}
                              className={`p-3 border rounded-lg cursor-pointer transition-all hover:border-primary/50 ${selectedModel === model.value
                                ? "border-primary bg-primary/5"
                                : "border-border hover:bg-secondary/50"
                                }`}
                              onClick={() => handleModelSelect(model.value)}
                            >
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <h5 className="font-semibold">{model.label}</h5>
                                  <p className="text-sm text-muted-foreground">{model.description}</p>
                                </div>
                                <div className="ml-4">
                                  <div className={`w-4 h-4 rounded-full border-2 ${selectedModel === model.value
                                    ? "bg-primary border-primary"
                                    : "border-border"
                                    }`}>
                                    {selectedModel === model.value && (
                                      <div className="w-full h-full rounded-full bg-primary scale-50"></div>
                                    )}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {selectedModel && (
                        <div className="space-y-4">
                          <h4 className="text-lg font-semibold">Configuración de Hiperparámetros</h4>
                          {selectedModel === "random-forest" && (
                            <div className="space-y-4">
                              <div>
                                <Label className="text-base font-medium">Número de Estimadores</Label>
                                <div className="mt-2">
                                  <Slider
                                    value={[hyperparameters.n_estimators || 100]}
                                    onValueChange={(value) => setHyperparameters(prev => ({ ...prev, n_estimators: value[0] }))}
                                    max={500}
                                    min={10}
                                    step={10}
                                    className="w-full"
                                  />
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Valor: {hyperparameters.n_estimators || 100}
                                  </p>
                                </div>
                              </div>

                              <div>
                                <Label className="text-base font-medium">Profundidad Máxima</Label>
                                <div className="mt-2">
                                  <Slider
                                    value={[hyperparameters.max_depth || 10]}
                                    onValueChange={(value) => setHyperparameters(prev => ({ ...prev, max_depth: value[0] }))}
                                    max={50}
                                    min={1}
                                    step={1}
                                    className="w-full"
                                  />
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Valor: {hyperparameters.max_depth || 10}
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {selectedModel === "neural-network" && (
                            <div className="space-y-4">
                              <div>
                                <Label className="text-base font-medium">Capas Ocultas</Label>
                                <div className="mt-2">
                                  <Slider
                                    value={[hyperparameters.hidden_layers || 2]}
                                    onValueChange={(value) => setHyperparameters(prev => ({ ...prev, hidden_layers: value[0] }))}
                                    max={10}
                                    min={1}
                                    step={1}
                                    className="w-full"
                                  />
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Valor: {hyperparameters.hidden_layers || 2}
                                  </p>
                                </div>
                              </div>

                              <div>
                                <Label className="text-base font-medium">Tasa de Aprendizaje</Label>
                                <div className="mt-2">
                                  <Slider
                                    value={[hyperparameters.learning_rate || 0.001]}
                                    onValueChange={(value) => setHyperparameters(prev => ({ ...prev, learning_rate: value[0] }))}
                                    max={0.1}
                                    min={0.0001}
                                    step={0.0001}
                                    className="w-full"
                                  />
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Valor: {hyperparameters.learning_rate || 0.001}
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {selectedModel === "svm" && (
                            <div className="space-y-4">
                              <div>
                                <Label className="text-base font-medium">Parámetro C</Label>
                                <div className="mt-2">
                                  <Slider
                                    value={[hyperparameters.C || 1.0]}
                                    onValueChange={(value) => setHyperparameters(prev => ({ ...prev, C: value[0] }))}
                                    max={10}
                                    min={0.1}
                                    step={0.1}
                                    className="w-full"
                                  />
                                  <p className="text-sm text-muted-foreground mt-1">
                                    Valor: {hyperparameters.C || 1.0}
                                  </p>
                                </div>
                              </div>

                              <div>
                                <Label className="text-base font-medium">Kernel</Label>
                                <Select
                                  value={hyperparameters.kernel || "rbf"}
                                  onValueChange={(value) => setHyperparameters(prev => ({ ...prev, kernel: value }))}
                                >
                                  <SelectTrigger className="w-full">
                                    <SelectValue />
                                  </SelectTrigger>
                                  <SelectContent>
                                    <SelectItem value="rbf">RBF</SelectItem>
                                    <SelectItem value="linear">Linear</SelectItem>
                                    <SelectItem value="poly">Polynomial</SelectItem>
                                    <SelectItem value="sigmoid">Sigmoid</SelectItem>
                                  </SelectContent>
                                </Select>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <FileText className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">Primero carga un archivo CSV para configurar el análisis</p>
                    </div>
                  )}
                </Card>
              </TabsContent>
            )}

            {analysisType !== "app-data" && (
              <TabsContent value="preview" className="space-y-6">
                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Resumen de Datos</h3>
                  {csvData.length > 0 ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Registros</p>
                        <p className="text-2xl font-bold">{csvData.length}</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Variables</p>
                        <p className="text-2xl font-bold">{csvColumns.length}</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Variable Objetivo</p>
                        <p className="text-lg font-semibold">{targetVariable || "No seleccionada"}</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Variables de Entrada</p>
                        <p className="text-2xl font-bold">{inputVariables.length}</p>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Registros</p>
                        <p className="text-2xl font-bold">-</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Variables</p>
                        <p className="text-2xl font-bold">-</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Valores Nulos</p>
                        <p className="text-2xl font-bold">-</p>
                      </div>
                      <div className="p-4 bg-secondary rounded-lg">
                        <p className="text-sm text-muted-foreground mb-1">Duplicados</p>
                        <p className="text-2xl font-bold">-</p>
                      </div>
                    </div>
                  )}
                </Card>

                <Card className="p-6">
                  <h3 className="text-lg font-semibold mb-4">Vista Previa de Datos</h3>
                  {csvData.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            {csvColumns.map((column, index) => (
                              <th key={index} className="text-left p-2 font-semibold">
                                {column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {csvData.slice(0, 10).map((row, rowIndex) => (
                            <tr key={rowIndex} className="border-b">
                              {csvColumns.map((column, colIndex) => (
                                <td key={colIndex} className="p-2">
                                  {row[column] || "-"}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {csvData.length > 10 && (
                        <p className="text-sm text-muted-foreground mt-2">
                          Mostrando 10 de {csvData.length} registros
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="h-64 flex items-center justify-center bg-secondary rounded-lg">
                      <p className="text-muted-foreground">Carga un archivo CSV para ver la vista previa</p>
                    </div>
                  )}
                </Card>
              </TabsContent>
            )}
          </Tabs>

          <div className="mt-8 flex justify-end">
            <Button size="lg" onClick={handleAnalyze}>
              {analysisType === "app-data" ? "Continuar a Entrenamiento" : "Continuar a Entrenamiento"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;
