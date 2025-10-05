import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowRight, Database, Brain, LineChart, CheckCircle } from "lucide-react";
import { Navigation } from "@/components/Navigation";

const Home = () => {
  const features = [
    {
      icon: Database,
      title: "Análisis de Datos",
      description: "Carga y analiza tus datos con enfoque MOSEL. Visualiza y prepara información para entrenamiento.",
    },
    {
      icon: Brain,
      title: "Entrenamiento de Modelos",
      description: "Entrena modelos de machine learning con tus datos. Obtén métricas de precisión en tiempo real.",
    },
    {
      icon: LineChart,
      title: "Predicciones Precisas",
      description: "Realiza predicciones individuales o masivas con modelos entrenados. Exporta y analiza resultados.",
    },
  ];

  const steps = [
    "Carga tus datos CSV o conéctate a tu base de datos",
    "Selecciona variables y configura parámetros",
    "Entrena tu modelo con algoritmos optimizados",
    "Realiza predicciones y exporta resultados",
  ];

  return (
    <div className="min-h-screen bg-background">
      <Navigation />

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-3xl mx-auto">
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-primary/10 rounded-full mb-6">
              <span className="w-2 h-2 bg-primary rounded-full animate-pulse" />
              <span className="text-sm font-medium text-primary">GCPA</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
              Exoplanet Finders
            </h1>

            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              En la búsqueda de Exoplanetas! by GCPA
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/analysis">
                <Button size="lg" className="group">
                  Comenzar análisis con datos propios
                  <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link to="/analysis?type=app-data">
                <Button size="lg" className="group">
                  Comenzar análisis con datos del App
                  <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-secondary/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Funcionalidades Principales</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Todo lo que necesitas para análisis predictivo avanzado
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className="bg-card p-8 rounded-xl border border-border hover:border-primary/50 transition-all hover:shadow-lg group"
                >
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Process Section */}
      <section className="py-20 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Proceso Simplificado</h2>
            <p className="text-muted-foreground">
              Cuatro pasos para obtener predicciones precisas
            </p>
          </div>

          <div className="space-y-4">
            {steps.map((step, index) => (
              <div
                key={index}
                className="flex items-start space-x-4 p-6 bg-card rounded-xl border border-border hover:border-primary/50 transition-all"
              >
                <div className="flex-shrink-0 w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center font-semibold">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <p className="text-lg">{step}</p>
                </div>
                <CheckCircle className="flex-shrink-0 w-5 h-5 text-success" />
              </div>
            ))}
          </div>

          <div className="mt-12 text-center">
            <Link to="/analysis">
              <Button size="lg" className="group">
                Iniciar Ahora
                <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12 px-4">
        <div className="max-w-7xl mx-auto text-center text-muted-foreground">
          <p>© 2025 GCPA. Sistema de análisis y predicción de datos de Exoplanetas.</p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
