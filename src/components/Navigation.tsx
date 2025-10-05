import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { Database, Brain, LineChart, Home } from "lucide-react";

const navItems = [
  { name: "Inicio", path: "/", icon: Home },
  { name: "Análisis", path: "/analysis", icon: Database },
  { name: "Entrenamiento", path: "/training", icon: Brain },
  { name: "Predicción", path: "/prediction", icon: LineChart },
];

export const Navigation = () => {
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-lg" />
            <span className="font-semibold text-lg">NASA MOREL</span>
          </Link>

          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={cn(
                    "flex items-center space-x-2 px-4 py-2 rounded-lg transition-all",
                    "hover:bg-secondary",
                    isActive && "bg-primary text-primary-foreground hover:bg-primary/90"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline text-sm font-medium">{item.name}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
};
