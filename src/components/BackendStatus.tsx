import { useState, useEffect } from 'react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Alert, AlertDescription } from './ui/alert';
import { Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { useMLAPI } from '../hooks/useMLAPI';

export function BackendStatus() {
  const { checkHealth } = useMLAPI();
  const [isConnected, setIsConnected] = useState<boolean | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const checkConnection = async () => {
    setIsChecking(true);
    try {
      const connected = await checkHealth();
      setIsConnected(connected);
    } catch {
      setIsConnected(false);
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();

    // Verificar conexión cada 30 segundos
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  if (isConnected === null) {
    return (
      <div className="flex items-center gap-2">
        <RefreshCw className="w-4 h-4 animate-spin" />
        <span className="text-sm text-muted-foreground">Verificando conexión...</span>
      </div>
    );
  }

  if (isConnected) {
    return (
      <div className="flex items-center gap-2">
        <Wifi className="w-4 h-4 text-green-500" />
        <Badge variant="secondary" className="bg-green-100 text-green-800">
          Backend Conectado
        </Badge>
        <Button
          variant="ghost"
          size="sm"
          onClick={checkConnection}
          disabled={isChecking}
          className="h-6 w-6 p-0"
        >
          <RefreshCw className={`w-3 h-3 ${isChecking ? 'animate-spin' : ''}`} />
        </Button>
      </div>
    );
  }

  return (
    <Alert className="border-red-200 bg-red-50">
      <WifiOff className="h-4 w-4 text-red-500" />
      <AlertDescription className="text-red-800">
        <div className="flex items-center justify-between">
          <span>Backend no disponible. Asegúrate de que el servidor Django esté ejecutándose.</span>
          <Button
            variant="outline"
            size="sm"
            onClick={checkConnection}
            disabled={isChecking}
            className="ml-2"
          >
            {isChecking ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              'Reintentar'
            )}
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}
