'use client';

import { cn } from '@/lib/utils';

interface Prediction {
  digit: number;
  confidence: number;
}

interface PredictionChartProps {
  predictions: Prediction[];
}

export default function PredictionChart({ predictions }: PredictionChartProps) {
  const sortedPredictions = [...predictions].sort((a, b) => a.digit - b.digit);
  const topPrediction = predictions.reduce((prev, current) => (prev.confidence > current.confidence) ? prev : current, {digit: -1, confidence: 0});
  const hasPredictions = predictions.some(p => p.confidence > 0.001); // Use a small threshold to avoid showing for near-zero preds

  if (!hasPredictions) {
    return (
        <div className="w-full h-full flex flex-col items-center justify-center text-center">
            <p className="text-muted-foreground">Draw a digit to see what the model predicts.</p>
        </div>
    )
  }
  
  return (
    <div className="w-full h-full flex items-end justify-around gap-2">
      {sortedPredictions.map((prediction) => (
        <div key={prediction.digit} className="flex flex-col items-center justify-end h-full w-full">
          <div className="text-xs text-muted-foreground mb-1">
            {(prediction.confidence * 100).toFixed(1)}%
          </div>
          <div className="w-full h-full bg-accent/20 rounded-t-sm flex items-end">
             <div 
                className={cn(
                  "w-full rounded-t-sm transition-all duration-300",
                  prediction.digit === topPrediction.digit ? "bg-primary" : "bg-accent"
                )}
                style={{ height: `${prediction.confidence * 100}%`}}
             />
          </div>
          <div className="mt-2 text-sm text-muted-foreground border border-border rounded-md w-full text-center py-1">
            {prediction.digit}
          </div>
        </div>
      ))}
    </div>
  );
}
