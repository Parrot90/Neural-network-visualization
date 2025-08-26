
'use client';

import React from 'react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Plus, X } from "lucide-react";
import { Input } from '../ui/input';

interface LayerConfig {
  neurons: number;
}

interface ModelVisualizationProps {
  layersConfig: LayerConfig[];
  activations: number[][];
  onAddLayer: () => void;
  onRemoveLayer: (index: number) => void;
  onNeuronCountChange: (index: number, neurons: number) => void;
}

const MAX_NODES_TO_RENDER = 6;
const NODE_SIZE = "h-8 w-8";

const Neuron = ({ activation }: { activation: number }) => (
  <div
    className={cn(
        NODE_SIZE, 
        "rounded-full border-2 bg-background transition-colors duration-300 flex-shrink-0",
        activation > 0 ? 'border-primary shadow-primary' : 'border-foreground/50',
    )}
    style={{
        boxShadow: activation > 0 ? `0 0 12px hsla(var(--primary-hsl), ${activation * 0.8})` : 'none',
        borderColor: `hsla(var(--primary-hsl), ${activation > 0 ? 1 : 0.5})`,
    }}
  />
);

const Connections = ({ from, to }: { from: number, to: number }) => {
    const fromNodes = Math.min(from, MAX_NODES_TO_RENDER);
    const toNodes = Math.min(to, MAX_NODES_TO_RENDER);

    const fromY = Array.from({ length: fromNodes }, (_, i) => `${(i / (fromNodes - 1)) * 100}%`);
    const toY = Array.from({ length: toNodes }, (_, i) => `${(i / (toNodes - 1)) * 100}%`);
    
    // Handle case with single node to prevent division by zero
    if (fromNodes === 1) fromY[0] = '50%';
    if (toNodes === 1) toY[0] = '50%';

    const lines = [];
    for (let i = 0; i < fromNodes; i++) {
        for (let j = 0; j < toNodes; j++) {
            lines.push(
                <line 
                    key={`${i}-${j}`}
                    x1="0" 
                    y1={fromY[i]}
                    x2="100%" 
                    y2={toY[j]}
                    stroke="url(#grad)"
                    strokeWidth="1"
                />
            );
        }
    }

    return (
        <svg className="absolute top-0 left-0 w-full h-full" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style={{stopColor: 'hsl(var(--border))', stopOpacity: 0.8}} />
                    <stop offset="100%" style={{stopColor: 'hsl(var(--border))', stopOpacity: 0.8}} />
                </linearGradient>
            </defs>
            {lines}
        </svg>
    );
};

const LayerColumn = ({ neuronCount, activations }: { neuronCount: number, activations: number[] }) => {
  const nodesToRender = Math.min(neuronCount, MAX_NODES_TO_RENDER);
  return (
    <div className="flex flex-col items-center justify-between h-full flex-shrink-0 w-8 z-10">
      {Array.from({ length: nodesToRender }).map((_, i) => (
        <Neuron key={i} activation={activations[i] ?? 0} />
      ))}
      {neuronCount > MAX_NODES_TO_RENDER && <div className="text-muted-foreground text-2xl -my-2">...</div>}
    </div>
  );
};

const LayerInfo = ({ title, neuronCount, onRemove, onNeuronCountChange, isRemovable, isEditable }: { title: string, neuronCount: number, onRemove?: () => void, onNeuronCountChange?: (neurons: number) => void, isRemovable: boolean, isEditable: boolean }) => (
    <div className="flex flex-col items-center text-center gap-1 w-20">
        <div className="font-semibold text-sm flex items-center gap-2 h-8">
            {title}
            {isRemovable && onRemove && (
                 <Button variant="ghost" size="icon" className="h-6 w-6 text-muted-foreground hover:text-destructive" onClick={onRemove}>
                    <X className="w-4 h-4" />
                </Button>
            )}
        </div>
        {isEditable ? (
            <Input type="number" 
                   value={neuronCount} 
                   onChange={(e) => onNeuronCountChange?.(Number(e.target.value))}
                   className="w-20 h-8 text-center bg-transparent border-primary/50" />
        ) : (
            <div className="text-xs text-muted-foreground h-8 flex items-center">{neuronCount} neurons</div>
        )}
    </div>
);

export default function ModelVisualization({ layersConfig, activations, onAddLayer, onRemoveLayer, onNeuronCountChange }: ModelVisualizationProps) {
  const allLayers = [{ neurons: 784 }, ...layersConfig, { neurons: 10 }];

  return (
    <div className="relative w-full h-full flex flex-col bg-card-foreground/5 p-4 rounded-lg gap-4">
        <div className='flex-none'>
            <Button variant="outline" size="sm" onClick={onAddLayer} disabled={layersConfig.length >= 4}>
                <Plus className="w-4 h-4 mr-2"/>
                Add Layer
            </Button>
        </div>
        
        {/* Visualization */}
        <div className="flex-grow flex justify-between items-center px-8">
            {allLayers.map((layer, i) => (
                <React.Fragment key={i}>
                    <LayerColumn neuronCount={layer.neurons} activations={activations[i] || []} />
                    {i < allLayers.length - 1 && (
                        <div className="relative flex-grow h-4/5 mx-2">
                           <Connections from={allLayers[i].neurons} to={allLayers[i+1].neurons} />
                           {(i > 0 && i < allLayers.length - 2) && (
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-code text-xs text-accent-foreground/80 bg-accent/50 px-2 py-1 rounded z-10">
                                ReLU
                            </div>
                           )}
                           {i === allLayers.length - 2 && (
                             <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 font-code text-xs text-accent-foreground/80 bg-accent/50 px-2 py-1 rounded z-10">
                                Softmax
                            </div>
                           )}
                        </div>
                    )}
                </React.Fragment>
            ))}
        </div>

        {/* Layer Info */}
        <div className="flex-none flex justify-between items-start px-8">
             {allLayers.map((layer, i) => {
                const isHidden = i > 0 && i < allLayers.length - 1;
                return (
                    <LayerInfo
                        key={i}
                        title={i === 0 ? "Input" : i === allLayers.length - 1 ? "Output" : `Hidden ${i}`}
                        neuronCount={layer.neurons}
                        onRemove={() => onRemoveLayer(i - 1)}
                        isRemovable={isHidden}
                        isEditable={isHidden}
                        onNeuronCountChange={(neurons) => onNeuronCountChange(i - 1, neurons)}
                    />
                );
            })}
        </div>
    </div>
  );
}
