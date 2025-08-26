'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { ConfigSchema } from '@/components/digit-ai/config-panel';
import { getHyperparameterSuggestion } from '@/app/actions';
import { useToast } from './use-toast';
import { loadMnistData } from '@/lib/tensorflow-helpers';
import type { InferenceCanvasRef } from '@/components/digit-ai/inference-canvas';

const INITIAL_CONFIG: ConfigSchema = {
  epochs: 5,
  batchSize: 128,
  trainingSamples: 10000,
  learningRate: 0.001,
  optimizer: 'adam',
  layers: [{ neurons: 128 }, { neurons: 64 }],
};

const DEFAULT_PREDICTIONS = Array.from({ length: 10 }, (_, i) => ({ digit: i, confidence: 0 }));

export function useDigitAi() {
  const [config, setConfig] = useState<ConfigSchema>(INITIAL_CONFIG);
  const [model, setModel] = useState<tf.Sequential | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState<any[]>([]);
  const [predictions, setPredictions] = useState(DEFAULT_PREDICTIONS);
  const [activations, setActivations] = useState<number[][]>([]);
  
  const mnistData = useRef<any>(null);
  const canvasRef = useRef<InferenceCanvasRef>(null);
  const { toast } = useToast();

  const createModel = useCallback((currentConfig: ConfigSchema) => {
    const newModel = tf.sequential();
    
    newModel.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));

    currentConfig.layers.forEach(layer => {
        newModel.add(tf.layers.dense({ units: layer.neurons, activation: 'relu' }));
    });

    newModel.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    const optimizer = currentConfig.optimizer === 'sgd' ? tf.train.sgd(currentConfig.learningRate)
      : currentConfig.optimizer === 'rmsprop' ? tf.train.rmsprop(currentConfig.learningRate)
      : tf.train.adam(currentConfig.learningRate);

    newModel.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    setModel(newModel);
    return newModel;
  }, []);
  
  useEffect(() => {
    async function setup() {
      await tf.ready();
      mnistData.current = await loadMnistData();
      createModel(config);
      setIsReady(true);
      toast({ title: "Ready!", description: "TensorFlow.js and data loaded." });
    }
    setup();
  }, [createModel, config, toast]);

  const handleTrain = async (trainingConfig: ConfigSchema) => {
    if (!mnistData.current) {
        toast({ variant: "destructive", title: "Error", description: "MNIST data not loaded." });
        return;
    }
    setIsTraining(true);
    setTrainingHistory([]);
    toast({ title: "Training Started", description: `Epochs: ${trainingConfig.epochs}, Batch: ${trainingConfig.batchSize}` });

    const currentModel = createModel(trainingConfig);
    const { images, labels } = mnistData.current.nextTrainBatch(trainingConfig.trainingSamples);
    const { images: validationImages, labels: validationLabels } = mnistData.current.nextTestBatch(1000);

    try {
        await currentModel.fit(images, labels, {
            epochs: trainingConfig.epochs,
            batchSize: trainingConfig.batchSize,
            validationData: [validationImages, validationLabels],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (logs) {
                        setTrainingHistory(prev => [...prev, { epoch: epoch + 1, ...logs }]);
                    }
                },
            },
        });
        toast({ title: "Training Complete!", variant: "default" });
    } catch(e) {
        const error = e as Error;
        toast({ title: "Training Failed", description: error.message, variant: "destructive" });
    } finally {
        setIsTraining(false);
    }
  };

  const runInference = useCallback(async (imageData: Float32Array) => {
    if (!model || isTraining) return;
    tf.tidy(() => {
        const imageTensor = tf.tensor(imageData, [1, 28, 28, 1]);
        const preds = model.predict(imageTensor) as tf.Tensor;
        const predsData = preds.dataSync();
        setPredictions(Array.from(predsData).map((confidence, digit) => ({ digit, confidence })));

        // Get activations
        const activationTensors = (model.layers.slice(0, -1).map(layer => layer.output) as tf.SymbolicTensor[])
          .map(output => model.execute(imageTensor, output.name)) as tf.Tensor[];

        const activationData = activationTensors.map(t => Array.from(t.dataSync()));
        setActivations(activationData);
    });
  }, [model, isTraining]);

  const handleDraw = useCallback((imageData: Float32Array) => {
      runInference(imageData);
  }, [runInference]);

  const handleClear = useCallback(() => {
    setPredictions(DEFAULT_PREDICTIONS);
    setActivations([]);
  }, []);

  const handleSuggestHyperparams = async () => {
    setIsSuggesting(true);
    const architectureString = config.layers.map(l => l.neurons).join(' -> ');
    try {
      const result = await getHyperparameterSuggestion({
        networkArchitecture: `Input(784) -> ${architectureString} -> Output(10)`,
        dataset: 'MNIST'
      });
      if (result.success && result.data) {
        toast({ title: "AI Suggestion", description: result.data.additionalNotes || `Set learning rate to ${result.data.learningRate}` });
        setConfig(prev => ({
            ...prev,
            learningRate: result.data!.learningRate,
            optimizer: result.data!.optimizer.toLowerCase() as 'adam' | 'sgd' | 'rmsprop',
        }));
        return result.data;
      } else {
        toast({ variant: "destructive", title: "AI Error", description: result.error });
      }
    } catch(e) {
        toast({ variant: "destructive", title: "Error", description: "Failed to get suggestions." });
    } finally {
        setIsSuggesting(false);
    }
    return null;
  };

  const addLayer = (layer: { neurons: number }) => {
    setConfig(prev => ({
      ...prev,
      layers: [...prev.layers, layer]
    }));
  };

  const removeLayer = (index: number) => {
    if (config.layers.length <= 1) {
        toast({ variant: "destructive", title: "Error", description: "Cannot remove the last hidden layer." });
        return;
    };
    setConfig(prev => ({
      ...prev,
      layers: prev.layers.filter((_, i) => i !== index)
    }));
  };

  const updateLayerNeurons = (index: number, neurons: number) => {
    if (neurons < 1 || neurons > 1024) return;
    setConfig(prev => {
        const newLayers = [...prev.layers];
        newLayers[index] = { ...newLayers[index], neurons };
        return { ...prev, layers: newLayers };
    });
  };

  return {
    config,
    setConfig,
    isTraining,
    isReady,
    isSuggesting,
    trainingHistory,
    predictions,
    activations,
    canvasRef,
    handleTrain,
    handleDraw,
    handleClear,
    handleSuggestHyperparams,
    addLayer,
    removeLayer,
    updateLayerNeurons,
  };
}

export type UseDigitAi = ReturnType<typeof useDigitAi>;
