'use client';

import { zodResolver } from "@hookform/resolvers/zod";
import { Loader, Play } from "lucide-react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel } from "@/components/ui/form";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { UseDigitAi } from "@/hooks/use-digit-ai";
import { Input } from "../ui/input";

const layerSchema = z.object({
  neurons: z.coerce.number().min(1, "At least 1 neuron is required.").max(1024, "Maximum 1024 neurons."),
});

export const configSchema = z.object({
  epochs: z.coerce.number().min(1).max(100),
  batchSize: z.coerce.number().min(1).max(512),
  trainingSamples: z.coerce.number().min(100).max(60000),
  learningRate: z.coerce.number().min(0.0001).max(1),
  optimizer: z.enum(["adam", "sgd", "rmsprop"]),
  layers: z.array(layerSchema).min(1, "At least one hidden layer is required."),
});

export type ConfigSchema = z.infer<typeof configSchema>;

type ConfigPanelProps = {
  useDigitAiManager: UseDigitAi;
};

export default function ConfigPanel({ useDigitAiManager }: ConfigPanelProps) {
  const {
    config,
    isTraining,
    setConfig,
    handleTrain,
    trainingHistory
  } = useDigitAiManager;

  const form = useForm<ConfigSchema>({
    resolver: zodResolver(configSchema),
    defaultValues: config,
    mode: 'onChange'
  });

  const onSubmit = (values: ConfigSchema) => {
    setConfig(values);
    handleTrain(values);
  };
  
  const finalAccuracy = trainingHistory.length > 0 ? trainingHistory[trainingHistory.length - 1].val_acc : null;

  return (
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="flex flex-wrap items-center gap-4 bg-background/50 p-2 rounded-lg border">
          <FormField control={form.control} name="epochs" render={({ field }) => (
            <FormItem className="flex items-center gap-2">
              <FormLabel className="text-xs">Epochs</FormLabel>
              <FormControl>
                <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <SelectTrigger className="w-20 h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                        {[1, 5, 10, 20, 30].map(e => <SelectItem key={e} value={String(e)}>{e}</SelectItem>)}
                    </SelectContent>
                </Select>
              </FormControl>
            </FormItem>
          )} />
           <FormField control={form.control} name="batchSize" render={({ field }) => (
            <FormItem className="flex items-center gap-2">
              <FormLabel className="text-xs">Batch</FormLabel>
              <FormControl>
                <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <SelectTrigger className="w-20 h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                        {[16, 32, 64, 128, 256].map(b => <SelectItem key={b} value={String(b)}>{b}</SelectItem>)}
                    </SelectContent>
                </Select>
              </FormControl>
            </FormItem>
          )} />
          <FormField control={form.control} name="trainingSamples" render={({ field }) => (
            <FormItem className="flex items-center gap-2">
              <FormLabel className="text-xs">Train Size</FormLabel>
              <FormControl>
                <Select onValueChange={(v) => field.onChange(Number(v))} defaultValue={String(field.value)}>
                    <SelectTrigger className="w-24 h-8 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                        {[1000, 5000, 10000, 20000, 55000].map(s => <SelectItem key={s} value={String(s)}>{s}</SelectItem>)}
                    </SelectContent>
                </Select>
              </FormControl>
            </FormItem>
          )} />

          <div className="flex-grow" />
          
          {finalAccuracy && (
            <div className="text-xs text-muted-foreground">
                Accuracy: <span className="font-bold text-primary">{(finalAccuracy * 100).toFixed(1)}%</span>
            </div>
          )}

          <div className="hidden md:block text-xs text-muted-foreground italic">
            Might take some time
          </div>

          <Button type="submit" className="h-8" disabled={isTraining}>
            {isTraining ? <Loader className="animate-spin" /> : <Play />}
            Train Model
          </Button>
        </form>
      </Form>
  );
}
