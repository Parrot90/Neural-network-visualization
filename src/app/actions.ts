'use server';

import { suggestHyperparameters } from '@/ai/flows/suggest-hyperparameters';
import type { SuggestHyperparametersInput } from '@/ai/flows/suggest-hyperparameters';

export async function getHyperparameterSuggestion(input: SuggestHyperparametersInput) {
  try {
    const result = await suggestHyperparameters(input);
    return { success: true, data: result };
  } catch (error) {
    console.error('Error fetching hyperparameter suggestions:', error);
    return { success: false, error: 'Failed to get suggestions from AI.' };
  }
}
