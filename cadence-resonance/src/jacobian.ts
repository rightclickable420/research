/**
 * Jacobian field computation for cadence matching
 * Computes partial derivatives to find category interaction hotspots
 */

import { JacobianHotspot } from './types';

/**
 * Compute Jacobian field using normalized cross-rates of change
 * @param cadenceMatrix time × categories matrix of purchase frequencies
 * @returns time × categories × categories tensor of partial derivatives
 */
export function computeJacobianField(cadenceMatrix: number[][]): number[][][] {
  const timeSteps = cadenceMatrix.length;
  const categories = cadenceMatrix[0]?.length || 0;
  
  if (timeSteps < 3 || categories < 2) {
    throw new Error('Need at least 3 time steps and 2 categories for Jacobian computation');
  }

  // Step 1: Compute rates of change for all categories across time
  const ratesOfChange: number[][] = [];
  for (let i = 0; i < categories; i++) {
    ratesOfChange[i] = [];
    for (let t = 0; t < timeSteps; t++) {
      if (t === 0) {
        // Forward difference for first time step
        ratesOfChange[i][t] = cadenceMatrix[1][i] - cadenceMatrix[0][i];
      } else if (t === timeSteps - 1) {
        // Backward difference for last time step
        ratesOfChange[i][t] = cadenceMatrix[t][i] - cadenceMatrix[t - 1][i];
      } else {
        // Central difference: dI/dt = (I[t+1] - I[t-1]) / 2
        ratesOfChange[i][t] = (cadenceMatrix[t + 1][i] - cadenceMatrix[t - 1][i]) / 2;
      }
    }
  }

  // Step 2: Compute standard deviations for normalization
  const stdDeviations: number[] = [];
  for (let i = 0; i < categories; i++) {
    const rates = ratesOfChange[i];
    const mean = rates.reduce((sum, r) => sum + r, 0) / rates.length;
    const variance = rates.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / rates.length;
    stdDeviations[i] = Math.sqrt(variance);
  }

  // Step 3: Initialize Jacobian tensor: [time][categoryA][categoryB]
  const jacobianField: number[][][] = [];
  for (let t = 0; t < timeSteps; t++) {
    jacobianField[t] = [];
    for (let i = 0; i < categories; i++) {
      jacobianField[t][i] = new Array(categories).fill(0);
    }
  }

  // Step 4: Compute normalized coupling at each time step
  for (let t = 0; t < timeSteps; t++) {
    for (let i = 0; i < categories; i++) {
      for (let j = 0; j < categories; j++) {
        if (i === j) continue; // Skip diagonal (self-interaction)
        
        const rateI = ratesOfChange[i][t];
        const rateJ = ratesOfChange[j][t];
        const sigmaI = stdDeviations[i];
        const sigmaJ = stdDeviations[j];
        
        // Normalized cross-correlation of rates: (dI/dt * dJ/dt) / (sigma_I * sigma_J)
        // This gives correlation of rates, not raw values
        if (sigmaI > 0.001 && sigmaJ > 0.001) { // Avoid division by near-zero
          jacobianField[t][i][j] = (rateI * rateJ) / (sigmaI * sigmaJ);
        } else {
          jacobianField[t][i][j] = 0;
        }
      }
    }
  }

  return jacobianField;
}

/**
 * Extract hotspots from Jacobian field where coupling exceeds threshold
 */
export function findHotspots(
  jacobianField: number[][][], 
  categories: string[], 
  threshold: number = 0.1
): JacobianHotspot[] {
  const hotspots: JacobianHotspot[] = [];
  const timeSteps = jacobianField.length;
  const numCategories = categories.length;

  for (let t = 0; t < timeSteps; t++) {
    for (let i = 0; i < numCategories; i++) {
      for (let j = 0; j < numCategories; j++) {
        if (i === j) continue;
        
        const coupling = Math.abs(jacobianField[t][i][j]);
        if (coupling > threshold) {
          hotspots.push({
            categoryA: categories[i],
            categoryB: categories[j],
            time: t,
            coupling: jacobianField[t][i][j], // Keep sign for directional coupling
          });
        }
      }
    }
  }

  // Sort by coupling strength (absolute value)
  hotspots.sort((a, b) => Math.abs(b.coupling) - Math.abs(a.coupling));
  
  return hotspots;
}