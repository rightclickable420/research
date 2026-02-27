/**
 * Hessian acceleration computation
 * Computes second derivatives to identify stable vs unstable interaction patterns
 */

import { JacobianHotspot } from './types';

/**
 * Compute Hessian (acceleration) at a specific hotspot
 * Uses second derivatives to determine if interaction is accelerating or decelerating
 */
export function computeHessianAtHotspot(
  jacobianField: number[][][],
  hotspot: JacobianHotspot,
  categories: string[]
): { acceleration: number; stable: boolean } {
  const { categoryA, categoryB, time } = hotspot;
  
  // Find category indices
  const categoryAIndex = categories.indexOf(categoryA);
  const categoryBIndex = categories.indexOf(categoryB);
  
  if (categoryAIndex === -1 || categoryBIndex === -1) {
    throw new Error(`Categories not found: ${categoryA}, ${categoryB}`);
  }
  
  const timeSteps = jacobianField.length;
  
  // Need at least 3 time points for second derivative
  if (timeSteps < 3 || time < 1 || time >= timeSteps - 1) {
    return { acceleration: 0, stable: false };
  }
  
  // Get coupling values at t-1, t, t+1
  const couplingPrev = jacobianField[time - 1][categoryAIndex][categoryBIndex];
  const couplingCurr = jacobianField[time][categoryAIndex][categoryBIndex];
  const couplingNext = jacobianField[time + 1][categoryAIndex][categoryBIndex];
  
  // Compute second derivative using finite differences
  // d²f/dt² ≈ f(t+1) - 2f(t) + f(t-1)
  const acceleration = couplingNext - 2 * couplingCurr + couplingPrev;
  
  // Check stability: interaction is stable if acceleration is not too negative
  // Positive acceleration = strengthening interaction
  // Small negative acceleration = stable plateau 
  // Large negative acceleration = rapidly weakening interaction
  const stable = acceleration >= -0.1; // Threshold can be tuned
  
  return { acceleration, stable };
}

/**
 * Filter hotspots to keep only stable or accelerating interactions
 * This removes rapidly decaying patterns that won't be useful for recommendations
 */
export function filterStableHotspots(
  hotspots: JacobianHotspot[],
  jacobianField: number[][][],
  categories: string[],
  accelerationThreshold: number = -0.1
): JacobianHotspot[] {
  const stableHotspots: JacobianHotspot[] = [];
  
  for (const hotspot of hotspots) {
    const { acceleration, stable } = computeHessianAtHotspot(jacobianField, hotspot, categories);
    
    // Keep if stable or accelerating
    if (acceleration >= accelerationThreshold) {
      stableHotspots.push({
        ...hotspot,
        acceleration
      });
    }
  }
  
  return stableHotspots;
}

/**
 * Compute stability score for an interaction pattern
 * Higher score means more stable/predictable pattern
 */
export function computeStabilityScore(acceleration: number): number {
  if (acceleration > 0) {
    // Accelerating pattern - very good for recommendations
    return Math.min(1.0, 0.7 + acceleration * 2);
  } else if (acceleration >= -0.05) {
    // Stable plateau - good for recommendations  
    return 0.6 + Math.abs(acceleration) * 2;
  } else if (acceleration >= -0.1) {
    // Slowly decaying - moderate for recommendations
    return 0.4 + Math.abs(acceleration + 0.1) * 4;
  } else {
    // Rapidly decaying - poor for recommendations
    return Math.max(0.1, 0.3 + acceleration * 2);
  }
}

/**
 * Analyze coupling strength trend over time
 * Returns trend direction and confidence
 */
export function analyzeCouplingTrend(
  jacobianField: number[][][],
  categoryAIndex: number,
  categoryBIndex: number,
  windowSize: number = 5
): { trend: 'increasing' | 'decreasing' | 'stable'; confidence: number } {
  const timeSteps = jacobianField.length;
  if (timeSteps < windowSize) {
    return { trend: 'stable', confidence: 0 };
  }
  
  const couplings: number[] = [];
  for (let t = 0; t < timeSteps; t++) {
    couplings.push(Math.abs(jacobianField[t][categoryAIndex][categoryBIndex]));
  }
  
  // Simple linear regression to detect trend
  const { slope, correlation } = linearRegression(couplings);
  
  const trend = slope > 0.01 ? 'increasing' : 
                slope < -0.01 ? 'decreasing' : 'stable';
  
  return { trend, confidence: Math.abs(correlation) };
}

/**
 * Simple linear regression helper
 */
function linearRegression(values: number[]): { slope: number; correlation: number } {
  const n = values.length;
  if (n < 2) return { slope: 0, correlation: 0 };
  
  const x = Array.from({ length: n }, (_, i) => i);
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = values.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * values[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumYY = values.reduce((sum, yi) => sum + yi * yi, 0);
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  
  // Correlation coefficient
  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
  const correlation = denominator === 0 ? 0 : numerator / denominator;
  
  return { slope, correlation };
}