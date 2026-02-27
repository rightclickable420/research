/**
 * Full cadence matching pipeline
 * Orchestrates: jacobian scan → hotspot extraction → phase lag → hessian filter → ranking
 */

import { CustomerCadence, CadenceRecommendation, PipelineResult, PipelineStepTiming } from './types';
import { computeJacobianField, findHotspots } from './jacobian';
import { computePhaseLag, smoothTimeSeries } from './phase-lag';
import { filterStableHotspots, computeStabilityScore } from './hessian';

/**
 * Run the complete cadence matching pipeline for a customer
 */
export function runCadencePipeline(
  customer: CustomerCadence,
  currentWeek: number = 26
): PipelineResult {
  const startTime = performance.now();
  const steps: PipelineStepTiming[] = [];
  
  // Extract categories and build time series matrix
  const stepStart = performance.now();
  const categories = Array.from(customer.categoryFrequencies.keys());
  const cadenceMatrix = buildCadenceMatrix(customer.categoryFrequencies, categories);
  steps.push({
    stepName: 'Matrix Construction',
    durationMs: performance.now() - stepStart,
    itemsProcessed: categories.length
  });
  
  if (categories.length < 2 || cadenceMatrix.length < 3) {
    return {
      recommendations: [],
      timing: { totalMs: performance.now() - startTime, steps },
      hotspots: [],
      fieldStats: { totalCouplings: 0, maxCoupling: 0, avgCoupling: 0 }
    };
  }
  
  // Step 1: Compute Jacobian field
  const jacobianStart = performance.now();
  const jacobianField = computeJacobianField(cadenceMatrix);
  steps.push({
    stepName: 'Jacobian Field Computation', 
    durationMs: performance.now() - jacobianStart,
    itemsProcessed: cadenceMatrix.length * categories.length * categories.length
  });
  
  // Step 2: Extract hotspots
  const hotspotsStart = performance.now();
  const threshold = computeAdaptiveThreshold(jacobianField);
  let hotspots = findHotspots(jacobianField, categories, threshold);
  steps.push({
    stepName: 'Hotspot Extraction',
    durationMs: performance.now() - hotspotsStart,
    itemsProcessed: hotspots.length
  });
  
  // Step 3: Phase lag analysis on hotspots
  const phaseLagStart = performance.now();
  for (const hotspot of hotspots) {
    const categoryAData = customer.categoryFrequencies.get(hotspot.categoryA) || [];
    const categoryBData = customer.categoryFrequencies.get(hotspot.categoryB) || [];
    
    if (categoryAData.length > 0 && categoryBData.length > 0) {
      // Smooth data to reduce noise in correlation
      const smoothA = smoothTimeSeries(categoryAData);
      const smoothB = smoothTimeSeries(categoryBData);
      
      const phaseLag = computePhaseLag(smoothA, smoothB, 6);
      hotspot.phase = phaseLag.lag;
    }
  }
  steps.push({
    stepName: 'Phase Lag Analysis',
    durationMs: performance.now() - phaseLagStart,
    itemsProcessed: hotspots.length
  });
  
  // Step 4: Hessian acceleration filter
  const hessianStart = performance.now();
  hotspots = filterStableHotspots(hotspots, jacobianField, categories);
  steps.push({
    stepName: 'Hessian Stability Filter',
    durationMs: performance.now() - hessianStart,
    itemsProcessed: hotspots.length
  });
  
  // Step 5: Generate recommendations
  const recommendationStart = performance.now();
  const recommendations = generateRecommendations(hotspots, customer, currentWeek);
  steps.push({
    stepName: 'Recommendation Generation',
    durationMs: performance.now() - recommendationStart,
    itemsProcessed: recommendations.length
  });
  
  // Compute field statistics
  const fieldStats = computeFieldStatistics(jacobianField);
  
  return {
    recommendations,
    timing: { totalMs: performance.now() - startTime, steps },
    hotspots,
    fieldStats
  };
}

/**
 * Build the cadence matrix from customer frequency data
 */
function buildCadenceMatrix(
  categoryFrequencies: Map<string, number[]>, 
  categories: string[]
): number[][] {
  const weekCount = 52; // Assume 52 weeks of data
  const matrix: number[][] = [];
  
  for (let week = 0; week < weekCount; week++) {
    const weekRow: number[] = [];
    for (const category of categories) {
      const frequencies = categoryFrequencies.get(category) || [];
      weekRow.push(frequencies[week] || 0);
    }
    matrix.push(weekRow);
  }
  
  return matrix;
}

/**
 * Compute adaptive threshold using median absolute deviation (MAD)
 * More robust to outliers than percentile-based thresholds
 */
function computeAdaptiveThreshold(jacobianField: number[][][]): number {
  const couplings: number[] = [];
  
  for (const timeSlice of jacobianField) {
    for (const row of timeSlice) {
      for (const coupling of row) {
        if (coupling !== 0) {
          couplings.push(Math.abs(coupling));
        }
      }
    }
  }
  
  if (couplings.length === 0) return 0.1;
  
  // Compute median absolute deviation (MAD)
  couplings.sort((a, b) => a - b);
  const median = couplings[Math.floor(couplings.length / 2)];
  
  const deviations = couplings.map(c => Math.abs(c - median));
  deviations.sort((a, b) => a - b);
  const mad = deviations[Math.floor(deviations.length / 2)];
  
  // Use median + 2 * MAD as threshold (robust equivalent of ~95th percentile)
  return Math.max(0.05, median + 2 * mad);
}

/**
 * Generate recommendations from stable hotspots
 */
function generateRecommendations(
  hotspots: import('./types').JacobianHotspot[],
  customer: CustomerCadence,
  currentWeek: number
): CadenceRecommendation[] {
  // Deduplicate: merge A→B and B→A into one interaction, keep strongest
  const pairMap = new Map<string, import('./types').JacobianHotspot>();
  for (const hotspot of hotspots) {
    const [a, b] = [hotspot.categoryA, hotspot.categoryB].sort();
    const pairKey = `${a}||${b}`;
    const existing = pairMap.get(pairKey);
    if (!existing || Math.abs(hotspot.coupling) > Math.abs(existing.coupling)) {
      pairMap.set(pairKey, hotspot);
    }
  }
  
  // Sort deduplicated pairs by coupling strength
  const uniqueHotspots = [...pairMap.values()].sort((a, b) => Math.abs(b.coupling) - Math.abs(a.coupling));
  
  const recommendations: CadenceRecommendation[] = [];
  const seenCategories = new Set<string>();
  
  for (const hotspot of uniqueHotspots.slice(0, 10)) {
    const { categoryA, categoryB, coupling, phase = 0, acceleration = 0 } = hotspot;
    
    // Determine which category to recommend — prefer one we haven't recommended yet
    let recommendCategory: string;
    let driverCategory: string;
    if (seenCategories.has(categoryA) && !seenCategories.has(categoryB)) {
      recommendCategory = categoryB;
      driverCategory = categoryA;
    } else if (seenCategories.has(categoryB) && !seenCategories.has(categoryA)) {
      recommendCategory = categoryA;
      driverCategory = categoryB;
    } else {
      recommendCategory = coupling > 0 ? categoryB : categoryA;
      driverCategory = coupling > 0 ? categoryA : categoryB;
    }
    
    // Skip if we already recommended this category
    if (seenCategories.has(recommendCategory)) continue;
    seenCategories.add(recommendCategory);
    
    // Calculate timing: current week + phase offset
    const optimalTiming = Math.max(0, Math.min(51, currentWeek + phase));
    
    // Score based on coupling strength, stability, and timing relevance
    const couplingScore = Math.abs(coupling);
    const stabilityScore = computeStabilityScore(acceleration);
    const timingScore = computeTimingRelevance(optimalTiming, currentWeek);
    
    const score = couplingScore * stabilityScore * timingScore;
    
    // Generate human-readable explanation
    const reason = generateExplanation(
      recommendCategory, 
      driverCategory, 
      coupling, 
      phase, 
      acceleration
    );
    
    recommendations.push({
      productCategory: recommendCategory,
      score,
      reason,
      timing: optimalTiming,
      interaction: `${driverCategory} → ${recommendCategory}`
    });
  }
  
  // Sort by score descending
  recommendations.sort((a, b) => b.score - a.score);
  
  return recommendations.slice(0, 5);
}

/**
 * Compute timing relevance score based on how close optimal timing is to current week
 */
function computeTimingRelevance(optimalWeek: number, currentWeek: number): number {
  const weekDiff = Math.abs(optimalWeek - currentWeek);
  if (weekDiff <= 1) return 1.0;
  if (weekDiff <= 2) return 0.8;
  if (weekDiff <= 4) return 0.6;
  if (weekDiff <= 8) return 0.4;
  return 0.2;
}

/**
 * Generate human-readable explanation for recommendation
 */
function generateExplanation(
  recommendCategory: string,
  driverCategory: string,
  coupling: number,
  phase: number,
  acceleration: number
): string {
  const strength = Math.abs(coupling) > 0.3 ? 'strong' : 
                  Math.abs(coupling) > 0.1 ? 'moderate' : 'weak';
  
  const stability = acceleration > 0.05 ? 'accelerating' :
                   acceleration > -0.05 ? 'stable' : 'declining';
  
  const timing = Math.abs(phase) <= 1 ? 'immediate' :
                phase > 1 ? `${Math.abs(phase)} weeks after` :
                `${Math.abs(phase)} weeks before`;
  
  const direction = coupling > 0 ? 'co-accelerates with' : 'substitutes for';
  
  return `${driverCategory} ${direction} ${recommendCategory} purchases with ${strength} ${stability} coupling. ` +
         `Optimal timing: ${timing} ${driverCategory} activity.`;
}

/**
 * Compute field statistics for visualization
 */
function computeFieldStatistics(jacobianField: number[][][]): {
  totalCouplings: number;
  maxCoupling: number;
  avgCoupling: number;
} {
  let totalCouplings = 0;
  let sumCouplings = 0;
  let maxCoupling = 0;
  
  for (const timeSlice of jacobianField) {
    for (const row of timeSlice) {
      for (const coupling of row) {
        if (coupling !== 0) {
          totalCouplings++;
          sumCouplings += Math.abs(coupling);
          maxCoupling = Math.max(maxCoupling, Math.abs(coupling));
        }
      }
    }
  }
  
  return {
    totalCouplings,
    maxCoupling,
    avgCoupling: totalCouplings > 0 ? sumCouplings / totalCouplings : 0
  };
}