/**
 * Cadence Matching Engine - Main exports
 * 
 * A differential geometry-based system for real-time product recommendations
 * using Jacobian fields, phase-lag cross-correlation, and Hessian acceleration analysis.
 */

// Core types
export type {
  CustomerCadence,
  JacobianHotspot, 
  CadenceRecommendation,
  PipelineResult,
  PipelineTiming,
  PipelineStepTiming,
  InstacartData
} from './types';

// Jacobian field computation
export {
  computeJacobianField,
  findHotspots
} from './jacobian';

// Phase lag analysis
export {
  computePhaseLag,
  smoothTimeSeries
} from './phase-lag';

// Hessian acceleration analysis
export {
  computeHessianAtHotspot,
  filterStableHotspots,
  computeStabilityScore,
  analyzeCouplingTrend
} from './hessian';

// Main pipeline
export {
  runCadencePipeline
} from './pipeline';

// Data loading
export {
  loadInstacartData,
  generateSyntheticData
} from './data-loader';

// Re-export everything for convenience
export * from './types';
export * from './jacobian';
export * from './phase-lag';
export * from './hessian';
export * from './pipeline';
export * from './data-loader';