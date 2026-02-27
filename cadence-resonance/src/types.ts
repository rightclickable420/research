/**
 * Core types for the cadence matching engine
 * Uses differential geometry to find product recommendation patterns
 */

export interface CustomerCadence {
  customerId: string;
  /** 52 weekly values per category - Map from category name to weekly purchase counts */
  categoryFrequencies: Map<string, number[]>;
}

export interface JacobianHotspot {
  categoryA: string;
  categoryB: string;
  time: number; // Week index (0-51)
  coupling: number; // Jacobian field strength
  phase?: number; // Phase lag in weeks
  acceleration?: number; // Hessian acceleration value
}

export interface CadenceRecommendation {
  productCategory: string;
  score: number; // Final recommendation score
  reason: string; // Human-readable explanation
  timing: number; // Optimal week for recommendation
  interaction: string; // Which category interaction drove this
}

export interface PipelineStepTiming {
  stepName: string;
  durationMs: number;
  itemsProcessed: number;
}

export interface PipelineTiming {
  totalMs: number;
  steps: PipelineStepTiming[];
}

export interface PipelineResult {
  recommendations: CadenceRecommendation[];
  timing: PipelineTiming;
  hotspots: JacobianHotspot[];
  fieldStats: {
    totalCouplings: number;
    maxCoupling: number;
    avgCoupling: number;
  };
}

export interface InstacartData {
  customers: CustomerCadence[];
  departments: string[];
}