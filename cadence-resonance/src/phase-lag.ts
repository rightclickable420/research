/**
 * Phase lag computation using cross-correlation
 * Determines optimal timing between category interactions
 */

/**
 * Compute phase lag between two time series using cross-correlation
 * @param seriesA First time series (e.g., category A purchases)
 * @param seriesB Second time series (e.g., category B purchases) 
 * @param maxLag Maximum lag to test (in time units)
 * @returns Object with optimal lag and correlation coefficient
 */
export function computePhaseLag(
  seriesA: number[], 
  seriesB: number[], 
  maxLag: number = 6
): { lag: number; correlation: number } {
  if (seriesA.length !== seriesB.length) {
    throw new Error('Time series must have equal length');
  }
  
  const n = seriesA.length;
  if (n < 4) {
    return { lag: 0, correlation: 0 }; // Not enough data
  }
  
  // Normalize series (zero mean, unit variance)
  const normalizeA = normalize(seriesA);
  const normalizeB = normalize(seriesB);
  
  let bestLag = 0;
  let bestCorrelation = 0;
  
  // Test lags from -maxLag to +maxLag
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const correlation = crossCorrelation(normalizeA, normalizeB, lag);
    
    if (Math.abs(correlation) > Math.abs(bestCorrelation)) {
      bestLag = lag;
      bestCorrelation = correlation;
    }
  }
  
  return { lag: bestLag, correlation: bestCorrelation };
}

/**
 * Normalize time series to zero mean and unit variance
 */
function normalize(series: number[]): number[] {
  const mean = series.reduce((sum, x) => sum + x, 0) / series.length;
  const variance = series.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / series.length;
  const stdDev = Math.sqrt(variance);
  
  if (stdDev === 0) {
    return series.map(() => 0); // Constant series
  }
  
  return series.map(x => (x - mean) / stdDev);
}

/**
 * Compute cross-correlation coefficient for a given lag
 * Positive lag means seriesB leads seriesA
 */
function crossCorrelation(seriesA: number[], seriesB: number[], lag: number): number {
  const n = seriesA.length;
  let sum = 0;
  let count = 0;
  
  for (let i = 0; i < n; i++) {
    const jIndex = i + lag;
    if (jIndex >= 0 && jIndex < n) {
      sum += seriesA[i] * seriesB[jIndex];
      count++;
    }
  }
  
  return count > 0 ? sum / count : 0;
}

/**
 * Smooth time series using simple moving average
 * Helps reduce noise in cross-correlation computation
 */
export function smoothTimeSeries(series: number[], windowSize: number = 3): number[] {
  if (windowSize <= 1) return [...series];
  
  const smoothed: number[] = [];
  const halfWindow = Math.floor(windowSize / 2);
  
  for (let i = 0; i < series.length; i++) {
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(series.length, i + halfWindow + 1);
    
    let sum = 0;
    for (let j = start; j < end; j++) {
      sum += series[j];
    }
    
    smoothed[i] = sum / (end - start);
  }
  
  return smoothed;
}