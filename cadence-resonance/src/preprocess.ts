/**
 * Preprocess Instacart data into compact cadence matrices
 * Streams large CSV files to avoid memory issues
 */
import { createReadStream } from 'fs';
import { createInterface } from 'readline';
import { resolve } from 'path';
import type { CustomerCadence } from './types';

export interface ProcessedInstacartData {
  customers: Array<{
    customerId: string;
    frequencies: Record<string, number[]>;
    topProducts: Record<string, string[]>;
  }>;
  departments: string[];
}

/**
 * Main preprocessing function
 * @param dataPath Path to the Instacart dataset directory
 * @param targetUsers Number of most active users to process
 * @returns Processed cadence data ready for JSON serialization
 */
export async function preprocessInstacartData(
  dataPath: string,
  targetUsers: number = 500
): Promise<ProcessedInstacartData> {
  console.time('preprocessing');
  
  // Step 1: Load departments (tiny)
  const departments = new Map<string, string>();
  const departmentsPath = resolve(dataPath, 'departments.csv');
  
  for await (const line of readLines(departmentsPath, true)) {
    const [id, name] = line.split(',');
    if (name && name !== 'missing') {
      departments.set(id, name.replace(/"/g, ''));
    }
  }
  console.log(`Loaded ${departments.size} departments`);

  // Step 2: Load products → department mapping and product names (2MB, fine in memory)
  const productToDept = new Map<string, string>();
  const productNames = new Map<string, string>();
  const productsPath = resolve(dataPath, 'products.csv');
  
  for await (const line of readLines(productsPath, true)) {
    const parts = line.split(',');
    const productId = parts[0];
    const productName = parts[1]?.replace(/"/g, '') || `Product ${productId}`;
    const deptId = parts[parts.length - 1]; // department_id is last column
    const deptName = departments.get(deptId);
    if (deptName) {
      productToDept.set(productId, deptName);
      productNames.set(productId, productName);
    }
  }
  console.log(`Loaded ${productToDept.size} product→department mappings`);

  // Step 3: Load orders - build user order timeline (104MB, stream it)
  const orderUser = new Map<string, string>(); // order_id → user_id
  const userOrderTimeline = new Map<string, { orderNum: number; daysSince: number; orderId: string }[]>();
  const userOrderCounts = new Map<string, number>();
  const ordersPath = resolve(dataPath, 'orders.csv');
  
  // First pass: count orders per user to find active shoppers
  console.log('Counting orders per user...');
  let orderCount = 0;
  for await (const line of readLines(ordersPath, true)) {
    const parts = line.split(',');
    if (parts[2] !== 'prior') continue; // Only prior orders
    const userId = parts[1];
    userOrderCounts.set(userId, (userOrderCounts.get(userId) || 0) + 1);
    orderCount++;
    
    if (orderCount % 50000 === 0) {
      console.log(`  ...processed ${orderCount} orders`);
    }
  }
  console.log(`${orderCount} prior orders across ${userOrderCounts.size} users`);
  
  // Get top N most active users
  const topUsers = new Set(
    [...userOrderCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, targetUsers)
      .map(([uid]) => uid)
  );
  console.log(`Selected ${topUsers.size} most active users (min ${
    Math.min(...[...topUsers].map(u => userOrderCounts.get(u) || 0))
  } orders)`);

  // Second pass: build timelines for selected users
  console.log('Building order timelines...');
  orderCount = 0;
  for await (const line of readLines(ordersPath, true)) {
    const parts = line.split(',');
    if (parts[2] !== 'prior') continue;
    const [orderId, userId, , orderNumStr, , , daysSinceStr] = parts;
    
    if (!topUsers.has(userId)) continue;
    
    orderUser.set(orderId, userId);
    
    if (!userOrderTimeline.has(userId)) {
      userOrderTimeline.set(userId, []);
    }
    userOrderTimeline.get(userId)!.push({
      orderNum: parseInt(orderNumStr),
      daysSince: parseFloat(daysSinceStr) || 0,
      orderId
    });
    
    orderCount++;
    if (orderCount % 10000 === 0) {
      console.log(`  ...processed ${orderCount} timeline entries`);
    }
  }
  
  // Sort timelines by order number
  for (const timeline of userOrderTimeline.values()) {
    timeline.sort((a, b) => a.orderNum - b.orderNum);
  }
  console.log(`Built timelines for ${userOrderTimeline.size} users, tracking ${orderUser.size} orders`);

  // Step 4: Stream order_products and build per-user, per-department, per-week counts + top products
  console.log('Streaming order products (this takes a while)...');
  const deptNames = [...new Set(departments.values())].filter(d => d !== 'missing');
  
  // For each user: department → week → count
  const userDeptWeekCounts = new Map<string, Map<string, number[]>>();
  
  // For each user: department → product → total count (for top products)
  const userDeptProductCounts = new Map<string, Map<string, Map<string, number>>>();
  
  // Pre-compute order → week mapping from timelines
  const orderWeek = new Map<string, { userId: string; week: number }>();
  for (const [userId, timeline] of userOrderTimeline) {
    let cumulativeDays = 0;
    for (const entry of timeline) {
      cumulativeDays += entry.daysSince;
      const week = Math.min(51, Math.floor(cumulativeDays / 7));
      orderWeek.set(entry.orderId, { userId, week });
    }
  }
  
  const orderProductsPath = resolve(dataPath, 'order_products__prior.csv');
  let productLines = 0;
  let matched = 0;
  
  for await (const line of readLines(orderProductsPath, true)) {
    productLines++;
    if (productLines % 1_000_000 === 0) {
      console.log(`  ...${(productLines / 1_000_000).toFixed(0)}M lines processed, ${matched} matched`);
    }
    
    const commaIdx1 = line.indexOf(',');
    const commaIdx2 = line.indexOf(',', commaIdx1 + 1);
    const orderId = line.substring(0, commaIdx1);
    const productId = line.substring(commaIdx1 + 1, commaIdx2);
    
    const orderInfo = orderWeek.get(orderId);
    if (!orderInfo) continue;
    
    const dept = productToDept.get(productId);
    if (!dept || dept === 'missing') continue;
    
    matched++;
    const { userId, week } = orderInfo;
    const productName = productNames.get(productId) || `Product ${productId}`;
    
    // Track weekly counts
    if (!userDeptWeekCounts.has(userId)) {
      const m = new Map<string, number[]>();
      deptNames.forEach(d => m.set(d, new Array(52).fill(0)));
      userDeptWeekCounts.set(userId, m);
    }
    
    const deptMap = userDeptWeekCounts.get(userId)!;
    const weekArr = deptMap.get(dept);
    if (weekArr) {
      weekArr[week]++;
    }
    
    // Track product counts for top products
    if (!userDeptProductCounts.has(userId)) {
      const m = new Map<string, Map<string, number>>();
      deptNames.forEach(d => m.set(d, new Map<string, number>()));
      userDeptProductCounts.set(userId, m);
    }
    
    const userProductMap = userDeptProductCounts.get(userId)!;
    const deptProductMap = userProductMap.get(dept)!;
    deptProductMap.set(productName, (deptProductMap.get(productName) || 0) + 1);
  }
  console.log(`Processed ${productLines} product lines, ${matched} matched to target users`);

  // Step 5: Convert to output format with JSON-serializable structure
  console.log('Converting to output format...');
  const customers: ProcessedInstacartData['customers'] = [];
  
  for (const [userId, deptMap] of userDeptWeekCounts) {
    const frequencies: Record<string, number[]> = {};
    const topProducts: Record<string, string[]> = {};
    let activeDepartments = 0;
    
    for (const [dept, weeks] of deptMap) {
      // Only include departments with actual purchases
      if (weeks.some(w => w > 0)) {
        frequencies[dept] = weeks;
        activeDepartments++;
        
        // Get top 5 products for this department
        const productMap = userDeptProductCounts.get(userId)?.get(dept);
        if (productMap && productMap.size > 0) {
          const sortedProducts = [...productMap.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([name]) => name);
          topProducts[dept] = sortedProducts;
        } else {
          topProducts[dept] = [];
        }
      }
    }
    
    // Only include customers with 3+ active departments for meaningful analysis
    if (activeDepartments >= 3) {
      customers.push({
        customerId: userId,
        frequencies,
        topProducts
      });
    }
  }
  
  console.log(`Final dataset: ${customers.length} customers with 3+ active departments`);
  console.timeEnd('preprocessing');
  
  return {
    customers,
    departments: deptNames
  };
}

/**
 * Convert processed data back to CustomerCadence format for pipeline
 */
export function toCustomerCadenceArray(data: ProcessedInstacartData): CustomerCadence[] {
  return data.customers.map(customer => ({
    customerId: customer.customerId,
    categoryFrequencies: new Map(Object.entries(customer.frequencies))
  }));
}

/**
 * Async generator to read lines from a file
 */
async function* readLines(filePath: string, skipHeader = false): AsyncGenerator<string> {
  const rl = createInterface({
    input: createReadStream(filePath),
    crlfDelay: Infinity
  });
  
  let first = true;
  for await (const line of rl) {
    if (first && skipHeader) { 
      first = false; 
      continue; 
    }
    first = false;
    if (line.trim()) {
      yield line;
    }
  }
}