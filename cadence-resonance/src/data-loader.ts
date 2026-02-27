/**
 * Instacart data loader
 * Processes CSV files to extract customer cadence patterns
 */

import { CustomerCadence, InstacartData } from './types';
import { promises as fs } from 'fs';
import path from 'path';

interface Order {
  order_id: string;
  user_id: string;
  order_number: number;
  order_dow: number; // Day of week (0-6)
  order_hour_of_day: number;
  days_since_prior_order?: number;
}

interface OrderProduct {
  order_id: string;
  product_id: string;
  add_to_cart_order: number;
  reordered: number;
}

interface Product {
  product_id: string;
  product_name: string;
  aisle_id: string;
  department_id: string;
}

interface Department {
  department_id: string;
  department: string;
}

/**
 * Load and process Instacart data from CSV files
 */
export async function loadInstacartData(dataDir: string): Promise<InstacartData> {
  console.log(`Loading Instacart data from ${dataDir}`);
  
  // Load all CSV files
  const [orders, orderProducts, products, departments] = await Promise.all([
    loadOrdersCSV(path.join(dataDir, 'orders.csv')),
    loadOrderProductsCSV(path.join(dataDir, 'order_products__prior.csv')),
    loadProductsCSV(path.join(dataDir, 'products.csv')),
    loadDepartmentsCSV(path.join(dataDir, 'departments.csv')),
  ]);

  console.log(`Loaded ${orders.length} orders, ${orderProducts.length} order products, ${products.length} products, ${departments.length} departments`);

  // Build lookup maps
  const productToDepartment = new Map<string, string>();
  const departmentIdToName = new Map<string, string>();
  
  departments.forEach(dept => {
    departmentIdToName.set(dept.department_id, dept.department);
  });
  
  products.forEach(product => {
    const deptName = departmentIdToName.get(product.department_id);
    if (deptName) {
      productToDepartment.set(product.product_id, deptName);
    }
  });

  // Group orders by user
  const userOrders = new Map<string, Order[]>();
  orders.forEach(order => {
    if (!userOrders.has(order.user_id)) {
      userOrders.set(order.user_id, []);
    }
    userOrders.get(order.user_id)!.push(order);
  });

  // Build order to products mapping
  const orderToProducts = new Map<string, OrderProduct[]>();
  orderProducts.forEach(op => {
    if (!orderToProducts.has(op.order_id)) {
      orderToProducts.set(op.order_id, []);
    }
    orderToProducts.get(op.order_id)!.push(op);
  });

  // Process customers to extract cadence patterns
  const customers: CustomerCadence[] = [];
  const departmentNames = Array.from(new Set(departments.map(d => d.department)));

  let processedUsers = 0;
  for (const [userId, userOrderList] of userOrders) {
    if (processedUsers >= 1000) break; // Limit for demo purposes
    
    // Sort orders by order number to get chronological sequence
    userOrderList.sort((a, b) => a.order_number - b.order_number);
    
    const cadence = extractCadencePattern(userId, userOrderList, orderToProducts, productToDepartment, departmentNames);
    
    // Only include customers with sufficient activity
    if (cadence.categoryFrequencies.size > 1) {
      customers.push(cadence);
    }
    
    processedUsers++;
    if (processedUsers % 100 === 0) {
      console.log(`Processed ${processedUsers} customers`);
    }
  }

  console.log(`Generated cadence patterns for ${customers.length} customers`);

  return {
    customers,
    departments: departmentNames
  };
}

/**
 * Extract weekly cadence pattern for a customer
 */
function extractCadencePattern(
  userId: string,
  orders: Order[],
  orderToProducts: Map<string, OrderProduct[]>,
  productToDepartment: Map<string, string>,
  departmentNames: string[]
): CustomerCadence {
  // Initialize 52-week frequency map for each department
  const categoryFrequencies = new Map<string, number[]>();
  departmentNames.forEach(dept => {
    categoryFrequencies.set(dept, new Array(52).fill(0));
  });

  // Convert orders to weekly timeline
  const firstOrder = orders[0];
  if (!firstOrder) {
    return { customerId: userId, categoryFrequencies };
  }

  let currentWeek = 0;

  for (const order of orders) {
    // Estimate week progression using days_since_prior_order
    if (order.days_since_prior_order !== undefined && order.order_number > 1) {
      const weeksElapsed = Math.floor(order.days_since_prior_order / 7);
      currentWeek = Math.min(51, currentWeek + weeksElapsed);
    }

    // Get products for this order
    const orderProducts = orderToProducts.get(order.order_id) || [];
    
    // Count purchases by department
    const deptCounts = new Map<string, number>();
    
    for (const orderProduct of orderProducts) {
      const department = productToDepartment.get(orderProduct.product_id);
      if (department) {
        deptCounts.set(department, (deptCounts.get(department) || 0) + 1);
      }
    }

    // Add to weekly frequencies
    for (const [department, count] of deptCounts) {
      const frequencies = categoryFrequencies.get(department);
      if (frequencies) {
        frequencies[currentWeek] += count;
      }
    }

    // Advance to next week (assuming weekly shopping pattern)
    currentWeek = Math.min(51, currentWeek + 1);
  }

  return { customerId: userId, categoryFrequencies };
}

/**
 * Parse CSV helper functions
 */
async function loadOrdersCSV(filePath: string): Promise<Order[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  const headers = lines[0].split(',');
  
  const orders: Order[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length < headers.length) continue;
    
    orders.push({
      order_id: values[0],
      user_id: values[1],
      order_number: parseInt(values[2]) || 0,
      order_dow: parseInt(values[3]) || 0,
      order_hour_of_day: parseInt(values[4]) || 0,
      days_since_prior_order: values[5] ? parseInt(values[5]) : undefined,
    });
  }
  
  return orders;
}

async function loadOrderProductsCSV(filePath: string): Promise<OrderProduct[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  
  const orderProducts: OrderProduct[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length < 4) continue;
    
    orderProducts.push({
      order_id: values[0],
      product_id: values[1],
      add_to_cart_order: parseInt(values[2]) || 0,
      reordered: parseInt(values[3]) || 0,
    });
  }
  
  return orderProducts;
}

async function loadProductsCSV(filePath: string): Promise<Product[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  
  const products: Product[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length < 4) continue;
    
    products.push({
      product_id: values[0],
      product_name: values[1],
      aisle_id: values[2],
      department_id: values[3],
    });
  }
  
  return products;
}

async function loadDepartmentsCSV(filePath: string): Promise<Department[]> {
  const content = await fs.readFile(filePath, 'utf-8');
  const lines = content.trim().split('\n');
  
  const departments: Department[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    if (values.length < 2) continue;
    
    departments.push({
      department_id: values[0],
      department: values[1],
    });
  }
  
  return departments;
}

/**
 * Parse CSV line handling quoted values
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  result.push(current.trim());
  return result;
}

/**
 * Generate synthetic demo data when CSV files are not available
 */
export function generateSyntheticData(customerCount: number = 100): InstacartData {
  const departments = [
    'produce', 'dairy eggs', 'beverages', 'snacks', 'frozen', 
    'pantry', 'bakery', 'meat seafood', 'deli', 'personal care'
  ];

  const customers: CustomerCadence[] = [];
  
  for (let i = 0; i < customerCount; i++) {
    const categoryFrequencies = new Map<string, number[]>();
    
    // Generate realistic purchase patterns
    for (const dept of departments) {
      const frequencies = generateRealisticPattern(dept);
      categoryFrequencies.set(dept, frequencies);
    }
    
    customers.push({
      customerId: `synthetic_${i}`,
      categoryFrequencies
    });
  }
  
  return { customers, departments };
}

/**
 * Generate realistic 52-week purchase pattern for a department
 */
function generateRealisticPattern(department: string): number[] {
  const pattern = new Array(52).fill(0);
  
  // Base frequency depends on department type
  const baseFreq = getBaseFrequency(department);
  const seasonality = getSeasonality(department);
  
  for (let week = 0; week < 52; week++) {
    // Seasonal variation
    const seasonalFactor = 1 + 0.3 * Math.sin(2 * Math.PI * week / 52 + seasonality);
    
    // Random variation
    const randomFactor = 0.8 + 0.4 * Math.random();
    
    // Some customers have habits (every 2-3 weeks)
    const habitFactor = week % 3 === 0 ? 1.2 : 1.0;
    
    pattern[week] = Math.max(0, Math.round(baseFreq * seasonalFactor * randomFactor * habitFactor));
  }
  
  return pattern;
}

function getBaseFrequency(department: string): number {
  const frequencies: Record<string, number> = {
    'produce': 3,
    'dairy eggs': 2,
    'beverages': 2,
    'snacks': 1,
    'frozen': 1,
    'pantry': 1,
    'bakery': 1,
    'meat seafood': 2,
    'deli': 1,
    'personal care': 0.5
  };
  
  return frequencies[department] || 1;
}

function getSeasonality(department: string): number {
  // Phase offset for seasonal patterns (in radians)
  const seasonalOffsets: Record<string, number> = {
    'produce': 0, // Peak in summer
    'frozen': Math.PI, // Peak in winter
    'beverages': 0, // Peak in summer
    'bakery': Math.PI / 2, // Peak in holidays
    'snacks': Math.PI / 4,
    'personal care': 0,
  };
  
  return seasonalOffsets[department] || 0;
}