/**
 * Org-Thermo Fathom Presentation Builder
 * 
 * Takes OrgThermoResult + classified entities and builds a navigable
 * Fathom spatial tree. Domain-agnostic — works with any InteractionEvent source.
 */

import {
  smartTree, type SectionConfig, type AccentColor,
} from '../connectors/smart-builder';
import type {
  OrgThermoResult,
  EntityClassification,
  EntityFlow,
  EntityEntropy,
  CadencePair,
  DownstreamPair,
  EntityFanout,
} from './types';

interface BuildOptions {
  title?: string;
  description?: string;
  sourceLabel?: string; // e.g. "Next.js" or "Engineering Slack"
}

const QUADRANT_EMOJI: Record<string, string> = {
  river: '🏞️',
  waterfall: '🌊',
  bottleneck: '🚧',
  swamp: '🐊',
};

const QUADRANT_COLOR: Record<string, AccentColor> = {
  river: 'emerald',
  waterfall: 'amber',
  bottleneck: 'orange',
  swamp: 'rose',
};

const QUADRANT_DESCRIPTIONS: Record<string, string> = {
  river: 'High flow, low entropy — work moves through predictably',
  waterfall: 'High flow, high entropy — productive but chaotic and fragile',
  bottleneck: 'Low flow, low entropy — predictable blockage point',
  swamp: 'Low flow, high entropy — energy enters and gets trapped',
};

export function buildOrgThermoPresentation(
  result: OrgThermoResult,
  classified: EntityClassification[],
  opts: BuildOptions = {},
) {
  const title = opts.title ?? 'Organizational Thermodynamics';
  const description = opts.description ?? `Analysis of ${opts.sourceLabel ?? 'communication metadata'}`;

  const quadrantCounts = { river: 0, waterfall: 0, bottleneck: 0, swamp: 0 };
  for (const c of classified) quadrantCounts[c.quadrant]++;
  const total = classified.length;

  // ═══ Overview Section ═══
  const overviewConfig: SectionConfig = {
    accent: 'blue',
    kpi: {
      value: `${total}`,
      label: 'Active entities analyzed',
    },
    stats: [
      { label: '🏞️ Rivers', value: `${quadrantCounts.river}` },
      { label: '🌊 Waterfalls', value: `${quadrantCounts.waterfall}` },
      { label: '🚧 Bottlenecks', value: `${quadrantCounts.bottleneck}` },
      { label: '🐊 Swamps', value: `${quadrantCounts.swamp}` },
    ],
    breakdown: {
      data: [
        { label: 'River', value: quadrantCounts.river, color: '#10b981' },
        { label: 'Waterfall', value: quadrantCounts.waterfall, color: '#f59e0b' },
        { label: 'Bottleneck', value: quadrantCounts.bottleneck, color: '#f97316' },
        { label: 'Swamp', value: quadrantCounts.swamp, color: '#f43f5e' },
      ],
      label: 'Quadrant Distribution',
      chartType: 'donut',
    },
    summary: `${quadrantCounts.swamp > quadrantCounts.river ? 'More swamps than rivers — energy is pooling faster than it flows' : 'More rivers than swamps — healthy overall throughput'}`,
    category: 'overview',
  };

  // ═══ Gradient Map (Scatter) ═══
  const gradientMapConfig: SectionConfig = {
    accent: 'violet',
    summary: 'Flow × Entropy — where each entity lands',
    table: {
      headers: ['Entity', 'Quadrant', 'Flow Rate', 'Entropy', 'Throughput', 'Inventory'],
      rows: classified
        .sort((a, b) => {
          const qOrder = { swamp: 0, bottleneck: 1, waterfall: 2, river: 3 };
          return (qOrder[a.quadrant] ?? 0) - (qOrder[b.quadrant] ?? 0);
        })
        .map(c => {
          const flow = result.flow.by_author.find(f => f.entity === c.entity);
          return [
            c.entity,
            `${QUADRANT_EMOJI[c.quadrant]} ${c.quadrant}`,
            c.flow_rate.toFixed(3),
            c.entropy.toFixed(3),
            `${flow?.flow.throughput ?? '?'}`,
            `${flow?.flow.inventory ?? '?'}`,
          ];
        }),
      label: 'All Entities by Quadrant',
    },
    category: 'analysis',
  };

  // ═══ Quadrant Deep-Dives ═══
  const quadrantSections = (['swamp', 'bottleneck', 'waterfall', 'river'] as const).map((q, qi) => {
    const entities = classified.filter(c => c.quadrant === q);
    if (entities.length === 0) return null;

    const topEntities = entities.slice(0, 20);
    const config: SectionConfig = {
      accent: QUADRANT_COLOR[q],
      kpi: {
        value: `${entities.length}`,
        label: `${QUADRANT_EMOJI[q]} ${q.charAt(0).toUpperCase() + q.slice(1)}s`,
      },
      summary: QUADRANT_DESCRIPTIONS[q],
      breakdown: {
        data: topEntities.map(e => ({
          label: e.entity,
          value: q === 'swamp' || q === 'bottleneck' ? e.entropy : e.flow_rate,
        })),
        label: q === 'swamp' || q === 'bottleneck' ? 'By Entropy' : 'By Flow Rate',
        chartType: 'horizontal-bar',
      },
      table: {
        headers: ['Entity', 'Flow Rate', 'Entropy', 'Resolved', 'Open'],
        rows: topEntities.map(e => {
          const flow = result.flow.by_author.find(f => f.entity === e.entity);
          return [
            e.entity,
            e.flow_rate.toFixed(3),
            e.entropy.toFixed(3),
            `${flow?.flow.throughput ?? 0}`,
            `${flow?.flow.inventory ?? 0}`,
          ];
        }),
        label: `${q.charAt(0).toUpperCase() + q.slice(1)} Details`,
      },
      subsections: topEntities.slice(0, 10).map(e => {
        const entropy = result.entropy.by_author.find(ee => ee.entity === e.entity);
        const fanout = result.fanout.by_author.find(f => f.entity === e.entity);
        const flow = result.flow.by_author.find(f => f.entity === e.entity);

        const entityConfig: SectionConfig = {
          accent: QUADRANT_COLOR[q],
          kpi: {
            value: e.flow_rate.toFixed(2),
            label: 'Flow Rate',
            delta: flow ? `${flow.flow.throughput} resolved / ${flow.flow.inventory} open` : undefined,
          },
          stats: [
            { label: 'Entropy', value: e.entropy.toFixed(3) },
            { label: 'Fan-out Slope', value: fanout ? (fanout.slope > 0 ? `+${fanout.slope.toFixed(3)}` : fanout.slope.toFixed(3)) : 'N/A' },
          ],
          category: 'entity',
        };

        // Add outcome breakdown if we have entropy data
        if (entropy) {
          const outcomes = Object.entries(entropy.outcome_counts).filter(([, v]) => v > 0);
          if (outcomes.length > 0) {
            entityConfig.breakdown = {
              data: outcomes.map(([k, v]) => ({
                label: k,
                value: v,
                color: k === 'resolved' ? '#10b981' : k === 'stalled' ? '#f43f5e' : k === 'looped' ? '#f59e0b' : k === 'expanded' ? '#8b5cf6' : '#6b7280',
              })),
              label: 'Thread Outcomes',
              chartType: 'donut',
            };
          }
        }

        // Add fan-out trend if available
        if (fanout && fanout.windows.length > 1) {
          entityConfig.trend = {
            data: fanout.windows.map(w => ({
              date: w.start,
              value: w.unique_participants,
            })),
            label: 'Unique Collaborators Over Time',
            chartType: 'area',
          };
        }

        return { title: e.entity, config: entityConfig };
      }),
      category: 'quadrant',
    };

    return { title: `${QUADRANT_EMOJI[q]} ${q.charAt(0).toUpperCase() + q.slice(1)}s (${entities.length})`, config };
  }).filter(Boolean) as Array<{ title: string; config: SectionConfig }>;

  // ═══ Flow Section ═══
  const flowConfig: SectionConfig = {
    accent: 'emerald',
    kpi: {
      value: result.flow.overall.flow_rate.toFixed(2),
      label: 'Overall Flow Rate',
      delta: `${result.flow.overall.throughput} resolved / ${result.flow.overall.inventory} open`,
    },
    stats: [
      { label: 'Throughput', value: `${result.flow.overall.throughput}` },
      { label: 'Inventory', value: `${result.flow.overall.inventory}` },
    ],
    breakdown: {
      data: result.flow.by_author.slice(0, 15).map(f => ({
        label: f.entity,
        value: f.flow.throughput,
      })),
      label: 'Top Contributors by Throughput',
      chartType: 'horizontal-bar',
    },
    subsections: [{
      title: 'By Label',
      config: {
        breakdown: {
          data: result.flow.by_label.slice(0, 15).map(f => ({
            label: f.entity,
            value: f.flow.throughput,
          })),
          label: 'Labels by Throughput',
          chartType: 'horizontal-bar',
        },
        table: {
          headers: ['Label', 'Resolved', 'Open', 'Flow Rate'],
          rows: result.flow.by_label.slice(0, 25).map(f => [
            f.entity,
            `${f.flow.throughput}`,
            `${f.flow.inventory}`,
            f.flow.flow_rate.toFixed(3),
          ]),
          label: 'Label Flow Details',
        },
      },
    }],
    category: 'metrics',
  };

  // ═══ Entropy Section ═══
  const entropyConfig: SectionConfig = {
    accent: 'rose',
    kpi: {
      value: result.entropy.by_label.length > 0
        ? result.entropy.by_label[0].entity
        : 'N/A',
      label: 'Highest Entropy Area',
      delta: result.entropy.by_label.length > 0
        ? `H=${result.entropy.by_label[0].entropy.toFixed(3)}`
        : undefined,
    },
    breakdown: {
      data: result.entropy.by_label.slice(0, 15).map(e => ({
        label: e.entity,
        value: Math.round(e.entropy * 1000) / 10, // show as percentage-like
      })),
      label: 'Codebase Areas by Entropy',
      chartType: 'horizontal-bar',
    },
    table: {
      headers: ['Area', 'Entropy', 'Resolved', 'Stalled', 'Expanded', 'Looped', 'Escalated'],
      rows: result.entropy.by_label.slice(0, 20).map(e => [
        e.entity,
        e.entropy.toFixed(3),
        `${e.outcome_counts.resolved}`,
        `${e.outcome_counts.stalled}`,
        `${e.outcome_counts.expanded}`,
        `${e.outcome_counts.looped}`,
        `${e.outcome_counts.escalated}`,
      ]),
      label: 'Entropy Breakdown by Area',
    },
    category: 'metrics',
  };

  // ═══ Cadence Section ═══
  const syncPairs = result.cadence.pairs.filter(p => p.phase_alignment > 0.3);
  const cadenceConfig: SectionConfig = {
    accent: 'cyan',
    kpi: {
      value: `${syncPairs.length}`,
      label: 'Synchronized Pairs',
      delta: syncPairs.length > 0
        ? `Strongest: ${syncPairs[0].entity_a} ↔ ${syncPairs[0].entity_b} (${syncPairs[0].phase_alignment.toFixed(2)})`
        : undefined,
    },
    table: {
      headers: ['Entity A', 'Entity B', 'Phase Alignment'],
      rows: result.cadence.pairs.slice(0, 30).map(p => [
        p.entity_a,
        p.entity_b,
        p.phase_alignment.toFixed(3),
      ]),
      label: 'Cadence Pairs',
    },
    // Build a heatmap-style breakdown showing who syncs with whom
    subsections: (() => {
      // Find hub entities (appear in most pairs)
      const hubCounts = new Map<string, number>();
      for (const p of syncPairs) {
        hubCounts.set(p.entity_a, (hubCounts.get(p.entity_a) ?? 0) + 1);
        hubCounts.set(p.entity_b, (hubCounts.get(p.entity_b) ?? 0) + 1);
      }
      const hubs = [...hubCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8);

      return hubs.map(([hub, count]) => {
        const hubPairs = syncPairs
          .filter(p => p.entity_a === hub || p.entity_b === hub)
          .sort((a, b) => b.phase_alignment - a.phase_alignment);

        return {
          title: `${hub} (${count} connections)`,
          config: {
            breakdown: {
              data: hubPairs.map(p => ({
                label: p.entity_a === hub ? p.entity_b : p.entity_a,
                value: Math.round(p.phase_alignment * 100),
              })),
              label: `Sync strength (%)`,
              chartType: 'horizontal-bar',
            },
          } as SectionConfig,
        };
      });
    })(),
    category: 'metrics',
  };

  // ═══ Downstream Section ═══
  const productivePairs = result.downstream.pairs.filter(p => p.downstream_ratio > 0.5);
  const closedLoops = result.downstream.pairs.filter(p => p.downstream_ratio === 0 && p.total_interactions >= 3);
  const downstreamConfig: SectionConfig = {
    accent: 'indigo',
    kpi: {
      value: `${productivePairs.length}`,
      label: 'Productive Partnerships',
      delta: `${closedLoops.length} closed loops detected`,
    },
    stats: [
      { label: 'Productive (>50%)', value: `${productivePairs.length}` },
      { label: 'Closed Loops', value: `${closedLoops.length}` },
      { label: 'Total Pairs', value: `${result.downstream.pairs.length}` },
    ],
    table: {
      headers: ['Node A', 'Node B', 'Downstream %', 'Shared Threads'],
      rows: result.downstream.pairs.slice(0, 25).map(p => [
        p.node_a,
        p.node_b,
        `${Math.round(p.downstream_ratio * 100)}%`,
        `${p.total_interactions}`,
      ]),
      label: 'Partnership Analysis',
    },
    category: 'metrics',
  };

  // ═══ Fan-out Section ═══
  const escalators = result.fanout.by_author.filter(f => f.slope > 0.01);
  const maturers = result.fanout.by_author.filter(f => f.slope < -0.01);
  const fanoutConfig: SectionConfig = {
    accent: 'lime',
    kpi: {
      value: `${escalators.length}`,
      label: 'Escalating (growing fan-out)',
      delta: `${maturers.length} maturing`,
    },
    breakdown: {
      data: result.fanout.by_author
        .filter(f => Math.abs(f.slope) > 0.005)
        .slice(0, 15)
        .map(f => ({
          label: f.entity,
          value: Math.round(f.slope * 1000) / 10,
          color: f.slope > 0 ? '#f59e0b' : '#10b981',
        })),
      label: 'Fan-out Slope (positive = escalating)',
      chartType: 'horizontal-bar',
    },
    category: 'metrics',
  };

  // ═══ Build the tree ═══
  const sections: Array<{ title: string; config: SectionConfig }> = [
    { title: 'Overview', config: overviewConfig },
    { title: 'Gradient Map', config: gradientMapConfig },
    ...quadrantSections,
    { title: 'Flow', config: flowConfig },
    { title: 'Entropy', config: entropyConfig },
    { title: 'Cadence Sync', config: cadenceConfig },
    { title: 'Downstream', config: downstreamConfig },
    { title: 'Fan-out', config: fanoutConfig },
  ];

  return smartTree(title, description, sections);
}
