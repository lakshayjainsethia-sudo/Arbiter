import React from 'react';

interface ProbabilityGaugeProps {
  probability: number; // 0.0 to 1.0
  reviewBand?: [number, number]; // e.g. [0.45, 0.55]
  baseThreshold?: number; // e.g. 0.50
}

export const ProbabilityGauge: React.FC<ProbabilityGaugeProps> = ({
  probability,
  reviewBand = [0.45, 0.55],
  baseThreshold = 0.50,
}) => {
  const pct = Math.min(Math.max(probability * 100, 0), 100);
  const [lowerBand, upperBand] = reviewBand;
  const lowerPct = lowerBand * 100;
  const upperPct = upperBand * 100;

  // Determine status color for needle and text
  let statusColor = 'text-emerald-600';
  let pinBg = 'bg-emerald-600';
  let statusLabel = 'Lower Predicted Default Risk';

  if (probability >= lowerBand && probability <= upperBand) {
    statusColor = 'text-amber-600';
    pinBg = 'bg-amber-600';
    statusLabel = 'Manual Review Required (Uncertainty Zone)';
  } else if (probability > upperBand) {
    statusColor = 'text-rose-600';
    pinBg = 'bg-rose-600';
    statusLabel = 'Higher Predicted Default Risk';
  }

  return (
    <div className="w-full">
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Estimated Default Probability
        </span>
        <span className={`text-2xl font-extrabold tracking-tight ${statusColor}`}>
          {(probability * 100).toFixed(2)}%
        </span>
      </div>

      {/* Main Track with Segmented Zones */}
      <div className="relative h-4 w-full rounded-full bg-slate-100 overflow-hidden shadow-inner flex">
        {/* Low Risk Zone: 0% to lowerBand% */}
        <div
          style={{ width: `${lowerPct}%` }}
          className="h-full bg-emerald-400/80 transition-all"
          title={`Low Risk Zone (0% - ${lowerPct}%)`}
        />
        {/* Review Band Zone: lowerBand% to upperBand% */}
        <div
          style={{ width: `${upperPct - lowerPct}%` }}
          className="h-full bg-amber-400/90 transition-all border-x border-white/50"
          title={`Manual Review Band (${lowerPct}% - ${upperPct}%)`}
        />
        {/* High Risk Zone: upperBand% to 100% */}
        <div
          style={{ width: `${100 - upperPct}%` }}
          className="h-full bg-rose-400/80 transition-all"
          title={`High Risk Zone (${upperPct}% - 100%)`}
        />
      </div>

      {/* Needle indicator pointer */}
      <div className="relative w-full h-6 mt-1">
        <div
          className="absolute top-0 -translate-x-1/2 flex flex-col items-center transition-all duration-500"
          style={{ left: `${pct}%` }}
        >
          <div className={`w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-b-[6px] border-b-slate-900`} />
          <div className={`w-2.5 h-2.5 rounded-full ${pinBg} ring-2 ring-white shadow-xs`} />
        </div>

        {/* Threshold Markers */}
        <div
          className="absolute top-1 -translate-x-1/2 text-[10px] font-mono text-slate-400"
          style={{ left: `${lowerPct}%` }}
        >
          45%
        </div>
        <div
          className="absolute top-1 -translate-x-1/2 text-[10px] font-mono text-slate-700 font-bold"
          style={{ left: `${baseThreshold * 100}%` }}
        >
          50%
        </div>
        <div
          className="absolute top-1 -translate-x-1/2 text-[10px] font-mono text-slate-400"
          style={{ left: `${upperPct}%` }}
        >
          55%
        </div>
      </div>

      {/* Zone Legend */}
      <div className="mt-3 grid grid-cols-3 gap-2 pt-2 border-t border-slate-100 text-[11px]">
        <div className="flex items-center gap-1.5 text-slate-600">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          <span>Low Risk (&lt; 45%)</span>
        </div>
        <div className="flex items-center justify-center gap-1.5 text-amber-700 font-medium">
          <span className="h-2 w-2 rounded-full bg-amber-500" />
          <span>Review Band (45%–55%)</span>
        </div>
        <div className="flex items-center justify-end gap-1.5 text-slate-600">
          <span className="h-2 w-2 rounded-full bg-rose-500" />
          <span>High Risk (&gt; 55%)</span>
        </div>
      </div>

      <div className="mt-2 text-center text-xs text-slate-500">
        Status Assessment:{' '}
        <span className={`font-semibold ${statusColor}`}>{statusLabel}</span>
      </div>
    </div>
  );
};
