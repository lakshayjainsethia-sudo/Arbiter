import type { DecisionCategory } from '../types/api';

/** Fixed conversion baseline: 1 USD = 85.00 INR */
export const USD_TO_INR_RATE = 85.0;

export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatDec(value: number | null | undefined, decimals = 4): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';
  return value.toFixed(decimals);
}

/** Format currency in USD or INR with localized thousands/lakhs separators */
export function formatCurrency(
  value: number | null | undefined,
  currency: 'USD' | 'INR' = 'USD'
): string {
  if (value === null || value === undefined || isNaN(value)) return 'N/A';

  if (currency === 'INR') {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(value);
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value);
}

/** Direct INR formatting using en-IN locale */
export function formatDirectINR(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return '₹0';
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(value);
}

/** Formats INR value into Lakhs (L) or Crores (Cr) */
export function formatIndianLakhsCrores(inrValue: number | null | undefined): string {
  if (inrValue === null || inrValue === undefined || isNaN(inrValue)) return '₹0';

  const abs = Math.abs(inrValue);
  const sign = inrValue < 0 ? '-' : '';

  if (abs >= 10000000) {
    return `${sign}₹${(abs / 10000000).toFixed(2)} Cr`;
  } else if (abs >= 100000) {
    return `${sign}₹${(abs / 100000).toFixed(2)} Lakh`;
  } else if (abs >= 1000) {
    return `${sign}₹${(abs / 1000).toFixed(1)}k`;
  }
  return `${sign}₹${abs.toLocaleString('en-IN')}`;
}

/** Dual currency formatter: "$65,000 (₹55.25 Lakh)" */
export function formatDualCurrency(usdValue: number | null | undefined): string {
  if (usdValue === null || usdValue === undefined || isNaN(usdValue)) return 'N/A';
  const inrValue = usdValue * USD_TO_INR_RATE;
  const usdStr = formatCurrency(usdValue, 'USD');
  const inrStr = formatIndianLakhsCrores(inrValue);
  return `${usdStr} (${inrStr})`;
}

/** Formats date into DD/MM/YYYY (Indian standard) */
export function formatDateIST(date: Date | string | number = new Date()): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'N/A';

  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const year = d.getFullYear();
  return `${day}/${month}/${year}`;
}

/** Formats timestamp into DD/MM/YYYY, HH:mm:ss IST */
export function formatTimestampIST(date: Date | string | number = new Date()): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'N/A';

  const dateStr = formatDateIST(d);
  const timeStr = d.toLocaleTimeString('en-IN', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
  return `${dateStr}, ${timeStr} IST`;
}

/** Complete list of 28 States and 8 Union Territories in India */
export const INDIAN_STATES_AND_UTS: Array<{ code: string; name: string; type: 'State' | 'UT' }> = [
  // 28 States
  { code: 'AP', name: 'Andhra Pradesh', type: 'State' },
  { code: 'AR', name: 'Arunachal Pradesh', type: 'State' },
  { code: 'AS', name: 'Assam', type: 'State' },
  { code: 'BR', name: 'Bihar', type: 'State' },
  { code: 'CG', name: 'Chhattisgarh', type: 'State' },
  { code: 'GA', name: 'Goa', type: 'State' },
  { code: 'GJ', name: 'Gujarat', type: 'State' },
  { code: 'HR', name: 'Haryana', type: 'State' },
  { code: 'HP', name: 'Himachal Pradesh', type: 'State' },
  { code: 'JH', name: 'Jharkhand', type: 'State' },
  { code: 'KA', name: 'Karnataka', type: 'State' },
  { code: 'KL', name: 'Kerala', type: 'State' },
  { code: 'MP', name: 'Madhya Pradesh', type: 'State' },
  { code: 'MH', name: 'Maharashtra', type: 'State' },
  { code: 'MN', name: 'Manipur', type: 'State' },
  { code: 'ML', name: 'Meghalaya', type: 'State' },
  { code: 'MZ', name: 'Mizoram', type: 'State' },
  { code: 'NL', name: 'Nagaland', type: 'State' },
  { code: 'OD', name: 'Odisha', type: 'State' },
  { code: 'PB', name: 'Punjab', type: 'State' },
  { code: 'RJ', name: 'Rajasthan', type: 'State' },
  { code: 'SK', name: 'Sikkim', type: 'State' },
  { code: 'TN', name: 'Tamil Nadu', type: 'State' },
  { code: 'TS', name: 'Telangana', type: 'State' },
  { code: 'TR', name: 'Tripura', type: 'State' },
  { code: 'UP', name: 'Uttar Pradesh', type: 'State' },
  { code: 'UK', name: 'Uttarakhand', type: 'State' },
  { code: 'WB', name: 'West Bengal', type: 'State' },
  // 8 Union Territories
  { code: 'AN', name: 'Andaman & Nicobar Islands', type: 'UT' },
  { code: 'CH', name: 'Chandigarh', type: 'UT' },
  { code: 'DN', name: 'Dadra & Nagar Haveli and Daman & Diu', type: 'UT' },
  { code: 'DL', name: 'Delhi (NCT)', type: 'UT' },
  { code: 'JK', name: 'Jammu & Kashmir', type: 'UT' },
  { code: 'LA', name: 'Ladakh', type: 'UT' },
  { code: 'LD', name: 'Lakshadweep', type: 'UT' },
  { code: 'PY', name: 'Puducherry', type: 'UT' },
];

export interface DecisionMeta {
  label: string;
  badgeBg: string;
  badgeText: string;
  badgeBorder: string;
  dotColor: string;
  statusClass: string;
  description: string;
}

export function getDecisionMeta(decision: DecisionCategory): DecisionMeta {
  switch (decision) {
    case 'lower_predicted_default_risk':
      return {
        label: 'Lower Predicted Default Risk',
        badgeBg: 'bg-emerald-50',
        badgeText: 'text-emerald-700',
        badgeBorder: 'border-emerald-200',
        dotColor: 'bg-emerald-500',
        statusClass: 'emerald',
        description: 'Estimated default probability is below the manual review band (< 45.0%).',
      };
    case 'manual_review':
      return {
        label: 'Manual Review Required',
        badgeBg: 'bg-amber-50',
        badgeText: 'text-amber-800',
        badgeBorder: 'border-amber-200',
        dotColor: 'bg-amber-500',
        statusClass: 'amber',
        description:
          'Probability falls within the designated uncertainty band [45.0% – 55.0%]. Routing to human underwriter for review.',
      };
    case 'higher_predicted_default_risk':
      return {
        label: 'Higher Predicted Default Risk',
        badgeBg: 'bg-rose-50',
        badgeText: 'text-rose-700',
        badgeBorder: 'border-rose-200',
        dotColor: 'bg-rose-500',
        statusClass: 'rose',
        description:
          'Estimated default probability exceeds the upper review threshold (> 55.0%). Research model flags elevated risk.',
      };
    default:
      return {
        label: String(decision).replace(/_/g, ' '),
        badgeBg: 'bg-slate-100',
        badgeText: 'text-slate-700',
        badgeBorder: 'border-slate-300',
        dotColor: 'bg-slate-500',
        statusClass: 'slate',
        description: 'Uncategorized risk classification.',
      };
  }
}
