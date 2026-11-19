import React, { createContext, useContext, useState, useEffect } from 'react';
import {
  USD_TO_INR_RATE,
  formatCurrency,
  formatIndianLakhsCrores,
  formatDualCurrency,
} from '../utils/formatters';

export type CurrencyMode = 'USD' | 'INR';

interface CurrencyContextValue {
  currency: CurrencyMode;
  setCurrency: (mode: CurrencyMode) => void;
  toggleCurrency: () => void;
  exchangeRate: number;
  formatAmount: (usdValue: number | null | undefined, options?: { dual?: boolean }) => string;
  formatLakhs: (inrValue: number | null | undefined) => string;
}

const CurrencyContext = createContext<CurrencyContextValue | undefined>(undefined);

export const CurrencyProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currency, setCurrencyState] = useState<CurrencyMode>(() => {
    const saved = localStorage.getItem('arbiter_currency');
    return saved === 'INR' ? 'INR' : 'USD';
  });

  useEffect(() => {
    localStorage.setItem('arbiter_currency', currency);
  }, [currency]);

  const setCurrency = (mode: CurrencyMode) => setCurrencyState(mode);

  const toggleCurrency = () => {
    setCurrencyState((prev) => (prev === 'USD' ? 'INR' : 'USD'));
  };

  const formatAmount = (
    usdValue: number | null | undefined,
    options?: { dual?: boolean }
  ): string => {
    if (usdValue === null || usdValue === undefined || isNaN(usdValue)) return 'N/A';

    if (options?.dual) {
      return formatDualCurrency(usdValue);
    }

    if (currency === 'INR') {
      const inrValue = usdValue * USD_TO_INR_RATE;
      return formatIndianLakhsCrores(inrValue);
    }

    return formatCurrency(usdValue, 'USD');
  };

  const formatLakhs = (inrValue: number | null | undefined): string => {
    return formatIndianLakhsCrores(inrValue);
  };

  return (
    <CurrencyContext.Provider
      value={{
        currency,
        setCurrency,
        toggleCurrency,
        exchangeRate: USD_TO_INR_RATE,
        formatAmount,
        formatLakhs,
      }}
    >
      {children}
    </CurrencyContext.Provider>
  );
};

export function useCurrency(): CurrencyContextValue {
  const context = useContext(CurrencyContext);
  if (!context) {
    throw new Error('useCurrency must be used within a CurrencyProvider');
  }
  return context;
}
