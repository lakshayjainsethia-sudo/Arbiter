import { apiClient } from './client';
import type {
  HealthResponse,
  ModelMetadata,
  MetricsResponse,
  RobustnessResponse,
  GlobalExplainabilityResponse,
  LocalExampleResponse,
  DynamicLocalExplainResponse,
  LoanApplicationInput,
  LoanPredictionResponse,
  IndianLoanApplicationInput,
  IndianLoanPredictionResponse,
  IndianModelMetadata,
} from '../types/api';

export const arbiterApi = {
  getHealth: async (): Promise<HealthResponse> => {
    const { data } = await apiClient.get<HealthResponse>('/health');
    return data;
  },

  getMetadata: async (): Promise<ModelMetadata> => {
    const { data } = await apiClient.get<ModelMetadata>('/metadata');
    return data;
  },

  getMetrics: async (): Promise<MetricsResponse> => {
    const { data } = await apiClient.get<MetricsResponse>('/metrics');
    return data;
  },

  getRobustness: async (): Promise<RobustnessResponse> => {
    const { data } = await apiClient.get<RobustnessResponse>('/robustness');
    return data;
  },

  getGlobalExplainability: async (): Promise<GlobalExplainabilityResponse> => {
    const { data } = await apiClient.get<GlobalExplainabilityResponse>(
      '/explainability/global'
    );
    return data;
  },

  getLocalExampleExplainability: async (): Promise<LocalExampleResponse> => {
    const { data } = await apiClient.get<LocalExampleResponse>(
      '/explainability/local/example'
    );
    return data;
  },

  predictLoan: async (
    input: LoanApplicationInput
  ): Promise<LoanPredictionResponse> => {
    const { data } = await apiClient.post<LoanPredictionResponse>(
      '/predict/loan',
      input
    );
    return data;
  },

  explainLocal: async (
    input: LoanApplicationInput
  ): Promise<DynamicLocalExplainResponse> => {
    const { data } = await apiClient.post<DynamicLocalExplainResponse>(
      '/explainability/local',
      input
    );
    return data;
  },

  // ── Indian Loan Research endpoints [EXPERIMENTAL] ──────────
  predictIndianLoan: async (
    input: IndianLoanApplicationInput
  ): Promise<IndianLoanPredictionResponse> => {
    const { data } = await apiClient.post<IndianLoanPredictionResponse>(
      '/predict/indian-loan',
      input
    );
    return data;
  },

  getIndianLoanMetadata: async (): Promise<IndianModelMetadata> => {
    const { data } = await apiClient.get<IndianModelMetadata>(
      '/predict/indian-loan/metadata'
    );
    return data;
  },
};
