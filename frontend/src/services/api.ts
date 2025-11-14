import axios, { AxiosInstance, AxiosError } from 'axios';
import type { AuthResponse, LoginRequest, RegisterRequest, User } from '../types';

// For unified deployment where frontend and backend are served from the same origin,
// use empty string to make axios use relative URLs. Otherwise use the configured URL.
const API_BASE_URL = import.meta.env.VITE_API_URL === '' 
  ? '' // Use relative URLs for same-origin deployment
  : import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // Generic HTTP methods for flexibility
  get(url: string, config?: any) {
    return this.client.get(url, config);
  }

  post(url: string, data?: any, config?: any) {
    return this.client.post(url, data, config);
  }

  patch(url: string, data?: any, config?: any) {
    return this.client.patch(url, data, config);
  }

  put(url: string, data?: any, config?: any) {
    return this.client.put(url, data, config);
  }

  delete(url: string, config?: any) {
    return this.client.delete(url, config);
  }

  // Auth endpoints
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const { data } = await this.client.post('/api/auth/login', credentials);
    return data;
  }

  async register(userData: RegisterRequest): Promise<AuthResponse> {
    const { data } = await this.client.post('/api/auth/register', userData);
    return data;
  }

  async getCurrentUser(): Promise<User> {
    const { data } = await this.client.get('/api/auth/me');
    return data;
  }

  async logout(): Promise<void> {
    await this.client.post('/api/auth/logout');
  }

  // ML endpoints
  async getModels() {
    const { data } = await this.client.get('/api/models');
    return data;
  }

  async trainModel(config: any) {
    const { data } = await this.client.post('/api/models/train', config);
    return data;
  }

  async predict(modelId: string, inputData: any) {
    const { data } = await this.client.post(`/api/models/${modelId}/predict`, inputData);
    return data;
  }

  // Conversation endpoints
  async getConversations() {
    const { data } = await this.client.get('/api/conversations');
    return data;
  }

  async createConversation(title: string) {
    const { data } = await this.client.post('/api/conversations', { title });
    return data;
  }

  async sendMessage(conversationId: string, message: string) {
    const { data } = await this.client.post(`/api/conversations/${conversationId}/messages`, {
      content: message,
    });
    return data;
  }

  // Admin endpoints
  async getAllUsers() {
    const { data } = await this.client.get('/api/admin/users');
    return data;
  }

  async updateUserRole(userId: string, role: 'user' | 'admin') {
    const { data } = await this.client.patch(`/api/admin/users/${userId}`, { role });
    return data;
  }

  async deleteUser(userId: string) {
    await this.client.delete(`/api/admin/users/${userId}`);
  }

  async getSystemMetrics() {
    const { data } = await this.client.get('/api/admin/metrics');
    return data;
  }

  // Corrections endpoints
  async createCorrection(correctionData: {
    execution_id: number;
    correction_type: string;
    corrected_output: any;
    correction_reason?: string;
    operator_confidence?: number;
  }) {
    const { data } = await this.client.post('/api/v1/corrections', correctionData);
    return data;
  }

  async getCorrections(params?: {
    status_filter?: string;
    correction_type?: string;
    limit?: number;
  }) {
    const { data } = await this.client.get('/api/v1/corrections', { params });
    return data;
  }

  async getCorrection(correctionId: number) {
    const { data } = await this.client.get(`/api/v1/corrections/${correctionId}`);
    return data;
  }

  async reviewCorrection(correctionId: number, reviewData: {
    status: 'approved' | 'rejected';
    review_notes?: string;
  }) {
    const { data } = await this.client.patch(`/api/v1/corrections/${correctionId}/review`, reviewData);
    return data;
  }

  async getPendingCorrectionsCount() {
    const { data } = await this.client.get('/api/v1/corrections/pending/count');
    return data;
  }

  // Certification endpoints
  async certifyFormula(certificationData: {
    formula_id: number;
    to_tier: number;
    certification_notes?: string;
    test_accuracy?: any;
    validation_metrics?: any;
    review_period_days?: number;
  }) {
    const { data } = await this.client.post('/api/v1/certifications', certificationData);
    return data;
  }

  async getCertifications(params?: {
    formula_id?: number;
    tier?: number;
    limit?: number;
  }) {
    const { data } = await this.client.get('/api/v1/certifications', { params });
    return data;
  }

  async getFormulaCertificationHistory(formulaId: number) {
    const { data } = await this.client.get(`/api/v1/formulas/${formulaId}/certification-history`);
    return data;
  }

  // Auditor endpoints
  async getAuditLogs(params?: {
    action?: string;
    entity_type?: string;
    user_id?: number;
    days?: number;
    limit?: number;
  }) {
    const { data } = await this.client.get('/api/v1/auditor/audit-logs', { params });
    return data;
  }

  async getExecutionTrail(executionId: number) {
    const { data } = await this.client.get(`/api/v1/auditor/execution-trail/${executionId}`);
    return data;
  }

  async getAuditorDashboardStats(days: number = 30) {
    const { data } = await this.client.get('/api/v1/auditor/dashboard/stats', {
      params: { days }
    });
    return data;
  }

  async getCorrectionsTimeline(days: number = 30) {
    const { data } = await this.client.get('/api/v1/auditor/corrections/timeline', {
      params: { days }
    });
    return data;
  }

  async getFormulaTierDistribution() {
    const { data } = await this.client.get('/api/v1/auditor/formulas/tier-distribution');
    return data;
  }

  // Formula execution endpoints
  async executeFormula(formulaData: {
    formula_id: string | number;
    input_values: any;
    context_data?: any;
  }) {
    const { data } = await this.client.post('/api/v1/formulas/execute', formulaData);
    return data;
  }

  async getFormulas(params?: {
    domain?: string;
    status?: string;
    tier?: number;
    limit?: number;
  }) {
    const { data } = await this.client.get('/api/v1/formulas', { params });
    return data;
  }

  async getFormula(formulaId: string) {
    const { data } = await this.client.get(`/api/v1/formulas/${formulaId}`);
    return data;
  }
}

export const apiService = new ApiService();
export default apiService;
