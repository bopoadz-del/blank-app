import axios, { AxiosInstance, AxiosError } from 'axios';
import type { AuthResponse, LoginRequest, RegisterRequest, User } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('accessToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor to handle token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && originalRequest) {
          const refreshToken = localStorage.getItem('refreshToken');

          if (refreshToken) {
            try {
              const { data } = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
                refreshToken,
              });

              localStorage.setItem('accessToken', data.accessToken);

              if (originalRequest.headers) {
                originalRequest.headers.Authorization = `Bearer ${data.accessToken}`;
              }

              return this.client(originalRequest);
            } catch (refreshError) {
              localStorage.removeItem('accessToken');
              localStorage.removeItem('refreshToken');
              window.location.href = '/login';
            }
          }
        }

        return Promise.reject(error);
      }
    );
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
}

export const apiService = new ApiService();
