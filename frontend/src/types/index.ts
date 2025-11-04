export interface User {
  id: string;
  email: string;
  username: string;
  role: 'user' | 'admin';
  createdAt: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  tokens: AuthTokens;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: 'classification' | 'regression' | 'nlp' | 'vision';
  status: 'active' | 'training' | 'inactive';
  accuracy?: number;
  createdAt: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  conversations: Conversation[];
  createdAt: string;
  updatedAt: string;
  color?: string;
}

export interface FileAttachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url?: string;
}
