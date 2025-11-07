export type UserRole = 'operator' | 'admin' | 'auditor' | 'system';

export interface User {
  id: string;
  email: string;
  username: string;
  role: UserRole;
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

// Formula types
export interface Formula {
  id: string;
  name: string;
  domain: string;
  equation: string;
  description?: string;
  tier: 1 | 2 | 3 | 4;
  confidence_score: number;
  execution_count: number;
  success_rate: number;
  status: 'active' | 'testing' | 'deprecated';
  input_parameters: FormulaParameter[];
  output_parameters: FormulaParameter[];
  validation_stages: ValidationStage[];
  created_at: string;
  updated_at: string;
  tags?: string[];
}

export interface FormulaParameter {
  name: string;
  unit: string;
  type: 'float' | 'int' | 'string' | 'boolean';
  required: boolean;
  default_value?: any;
  description?: string;
  min_value?: number;
  max_value?: number;
}

export interface ValidationStage {
  stage: string;
  status: 'passed' | 'failed' | 'pending';
  message?: string;
}

export interface FormulaExecutionRequest {
  formula_id: string;
  input_values: Record<string, any>;
  context_data?: Record<string, any>;
}

export interface FormulaExecutionResult {
  execution_id: string;
  formula_id: string;
  status: 'success' | 'failed' | 'warning';
  output: any;
  execution_time_ms: number;
  validation_results: ValidationStage[];
  confidence_score?: number;
  timestamp: string;
}
