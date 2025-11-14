import React, { createContext, useContext, useState, ReactNode } from 'react';
import type { User, LoginRequest, RegisterRequest } from '../types';

interface AuthContextType {
  user: User;
  loading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
  isAdmin: boolean;
  isAuditor: boolean;
  isOperator: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  // No authentication - create a default guest user
  const [user] = useState<User>({
    id: 'guest',
    email: 'guest@app.local',
    username: 'guest',
    role: 'admin', // Give guest admin access for full functionality
    createdAt: new Date().toISOString(),
  });
  const [loading] = useState(false);

  const login = async (_credentials: LoginRequest) => {
    // No-op - authentication disabled
    console.log('Authentication is disabled');
  };

  const register = async (_userData: RegisterRequest) => {
    // No-op - authentication disabled
    console.log('Authentication is disabled');
  };

  const logout = async () => {
    // No-op - authentication disabled
    console.log('Authentication is disabled');
  };

  const value: AuthContextType = {
    user,
    loading,
    login,
    register,
    logout,
    isAuthenticated: true, // Always authenticated as guest
    isAdmin: true, // Guest has admin access
    isAuditor: true, // Guest has auditor access
    isOperator: true, // Guest has operator access
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
