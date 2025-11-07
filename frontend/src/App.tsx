import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { Login } from './pages/Login';
import { DashboardEnhanced as Dashboard } from './pages/DashboardEnhanced';
import { AdminPanel } from './pages/AdminPanel';
import AuditorDashboard from './pages/AuditorDashboard';
import CertificationPanel from './components/CertificationPanel';
import FormulaExecution from './pages/FormulaExecution';
import FormulaCatalog from './pages/FormulaCatalog';

const App: React.FC = () => {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/formulas"
          element={
            <ProtectedRoute>
              <FormulaExecution />
            </ProtectedRoute>
          }
        />
        <Route
          path="/catalog"
          element={
            <ProtectedRoute>
              <FormulaCatalog />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin"
          element={
            <ProtectedRoute requireAdmin>
              <AdminPanel />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin/certifications"
          element={
            <ProtectedRoute requireAdmin>
              <CertificationPanel />
            </ProtectedRoute>
          }
        />
        <Route
          path="/auditor"
          element={
            <ProtectedRoute requireAuditor>
              <AuditorDashboard />
            </ProtectedRoute>
          }
        />
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </AuthProvider>
  );
};

export default App;
