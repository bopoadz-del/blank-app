import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { DashboardEnhanced as Dashboard } from './pages/DashboardEnhanced';
import { AdminPanel } from './pages/AdminPanel';
import AuditorDashboard from './pages/AuditorDashboard';
import CertificationPanel from './components/CertificationPanel';
import FormulaExecution from './pages/FormulaExecution';
import FormulaCatalog from './pages/FormulaCatalog';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/formulas" element={<FormulaExecution />} />
      <Route path="/catalog" element={<FormulaCatalog />} />
      <Route path="/admin" element={<AdminPanel />} />
      <Route path="/admin/certifications" element={<CertificationPanel />} />
      <Route path="/auditor" element={<AuditorDashboard />} />
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
};

export default App;
