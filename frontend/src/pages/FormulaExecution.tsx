import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiService } from '../services/api';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { Play, ArrowLeft, Cpu, TrendingUp } from 'lucide-react';
import ExecutionResult from '../components/ExecutionResult';
import CorrectionModal from '../components/CorrectionModal';

interface Formula {
  id: number;
  formula_id: string;
  name: string;
  description?: string;
  version: string;
  tier: number;
  is_locked: boolean;
  input_schema?: any;
}

interface FormulaExecution {
  id: number;
  execution_id: string;
  formula_id: number;
  formula_name?: string;
  tier?: number;
  input_values: any;
  output_values: any;
  status: string;
  execution_time?: number;
  execution_timestamp: string;
  created_at?: string;
  execution_time_ms?: number;
}

const FormulaExecution: React.FC = () => {
  const navigate = useNavigate();
  const [formulas, setFormulas] = useState<Formula[]>([]);
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);
  const [inputValues, setInputValues] = useState<string>('{}');
  const [isExecuting, setIsExecuting] = useState(false);
  const [execution, setExecution] = useState<FormulaExecution | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showCorrectionModal, setShowCorrectionModal] = useState(false);

  // Track current user and derive isOperator; attempt localStorage first then apiService
  const [currentUser, setCurrentUser] = useState<{ id?: number; username?: string; role?: string } | null>(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('user');
      if (raw) {
        setCurrentUser(JSON.parse(raw));
      } else if (typeof (apiService as any).getCurrentUser === 'function') {
        (apiService as any).getCurrentUser().then((u: any) => setCurrentUser(u)).catch(() => {});
      } else if ((window as any).CURRENT_USER) {
        setCurrentUser((window as any).CURRENT_USER);
      }
    } catch (e) {
      // ignore parse errors
      setCurrentUser(null);
    }
  }, []);

  const user = currentUser;
  const isOperator = user?.role === 'operator';

  useEffect(() => {
    fetchFormulas();
  }, []);

  const fetchFormulas = async () => {
    setIsLoading(true);
    try {
      const data = await apiService.getFormulas({ limit: 100 });
      setFormulas(data);
    } catch (error) {
      toast.error('Failed to load formulas');
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecuteFormula = async () => {
    if (!selectedFormula) {
      toast.error('Please select a formula');
      return;
    }

    // Prevent operators from executing anything other than Tier 1
    if (isOperator && selectedFormula.tier !== 1) {
      toast.error('Operators can only execute Tier 1 certified formulas');
      return;
    }

    try {
      const parsedInputs = JSON.parse(inputValues);
      setIsExecuting(true);

      const result = await apiService.executeFormula({
        formula_id: selectedFormula.id,
        input_values: parsedInputs
      });

      setExecution(result);
      toast.success('Formula executed successfully!');
    } catch (error: any) {
      if (error instanceof SyntaxError) {
        toast.error('Invalid JSON input');
      } else {
        toast.error(error.response?.data?.detail || 'Failed to execute formula');
      }
    } finally {
      setIsExecuting(false);
    }
  };

  const handleSelectFormula = (formula: Formula) => {
    setSelectedFormula(formula);
    setExecution(null);

    // Set default input values based on schema if available
    if (formula.input_schema) {
      setInputValues(JSON.stringify(formula.input_schema, null, 2));
    } else {
      setInputValues(`{
  "example_input": "value"
}`);
    }
  };

  const getTierBadge = (tier: number) => {
    const badges = {
      1: { label: 'Tier 1: Certified', color: 'bg-green-500' },
      2: { label: 'Tier 2: Validated', color: 'bg-blue-500' },
      3: { label: 'Tier 3: Testing', color: 'bg-yellow-500' },
      4: { label: 'Tier 4: Experimental', color: 'bg-red-500' }
    };
    return badges[tier as keyof typeof badges] || badges[4];
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/dashboard')}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-gray-900">Formula Execution</h1>
              <p className="text-sm text-gray-500">Execute AI formulas and review results</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Formula Selection */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5" />
                Available Formulas
              </h2>

              <div className="space-y-2 max-h-[600px] overflow-y-auto">
                {formulas.length === 0 ? (
                  <p className="text-center text-gray-500 py-8">No formulas available</p>
                ) : (
                  formulas.map((formula) => {
                    const badge = getTierBadge(formula.tier);

                    return (
                      <motion.div
                        key={formula.id}
                        whileHover={{ scale: 1.02 }}
                        onClick={() => handleSelectFormula(formula)}
                        className={`p-3 rounded-lg border-2 transition-all ${
                          selectedFormula?.id === formula.id
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-200 hover:border-blue-300 cursor-pointer'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <h3 className="font-semibold text-gray-900 text-sm">{formula.name}</h3>
                          <span className={`text-xs px-2 py-0.5 rounded-full text-white ${badge.color}`}>T{formula.tier}</span>
                        </div>
                        <p className="text-xs text-gray-600 font-mono truncate">{formula.formula_id}</p>
                        {formula.description && (
                          <p className="text-xs text-gray-500 mt-1 line-clamp-2">{formula.description}</p>
                        )}
                      </motion.div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Stats */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <TrendingUp className="w-4 h-4 text-blue-600" />
                  <span className="text-xs text-gray-600">Total Formulas</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">{formulas.length}</p>
              </div>
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-3">
                <div className="flex items-center gap-2 mb-1">
                  <Cpu className="w-4 h-4 text-green-600" />
                  <span className="text-xs text-gray-600">Tier 1</span>
                </div>
                <p className="text-2xl font-bold text-gray-900">{formulas.filter(f => f.tier === 1).length}</p>
              </div>
            </div>
          </div>

          {/* Right: Execution & Results */}
          <div className="lg:col-span-2">
            {selectedFormula ? (
              <div className="space-y-6">
                {/* Execution Form */}
                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">
                    Execute: {selectedFormula.name}
                  </h2>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Input Values (JSON)
                    </label>
                    <textarea
                      value={inputValues}
                      onChange={(e) => setInputValues(e.target.value)}
                      className="w-full h-48 px-3 py-2 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder='{"input_key": "value"}'
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Enter JSON object with input values for the formula
                    </p>
                  </div>

                  <button
                    onClick={handleExecuteFormula}
                    disabled={isExecuting}
                    className="w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center">
                    {isExecuting ? (
                      <>
                        <Cpu className="animate-spin mr-2 h-5 w-5" />
                        Executing...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-5 w-5" />
                        Execute Formula
                      </>
                    )}
                  </button>
                </div>

                {/* Results Section */}
                {execution && (
                  <div className="mt-6">
                    <ExecutionResult
                      execution={execution}
                      userRole={currentUser?.role || 'guest'}
                      onCorrectionsUpdated={() => {
                        // Refresh or handle corrections update
                      }}
                    />
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* Correction Modal */}
      {showCorrectionModal && execution && (
        <CorrectionModal
          isOpen={showCorrectionModal}
          executionId={execution.id}
          originalOutput={execution.output_values}
          onClose={() => setShowCorrectionModal(false)}
          onCorrectionSubmitted={() => {
            setShowCorrectionModal(false);
            toast.success('Correction submitted successfully');
          }}
        />
      )}
    </div>
  );
};

export default FormulaExecution;
