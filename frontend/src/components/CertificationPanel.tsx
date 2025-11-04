import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Shield, TrendingUp, Clock, AlertTriangle, Check, X, Award } from 'lucide-react';
import { apiService } from '../services/api';
import { toast } from 'sonner';

interface Formula {
  id: number;
  formula_id: string;
  name: string;
  version: string;
  tier: number;
  is_locked: boolean;
  total_executions: number;
  successful_executions: number;
  confidence_score: number;
}

const CertificationPanel: React.FC = () => {
  const [formulas, setFormulas] = useState<Formula[]>([]);
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isCertifying, setIsCertifying] = useState(false);
  const [certificationNotes, setCertificationNotes] = useState('');
  const [certificationHistory, setCertificationHistory] = useState<any[]>([]);

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

  const fetchCertificationHistory = async (formulaId: number) => {
    try {
      const history = await apiService.getFormulaCertificationHistory(formulaId);
      setCertificationHistory(history);
    } catch (error) {
      console.error('Error loading certification history:', error);
    }
  };

  const handleSelectFormula = (formula: Formula) => {
    setSelectedFormula(formula);
    setCertificationNotes('');
    fetchCertificationHistory(formula.id);
  };

  const handleCertify = async () => {
    if (!selectedFormula) return;

    if (!certificationNotes.trim()) {
      toast.error('Please provide certification notes');
      return;
    }

    setIsCertifying(true);
    try {
      const targetTier = selectedFormula.tier - 1;

      await apiService.certifyFormula({
        formula_id: selectedFormula.id,
        to_tier: targetTier,
        certification_notes: certificationNotes,
        test_accuracy: {
          success_rate: (selectedFormula.successful_executions / selectedFormula.total_executions * 100).toFixed(2)
        },
        validation_metrics: {
          confidence_score: selectedFormula.confidence_score,
          total_executions: selectedFormula.total_executions
        },
        review_period_days: 7
      });

      toast.success(`Formula certified to Tier ${targetTier}!`);
      fetchFormulas();
      setSelectedFormula(null);
      setCertificationNotes('');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to certify formula');
    } finally {
      setIsCertifying(false);
    }
  };

  const getTierInfo = (tier: number) => {
    const tiers = {
      1: { label: 'Tier 1: Certified', icon: Shield, color: 'text-green-600', bg: 'bg-green-100', badge: 'bg-green-500' },
      2: { label: 'Tier 2: Validated', icon: TrendingUp, color: 'text-blue-600', bg: 'bg-blue-100', badge: 'bg-blue-500' },
      3: { label: 'Tier 3: Testing', icon: Clock, color: 'text-yellow-600', bg: 'bg-yellow-100', badge: 'bg-yellow-500' },
      4: { label: 'Tier 4: Experimental', icon: AlertTriangle, color: 'text-red-600', bg: 'bg-red-100', badge: 'bg-red-500' }
    };
    return tiers[tier as keyof typeof tiers] || tiers[4];
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Award className="h-6 w-6 text-purple-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Formula Certification</h2>
              <p className="text-gray-600 mt-1">
                Promote formulas through the tier system (Tier 4 → Tier 1)
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
          {/* Formula List */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Available Formulas</h3>
            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {formulas.filter(f => !f.is_locked && f.tier > 1).map((formula) => {
                const tierInfo = getTierInfo(formula.tier);
                const TierIcon = tierInfo.icon;
                const successRate = formula.total_executions > 0
                  ? (formula.successful_executions / formula.total_executions * 100).toFixed(1)
                  : '0';

                return (
                  <motion.div
                    key={formula.id}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => handleSelectFormula(formula)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedFormula?.id === formula.id
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 hover:border-purple-300'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h4 className="font-semibold text-gray-900">{formula.name}</h4>
                          <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${tierInfo.bg} ${tierInfo.color}`}>
                            <TierIcon className="h-3 w-3" />
                            Tier {formula.tier}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 font-mono mb-2">{formula.formula_id}</p>
                        <div className="flex items-center gap-4 text-xs text-gray-600">
                          <span>Version: {formula.version}</span>
                          <span>Executions: {formula.total_executions}</span>
                          <span>Success: {successRate}%</span>
                          <span>Confidence: {(formula.confidence_score * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}

              {formulas.filter(f => !f.is_locked && f.tier > 1).length === 0 && (
                <div className="text-center py-12 text-gray-500">
                  <Shield className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>No formulas available for certification</p>
                </div>
              )}
            </div>
          </div>

          {/* Certification Form */}
          <div>
            {selectedFormula ? (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-gray-900">Certify Formula</h3>

                {/* Current Formula Info */}
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="mb-3">
                    <h4 className="font-semibold text-gray-900">{selectedFormula.name}</h4>
                    <p className="text-sm text-gray-600 font-mono">{selectedFormula.formula_id}</p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">Current Tier:</span>
                      <span className="ml-2 font-semibold">Tier {selectedFormula.tier}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Target Tier:</span>
                      <span className="ml-2 font-semibold text-green-600">Tier {selectedFormula.tier - 1}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Executions:</span>
                      <span className="ml-2 font-semibold">{selectedFormula.total_executions}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Success Rate:</span>
                      <span className="ml-2 font-semibold">
                        {selectedFormula.total_executions > 0
                          ? ((selectedFormula.successful_executions / selectedFormula.total_executions) * 100).toFixed(1)
                          : '0'}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Certification Notes */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Certification Notes <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    value={certificationNotes}
                    onChange={(e) => setCertificationNotes(e.target.value)}
                    className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    placeholder="Explain why this formula is ready for promotion..."
                    required
                  />
                </div>

                {/* Warning */}
                {selectedFormula.tier === 2 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                    <div className="text-sm text-yellow-800">
                      <strong>Warning:</strong> Promoting to Tier 1 will lock this formula and make it immutable.
                      This action cannot be reversed.
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={() => setSelectedFormula(null)}
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors flex items-center justify-center gap-2"
                  >
                    <X className="h-4 w-4" />
                    Cancel
                  </button>
                  <button
                    onClick={handleCertify}
                    disabled={isCertifying || !certificationNotes.trim()}
                    className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isCertifying ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                        Certifying...
                      </>
                    ) : (
                      <>
                        <Check className="h-4 w-4" />
                        Certify to Tier {selectedFormula.tier - 1}
                      </>
                    )}
                  </button>
                </div>

                {/* Certification History */}
                {certificationHistory.length > 0 && (
                  <div className="mt-6">
                    <h4 className="text-sm font-semibold text-gray-900 mb-3">Certification History</h4>
                    <div className="space-y-2">
                      {certificationHistory.map((cert: any) => (
                        <div key={cert.id} className="bg-white border border-gray-200 rounded-lg p-3 text-sm">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-medium">
                              Tier {cert.from_tier} → Tier {cert.to_tier}
                            </span>
                            <span className="text-gray-500">
                              {new Date(cert.certified_at).toLocaleDateString()}
                            </span>
                          </div>
                          <p className="text-gray-600">{cert.certification_notes}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <Award className="h-16 w-16 mx-auto mb-3 text-gray-300" />
                  <p>Select a formula to certify</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CertificationPanel;
