import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TierBadge } from './TierBadge';
import type { Formula } from '../types';

interface DeploymentWizardProps {
  formula: Formula | null;
  isOpen: boolean;
  onClose: () => void;
  onDeploy: (config: DeploymentConfig) => Promise<void>;
}

export interface DeploymentConfig {
  formula_id: string;
  deployment_type: 'edge' | 'cloud' | 'hybrid';
  target_environment: 'production' | 'staging' | 'testing';
  auto_scale: boolean;
  max_concurrent_executions: number;
  monitoring_enabled: boolean;
  input_values?: Record<string, any>;
  context_data?: Record<string, any>;
}

export const DeploymentWizard: React.FC<DeploymentWizardProps> = ({
  formula,
  isOpen,
  onClose,
  onDeploy
}) => {
  const [step, setStep] = useState(1);
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploymentSuccess, setDeploymentSuccess] = useState(false);

  const [config, setConfig] = useState<DeploymentConfig>({
    formula_id: formula?.id || '',
    deployment_type: 'cloud',
    target_environment: 'production',
    auto_scale: true,
    max_concurrent_executions: 10,
    monitoring_enabled: true,
    input_values: {},
    context_data: {}
  });

  const totalSteps = 3;

  const handleDeploy = async () => {
    if (!formula) return;

    setIsDeploying(true);
    try {
      await onDeploy({ ...config, formula_id: formula.id });
      setDeploymentSuccess(true);
      setTimeout(() => {
        onClose();
        resetWizard();
      }, 2000);
    } catch (error) {
      console.error('Deployment failed:', error);
    } finally {
      setIsDeploying(false);
    }
  };

  const resetWizard = () => {
    setStep(1);
    setDeploymentSuccess(false);
    setConfig({
      formula_id: formula?.id || '',
      deployment_type: 'cloud',
      target_environment: 'production',
      auto_scale: true,
      max_concurrent_executions: 10,
      monitoring_enabled: true,
      input_values: {},
      context_data: {}
    });
  };

  const handleClose = () => {
    onClose();
    setTimeout(resetWizard, 300);
  };

  if (!formula) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            className="fixed inset-0 bg-black bg-opacity-50 z-40 flex items-center justify-center p-4"
          >
            {/* Modal */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col"
            >
              {/* Header */}
              <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-indigo-50">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                      <span>⚡</span>
                      Deploy Formula
                    </h2>
                    <p className="text-sm text-gray-600 mt-1">
                      {formula.name}
                    </p>
                  </div>
                  <button
                    onClick={handleClose}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                {/* Progress Bar */}
                <div className="mt-4 flex items-center gap-2">
                  {[1, 2, 3].map((stepNum) => (
                    <div key={stepNum} className="flex-1">
                      <div className="flex items-center gap-2">
                        <div
                          className={`
                            w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold
                            ${step >= stepNum
                              ? 'bg-purple-500 text-white'
                              : 'bg-gray-200 text-gray-500'
                            }
                            transition-all duration-300
                          `}
                        >
                          {step > stepNum ? '✓' : stepNum}
                        </div>
                        {stepNum < totalSteps && (
                          <div
                            className={`
                              flex-1 h-1 rounded-full transition-all duration-300
                              ${step > stepNum ? 'bg-purple-500' : 'bg-gray-200'}
                            `}
                          />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Body */}
              <div className="flex-1 overflow-y-auto p-6">
                <AnimatePresence mode="wait">
                  {/* Step 1: Formula Overview */}
                  {step === 1 && (
                    <motion.div
                      key="step1"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="space-y-4"
                    >
                      <h3 className="text-lg font-semibold text-gray-900">Formula Overview</h3>

                      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Tier Level</span>
                          <TierBadge tier={formula.tier} size="sm" animated />
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Domain</span>
                          <span className="text-sm text-gray-900 capitalize">
                            {formula.domain.replace('_', ' ')}
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Confidence Score</span>
                          <span className="text-sm font-semibold text-purple-600">
                            {(formula.confidence_score * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Success Rate</span>
                          <span className="text-sm font-semibold text-green-600">
                            {(formula.success_rate * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700">Executions</span>
                          <span className="text-sm text-gray-900">
                            {formula.execution_count.toLocaleString()}
                          </span>
                        </div>
                      </div>

                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="text-sm font-semibold text-blue-900 mb-2">Equation</h4>
                        <code className="text-sm text-blue-800 font-mono break-all">
                          {formula.equation}
                        </code>
                      </div>

                      {formula.tier < 2 && (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
                          <span className="text-yellow-600 text-xl">⚠️</span>
                          <div className="flex-1">
                            <h4 className="text-sm font-semibold text-yellow-900">Tier 1 Warning</h4>
                            <p className="text-xs text-yellow-700 mt-1">
                              This formula is experimental and requires human supervision in production.
                            </p>
                          </div>
                        </div>
                      )}
                    </motion.div>
                  )}

                  {/* Step 2: Deployment Configuration */}
                  {step === 2 && (
                    <motion.div
                      key="step2"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="space-y-4"
                    >
                      <h3 className="text-lg font-semibold text-gray-900">Deployment Configuration</h3>

                      {/* Deployment Type */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Deployment Type
                        </label>
                        <div className="grid grid-cols-3 gap-3">
                          {[
                            { value: 'cloud', label: 'Cloud', icon: '☁️', desc: 'Hosted service' },
                            { value: 'edge', label: 'Edge', icon: '📡', desc: 'On-device' },
                            { value: 'hybrid', label: 'Hybrid', icon: '🌐', desc: 'Cloud + Edge' }
                          ].map((type) => (
                            <button
                              key={type.value}
                              onClick={() => setConfig({ ...config, deployment_type: type.value as any })}
                              className={`
                                p-3 rounded-lg border-2 transition-all text-center
                                ${config.deployment_type === type.value
                                  ? 'border-purple-500 bg-purple-50'
                                  : 'border-gray-200 hover:border-gray-300'
                                }
                              `}
                            >
                              <div className="text-2xl mb-1">{type.icon}</div>
                              <div className="text-sm font-semibold text-gray-900">{type.label}</div>
                              <div className="text-xs text-gray-500">{type.desc}</div>
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Environment */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Target Environment
                        </label>
                        <div className="grid grid-cols-3 gap-3">
                          {[
                            { value: 'production', label: 'Production', color: 'red' },
                            { value: 'staging', label: 'Staging', color: 'yellow' },
                            { value: 'testing', label: 'Testing', color: 'blue' }
                          ].map((env) => (
                            <button
                              key={env.value}
                              onClick={() => setConfig({ ...config, target_environment: env.value as any })}
                              className={`
                                p-2 rounded-lg border-2 transition-all text-sm font-medium
                                ${config.target_environment === env.value
                                  ? `border-${env.color}-500 bg-${env.color}-50 text-${env.color}-700`
                                  : 'border-gray-200 text-gray-700 hover:border-gray-300'
                                }
                              `}
                            >
                              {env.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Max Concurrent Executions */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Max Concurrent Executions
                        </label>
                        <input
                          type="number"
                          min="1"
                          max="100"
                          value={config.max_concurrent_executions}
                          onChange={(e) => setConfig({ ...config, max_concurrent_executions: parseInt(e.target.value) })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                        />
                      </div>

                      {/* Toggles */}
                      <div className="space-y-3">
                        <label className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer">
                          <span className="text-sm font-medium text-gray-700">Auto-scaling</span>
                          <input
                            type="checkbox"
                            checked={config.auto_scale}
                            onChange={(e) => setConfig({ ...config, auto_scale: e.target.checked })}
                            className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                          />
                        </label>

                        <label className="flex items-center justify-between p-3 bg-gray-50 rounded-lg cursor-pointer">
                          <span className="text-sm font-medium text-gray-700">Monitoring & Alerts</span>
                          <input
                            type="checkbox"
                            checked={config.monitoring_enabled}
                            onChange={(e) => setConfig({ ...config, monitoring_enabled: e.target.checked })}
                            className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                          />
                        </label>
                      </div>
                    </motion.div>
                  )}

                  {/* Step 3: Review & Deploy */}
                  {step === 3 && (
                    <motion.div
                      key="step3"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="space-y-4"
                    >
                      <h3 className="text-lg font-semibold text-gray-900">Review & Deploy</h3>

                      {!deploymentSuccess ? (
                        <>
                          <div className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-6 space-y-4">
                            <div className="flex items-center gap-3">
                              <div className="text-4xl">⚡</div>
                              <div className="flex-1">
                                <h4 className="text-lg font-semibold text-gray-900">{formula.name}</h4>
                                <p className="text-sm text-gray-600">Ready to deploy</p>
                              </div>
                              <TierBadge tier={formula.tier} size="md" animated />
                            </div>

                            <div className="grid grid-cols-2 gap-4 pt-2">
                              <div className="text-center p-3 bg-white rounded-lg">
                                <div className="text-xs text-gray-600 mb-1">Deployment</div>
                                <div className="text-sm font-semibold text-purple-600 capitalize">
                                  {config.deployment_type}
                                </div>
                              </div>
                              <div className="text-center p-3 bg-white rounded-lg">
                                <div className="text-xs text-gray-600 mb-1">Environment</div>
                                <div className="text-sm font-semibold text-purple-600 capitalize">
                                  {config.target_environment}
                                </div>
                              </div>
                              <div className="text-center p-3 bg-white rounded-lg">
                                <div className="text-xs text-gray-600 mb-1">Concurrent</div>
                                <div className="text-sm font-semibold text-purple-600">
                                  {config.max_concurrent_executions}
                                </div>
                              </div>
                              <div className="text-center p-3 bg-white rounded-lg">
                                <div className="text-xs text-gray-600 mb-1">Features</div>
                                <div className="text-sm font-semibold text-purple-600">
                                  {[config.auto_scale && 'Auto-scale', config.monitoring_enabled && 'Monitoring']
                                    .filter(Boolean)
                                    .join(', ') || 'Basic'}
                                </div>
                              </div>
                            </div>
                          </div>

                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <h4 className="text-sm font-semibold text-blue-900 mb-2">What happens next?</h4>
                            <ul className="space-y-1 text-xs text-blue-800">
                              <li className="flex items-center gap-2">
                                <span>✓</span>
                                <span>Formula will be deployed to {config.target_environment}</span>
                              </li>
                              <li className="flex items-center gap-2">
                                <span>✓</span>
                                <span>Endpoint will be available for execution</span>
                              </li>
                              {config.monitoring_enabled && (
                                <li className="flex items-center gap-2">
                                  <span>✓</span>
                                  <span>Real-time monitoring dashboard activated</span>
                                </li>
                              )}
                              <li className="flex items-center gap-2">
                                <span>✓</span>
                                <span>Deployment logs will be available</span>
                              </li>
                            </ul>
                          </div>
                        </>
                      ) : (
                        <motion.div
                          initial={{ scale: 0.9, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          className="text-center py-8"
                        >
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1, rotate: 360 }}
                            transition={{ duration: 0.5 }}
                            className="text-6xl mb-4"
                          >
                            ✅
                          </motion.div>
                          <h3 className="text-2xl font-bold text-green-600 mb-2">Deployment Successful!</h3>
                          <p className="text-gray-600">Formula is now live and ready to use</p>
                        </motion.div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Footer */}
              <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
                <button
                  onClick={handleClose}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>

                <div className="flex items-center gap-2">
                  {step > 1 && !deploymentSuccess && (
                    <button
                      onClick={() => setStep(step - 1)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Back
                    </button>
                  )}

                  {step < totalSteps ? (
                    <button
                      onClick={() => setStep(step + 1)}
                      className="px-6 py-2 text-sm font-medium text-white bg-purple-500 rounded-lg hover:bg-purple-600 transition-colors"
                    >
                      Next
                    </button>
                  ) : !deploymentSuccess && (
                    <button
                      onClick={handleDeploy}
                      disabled={isDeploying}
                      className="px-6 py-2 text-sm font-medium text-white bg-gradient-to-r from-purple-500 to-indigo-500 rounded-lg hover:from-purple-600 hover:to-indigo-600 transition-all shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      {isDeploying ? (
                        <>
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Deploying...
                        </>
                      ) : (
                        <>
                          <span>⚡</span>
                          Deploy Now
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default DeploymentWizard;
