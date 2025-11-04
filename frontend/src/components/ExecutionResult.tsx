import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Edit3, Clock, TrendingUp, Shield, AlertTriangle } from 'lucide-react';
import CorrectionModal from './CorrectionModal';
import { toast } from 'sonner';

interface ExecutionResultProps {
  execution: {
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
  };
  userRole: string;
  onCorrectionsUpdated?: () => void;
}

const ExecutionResult: React.FC<ExecutionResultProps> = ({
  execution,
  userRole,
  onCorrectionsUpdated
}) => {
  const [isCorrectionModalOpen, setIsCorrectionModalOpen] = useState(false);
  const [isApproved, setIsApproved] = useState(false);

  const tierInfo = {
    1: { label: 'Tier 1: Certified', color: 'text-green-600', bg: 'bg-green-100', icon: Shield },
    2: { label: 'Tier 2: Validated', color: 'text-blue-600', bg: 'bg-blue-100', icon: TrendingUp },
    3: { label: 'Tier 3: Testing', color: 'text-yellow-600', bg: 'bg-yellow-100', icon: Clock },
    4: { label: 'Tier 4: Experimental', color: 'text-red-600', bg: 'bg-red-100', icon: AlertTriangle }
  };

  const tier = execution.tier || 4;
  const tierData = tierInfo[tier as keyof typeof tierInfo] || tierInfo[4];
  const TierIcon = tierData.icon;

  const handleApprove = () => {
    setIsApproved(true);
    toast.success('Result approved');
  };

  const handleCorrect = () => {
    setIsCorrectionModalOpen(true);
  };

  const handleCorrectionSubmitted = () => {
    onCorrectionsUpdated?.();
    toast.success('Correction submitted and will be reviewed by an admin');
  };

  const canCorrect = userRole === 'operator' || userRole === 'admin';

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-lg font-semibold text-gray-900">
                AI Response
              </h3>
              <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${tierData.bg} ${tierData.color}`}>
                <TierIcon className="h-3 w-3" />
                {tierData.label}
              </span>
              {execution.status === 'completed' && (
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                  <CheckCircle className="h-3 w-3" />
                  Completed
                </span>
              )}
            </div>
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span>Execution ID: <span className="font-mono">{execution.execution_id}</span></span>
              {execution.execution_time && (
                <span>Time: {execution.execution_time.toFixed(3)}s</span>
              )}
              <span>
                {new Date(execution.execution_timestamp).toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        {/* Input Values */}
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Input Parameters</h4>
          <div className="bg-gray-50 rounded-lg p-3">
            <pre className="text-sm font-mono text-gray-700 overflow-x-auto">
              {JSON.stringify(execution.input_values, null, 2)}
            </pre>
          </div>
        </div>

        {/* Output Values */}
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Output Results</h4>
          <div className={`rounded-lg p-4 border-2 ${isApproved ? 'border-green-300 bg-green-50' : 'bg-blue-50 border-blue-200'}`}>
            <pre className="text-sm font-mono text-gray-800 overflow-x-auto">
              {JSON.stringify(execution.output_values, null, 2)}
            </pre>
            {isApproved && (
              <div className="mt-3 flex items-center gap-2 text-sm text-green-700">
                <CheckCircle className="h-4 w-4" />
                <span className="font-medium">Approved by operator</span>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons for Operators */}
        {canCorrect && !isApproved && (
          <div className="flex items-center gap-3 pt-4 border-t border-gray-200">
            <div className="flex-1 text-sm text-gray-600">
              Is this result correct?
            </div>
            <div className="flex gap-3">
              <button
                onClick={handleApprove}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <CheckCircle className="h-4 w-4" />
                Approve
              </button>
              <button
                onClick={handleCorrect}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors flex items-center gap-2"
              >
                <Edit3 className="h-4 w-4" />
                Correct
              </button>
            </div>
          </div>
        )}

        {/* Tier Warning for Non-Certified */}
        {tier > 1 && (
          <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-start gap-2">
            <AlertTriangle className="h-5 w-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-yellow-800">
              <strong>Note:</strong> This formula is not yet certified for production use.
              {userRole === 'operator' && ' Only admins can execute non-certified formulas.'}
            </div>
          </div>
        )}
      </motion.div>

      {/* Correction Modal */}
      <CorrectionModal
        isOpen={isCorrectionModalOpen}
        onClose={() => setIsCorrectionModalOpen(false)}
        executionId={execution.id}
        originalOutput={execution.output_values}
        onCorrectionSubmitted={handleCorrectionSubmitted}
      />
    </>
  );
};

export default ExecutionResult;
