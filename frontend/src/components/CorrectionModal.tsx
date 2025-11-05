import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertCircle, CheckCircle, Edit3 } from 'lucide-react';
import { apiService } from '../services/api';
import { toast } from 'sonner';

interface CorrectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  executionId: number;
  originalOutput: any;
  onCorrectionSubmitted?: () => void;
}

const CorrectionModal: React.FC<CorrectionModalProps> = ({
  isOpen,
  onClose,
  executionId,
  originalOutput,
  onCorrectionSubmitted
}) => {
  const [correctionType, setCorrectionType] = useState('value_correction');
  const [correctedOutput, setCorrectedOutput] = useState(JSON.stringify(originalOutput, null, 2));
  const [correctionReason, setCorrectionReason] = useState('');
  const [operatorConfidence, setOperatorConfidence] = useState(100);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const correctionTypes = [
    { value: 'value_correction', label: 'Value Correction', description: 'Correcting output values' },
    { value: 'classification_correction', label: 'Classification Correction', description: 'Correcting classifications' },
    { value: 'detection_correction', label: 'Detection Correction', description: 'Correcting object detections' },
    { value: 'formula_correction', label: 'Formula Correction', description: 'Correcting formula logic' },
    { value: 'parameter_correction', label: 'Parameter Correction', description: 'Correcting input parameters' }
  ];

  const handleSubmit = async () => {
    if (!correctionReason.trim()) {
      toast.error('Please provide a reason for the correction');
      return;
    }

    try {
      const parsedOutput = JSON.parse(correctedOutput);
      setIsSubmitting(true);

      await apiService.createCorrection({
        execution_id: executionId,
        correction_type: correctionType,
        corrected_output: parsedOutput,
        correction_reason: correctionReason,
        operator_confidence: operatorConfidence
      });

      toast.success('Correction submitted successfully!');
      onCorrectionSubmitted?.();
      onClose();
    } catch (error: any) {
      if (error instanceof SyntaxError) {
        toast.error('Invalid JSON format in corrected output');
      } else {
        toast.error(error.response?.data?.detail || 'Failed to submit correction');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    setCorrectedOutput(JSON.stringify(originalOutput, null, 2));
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
          >
            <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
              {/* Header */}
              <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-orange-100 rounded-lg">
                    <Edit3 className="h-6 w-6 text-orange-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Submit Correction</h2>
                    <p className="text-gray-600 text-sm mt-1">
                      Correct the AI output - This creates a verifiable record
                    </p>
                  </div>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="h-5 w-5 text-gray-500" />
                </button>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {/* Info Box */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
                  <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-blue-800">
                    <strong>Important:</strong> Your correction will be reviewed by an admin and can be used to improve the AI model through retraining.
                  </div>
                </div>

                {/* Correction Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Correction Type
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {correctionTypes.map((type) => (
                      <button
                        key={type.value}
                        onClick={() => setCorrectionType(type.value)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          correctionType === type.value
                            ? 'border-orange-500 bg-orange-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                      >
                        <div className="font-medium text-gray-900">{type.label}</div>
                        <div className="text-xs text-gray-600 mt-1">{type.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Original vs Corrected Output */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Original Output */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Original AI Output (Read-only)
                    </label>
                    <div className="bg-gray-50 border border-gray-300 rounded-lg p-3 h-64 overflow-auto">
                      <pre className="text-xs font-mono text-gray-700">
                        {JSON.stringify(originalOutput, null, 2)}
                      </pre>
                    </div>
                  </div>

                  {/* Corrected Output */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="block text-sm font-medium text-gray-700">
                        Corrected Output
                      </label>
                      <button
                        onClick={handleReset}
                        className="text-xs text-blue-600 hover:text-blue-700"
                      >
                        Reset
                      </button>
                    </div>
                    <textarea
                      value={correctedOutput}
                      onChange={(e) => setCorrectedOutput(e.target.value)}
                      className="w-full h-64 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent font-mono text-xs"
                      placeholder="Enter corrected output as JSON"
                    />
                  </div>
                </div>

                {/* Correction Reason */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Correction Reason <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    value={correctionReason}
                    onChange={(e) => setCorrectionReason(e.target.value)}
                    className="w-full h-24 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                    placeholder="Explain why you're making this correction..."
                    required
                  />
                </div>

                {/* Operator Confidence */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Your Confidence Level: {operatorConfidence}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={operatorConfidence}
                    onChange={(e) => setOperatorConfidence(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-600"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Not sure (0%)</span>
                    <span>Very confident (100%)</span>
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-gray-200 flex justify-between items-center">
                <div className="text-sm text-gray-600">
                  Execution ID: <span className="font-mono">{executionId}</span>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={onClose}
                    disabled={isSubmitting}
                    className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={isSubmitting || !correctionReason.trim()}
                    className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                        Submitting...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4" />
                        Submit Correction
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default CorrectionModal;
