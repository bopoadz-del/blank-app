import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { TierBadge, TierProgress } from './TierBadge';
import type { Formula } from '../types';

interface FormulaCardProps {
  formula: Formula;
  onDeploy?: (formula: Formula) => void;
  onViewDetails?: (formula: Formula) => void;
  onExecute?: (formula: Formula) => void;
}

export const FormulaCard: React.FC<FormulaCardProps> = ({
  formula,
  onDeploy,
  onViewDetails,
  onExecute
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const statusColors = {
    active: 'bg-green-100 text-green-800 border-green-200',
    testing: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    deprecated: 'bg-red-100 text-red-800 border-red-200'
  };

  const statusIcons = {
    active: '✓',
    testing: '⚠',
    deprecated: '✗'
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      whileHover={{ y: -4 }}
      className="bg-white rounded-lg shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-200"
    >
      {/* Card Header */}
      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-bold text-gray-900 truncate mb-1">
              {formula.name}
            </h3>
            <p className="text-sm text-gray-600 capitalize">
              {formula.domain.replace('_', ' ')}
            </p>
          </div>

          <div className="flex flex-col items-end gap-2">
            <TierBadge tier={formula.tier} size="sm" animated />
            <span
              className={`
                inline-flex items-center gap-1 px-2 py-0.5 rounded-full
                text-xs font-medium border
                ${statusColors[formula.status]}
              `}
            >
              <span>{statusIcons[formula.status]}</span>
              {formula.status}
            </span>
          </div>
        </div>

        {/* Formula Equation */}
        <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
          <code className="text-sm text-gray-800 font-mono break-all">
            {formula.equation}
          </code>
        </div>

        {/* Description */}
        {formula.description && (
          <p className="text-sm text-gray-600 line-clamp-2">
            {formula.description}
          </p>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-2 pt-2">
          <div className="text-center p-2 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">
              {formula.execution_count.toLocaleString()}
            </div>
            <div className="text-xs text-gray-600">Executions</div>
          </div>
          <div className="text-center p-2 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {(formula.success_rate * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-600">Success Rate</div>
          </div>
          <div className="text-center p-2 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {(formula.confidence_score * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-600">Confidence</div>
          </div>
        </div>

        {/* Tier Progress */}
        {formula.tier < 4 && (
          <div className="pt-2">
            <TierProgress
              currentTier={formula.tier}
              confidenceScore={formula.confidence_score}
              nextTierThreshold={formula.tier === 1 ? 0.70 : formula.tier === 2 ? 0.95 : 0.99}
            />
          </div>
        )}

        {/* Tags */}
        {formula.tags && formula.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {formula.tags.map((tag, idx) => (
              <span
                key={idx}
                className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Expandable Details */}
      <motion.div
        initial={false}
        animate={{ height: isExpanded ? 'auto' : 0 }}
        className="overflow-hidden"
      >
        <div className="px-5 pb-3 border-t border-gray-200 pt-3 space-y-3">
          {/* Input Parameters */}
          <div>
            <h4 className="text-xs font-semibold text-gray-700 uppercase mb-2">
              Input Parameters ({formula.input_parameters.length})
            </h4>
            <div className="space-y-1">
              {formula.input_parameters.slice(0, 5).map((param, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs bg-gray-50 px-2 py-1 rounded"
                >
                  <span className="font-medium text-gray-700">{param.name}</span>
                  <span className="text-gray-500">
                    {param.unit} {param.required && <span className="text-red-500">*</span>}
                  </span>
                </div>
              ))}
              {formula.input_parameters.length > 5 && (
                <div className="text-xs text-gray-500 text-center py-1">
                  +{formula.input_parameters.length - 5} more
                </div>
              )}
            </div>
          </div>

          {/* Output Parameters */}
          <div>
            <h4 className="text-xs font-semibold text-gray-700 uppercase mb-2">
              Output Parameters ({formula.output_parameters.length})
            </h4>
            <div className="space-y-1">
              {formula.output_parameters.map((param, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs bg-green-50 px-2 py-1 rounded border border-green-100"
                >
                  <span className="font-medium text-green-700">{param.name}</span>
                  <span className="text-green-600">{param.unit}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Validation Stages */}
          {formula.validation_stages && formula.validation_stages.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-gray-700 uppercase mb-2">
                Validation Status
              </h4>
              <div className="space-y-1">
                {formula.validation_stages.map((stage, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between text-xs px-2 py-1 rounded bg-gray-50"
                  >
                    <span className="text-gray-700 capitalize">
                      {stage.stage.replace('_', ' ')}
                    </span>
                    <span
                      className={`font-medium ${
                        stage.status === 'passed'
                          ? 'text-green-600'
                          : stage.status === 'failed'
                          ? 'text-red-600'
                          : 'text-yellow-600'
                      }`}
                    >
                      {stage.status === 'passed' ? '✓' : stage.status === 'failed' ? '✗' : '○'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Action Buttons */}
      <div className="px-5 py-3 bg-gray-50 border-t border-gray-200 flex items-center gap-2">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex-1 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
        >
          {isExpanded ? '− Less Info' : '+ More Info'}
        </button>

        {onExecute && formula.status === 'active' && (
          <button
            onClick={() => onExecute(formula)}
            className="flex-1 px-3 py-2 text-sm font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-lg hover:bg-blue-100 transition-colors flex items-center justify-center gap-1"
          >
            <span>▶</span> Execute
          </button>
        )}

        {onDeploy && formula.tier >= 2 && (
          <button
            onClick={() => onDeploy(formula)}
            className="flex-1 px-3 py-2 text-sm font-medium text-white bg-gradient-to-r from-purple-500 to-indigo-500 rounded-lg hover:from-purple-600 hover:to-indigo-600 transition-all shadow-sm hover:shadow-md flex items-center justify-center gap-1"
          >
            <span>⚡</span> Deploy
          </button>
        )}

        {onViewDetails && (
          <button
            onClick={() => onViewDetails(formula)}
            className="px-3 py-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors"
            title="View full details"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>
        )}
      </div>
    </motion.div>
  );
};

// Skeleton loader for formula cards
export const FormulaCardSkeleton: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200 animate-pulse">
      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 space-y-2">
            <div className="h-6 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
          <div className="flex flex-col gap-2">
            <div className="h-6 bg-gray-200 rounded w-16"></div>
            <div className="h-5 bg-gray-200 rounded w-16"></div>
          </div>
        </div>
        <div className="h-16 bg-gray-100 rounded"></div>
        <div className="h-12 bg-gray-100 rounded"></div>
        <div className="grid grid-cols-3 gap-2">
          <div className="h-16 bg-gray-100 rounded"></div>
          <div className="h-16 bg-gray-100 rounded"></div>
          <div className="h-16 bg-gray-100 rounded"></div>
        </div>
      </div>
      <div className="px-5 py-3 bg-gray-50 border-t border-gray-200 flex gap-2">
        <div className="flex-1 h-9 bg-gray-200 rounded"></div>
        <div className="flex-1 h-9 bg-gray-200 rounded"></div>
      </div>
    </div>
  );
};

export default FormulaCard;
