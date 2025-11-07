import React from 'react';
import { motion } from 'framer-motion';

interface TierBadgeProps {
  tier: 1 | 2 | 3 | 4;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  animated?: boolean;
}

const tierConfig = {
  1: {
    label: 'Tier 1: Experimental',
    shortLabel: 'T1',
    color: 'from-gray-400 to-gray-500',
    borderColor: 'border-gray-400',
    textColor: 'text-gray-700',
    bgColor: 'bg-gray-100',
    icon: '🧪',
    description: 'Unvalidated formulas requiring human supervision'
  },
  2: {
    label: 'Tier 2: Validated',
    shortLabel: 'T2',
    color: 'from-blue-400 to-blue-500',
    borderColor: 'border-blue-400',
    textColor: 'text-blue-700',
    bgColor: 'bg-blue-50',
    icon: '✓',
    description: 'Empirically validated with ≥70% confidence'
  },
  3: {
    label: 'Tier 3: Certified',
    shortLabel: 'T3',
    color: 'from-green-400 to-green-500',
    borderColor: 'border-green-400',
    textColor: 'text-green-700',
    bgColor: 'bg-green-50',
    icon: '✓✓',
    description: 'Certified formulas with ≥95% confidence'
  },
  4: {
    label: 'Tier 4: Auto-Deploy',
    shortLabel: 'T4',
    color: 'from-purple-400 via-purple-500 to-indigo-500',
    borderColor: 'border-purple-500',
    textColor: 'text-purple-800',
    bgColor: 'bg-purple-50',
    icon: '⚡',
    description: 'Fully autonomous with near-perfect accuracy'
  }
};

export const TierBadge: React.FC<TierBadgeProps> = ({
  tier,
  size = 'md',
  showLabel = true,
  animated = false
}) => {
  const config = tierConfig[tier];

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-2'
  };

  const iconSizes = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-lg'
  };

  const BadgeContent = (
    <div
      className={`
        inline-flex items-center gap-1.5 rounded-full
        border-2 ${config.borderColor}
        ${config.bgColor} ${config.textColor}
        ${sizeClasses[size]}
        font-semibold
        shadow-sm
        transition-all duration-200
        hover:shadow-md
        group
      `}
    >
      <span className={`${iconSizes[size]}`}>{config.icon}</span>
      {showLabel && (
        <span className="whitespace-nowrap">
          {size === 'sm' ? config.shortLabel : config.label}
        </span>
      )}
    </div>
  );

  if (animated) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.05 }}
        transition={{ duration: 0.2 }}
        className="inline-block relative group"
        title={config.description}
      >
        {BadgeContent}

        {/* Tooltip on hover */}
        <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-50">
          <div className="bg-gray-900 text-white text-xs rounded py-2 px-3 whitespace-nowrap shadow-lg">
            {config.description}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-4 border-transparent border-t-gray-900"></div>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <div className="inline-block" title={config.description}>
      {BadgeContent}
    </div>
  );
};

// Tier Progress Bar Component
interface TierProgressProps {
  currentTier: 1 | 2 | 3 | 4;
  confidenceScore: number;
  nextTierThreshold?: number;
}

export const TierProgress: React.FC<TierProgressProps> = ({
  currentTier,
  confidenceScore,
  nextTierThreshold = 0.95
}) => {
  const progress = (confidenceScore / nextTierThreshold) * 100;
  const config = tierConfig[currentTier];

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <TierBadge tier={currentTier} size="sm" />
        <span className={`font-semibold ${config.textColor}`}>
          {(confidenceScore * 100).toFixed(1)}%
        </span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(progress, 100)}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className={`h-full bg-gradient-to-r ${config.color} rounded-full`}
        />
      </div>

      {currentTier < 4 && (
        <div className="text-xs text-gray-500 flex items-center justify-between">
          <span>
            Next tier: {tierConfig[(currentTier + 1) as 1 | 2 | 3 | 4].shortLabel}
          </span>
          <span>
            {((nextTierThreshold - confidenceScore) * 100).toFixed(1)}% to go
          </span>
        </div>
      )}
    </div>
  );
};

// Tier Filter Component for FormulaCatalog
interface TierFilterProps {
  selectedTiers: number[];
  onTierToggle: (tier: number) => void;
  counts?: Record<number, number>;
}

export const TierFilter: React.FC<TierFilterProps> = ({
  selectedTiers,
  onTierToggle,
  counts = {}
}) => {
  const tiers = [1, 2, 3, 4] as const;

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Filter by Tier
      </label>
      <div className="space-y-1.5">
        {tiers.map((tier) => {
          const isSelected = selectedTiers.includes(tier);
          const config = tierConfig[tier];
          const count = counts[tier] || 0;

          return (
            <motion.button
              key={tier}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onTierToggle(tier)}
              className={`
                w-full flex items-center justify-between
                px-3 py-2 rounded-lg border-2 transition-all
                ${isSelected
                  ? `${config.borderColor} ${config.bgColor}`
                  : 'border-gray-200 bg-white hover:border-gray-300'
                }
              `}
            >
              <div className="flex items-center gap-2">
                <span className="text-lg">{config.icon}</span>
                <span className={`text-sm font-medium ${isSelected ? config.textColor : 'text-gray-700'}`}>
                  {config.shortLabel}
                </span>
              </div>
              <span className={`text-xs font-semibold ${isSelected ? config.textColor : 'text-gray-400'}`}>
                {count}
              </span>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};

export default TierBadge;
