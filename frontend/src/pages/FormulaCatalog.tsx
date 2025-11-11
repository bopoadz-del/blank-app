import React, { useState, useEffect, useMemo } from 'react';
import { AnimatePresence } from 'framer-motion';
import { FormulaCard, FormulaCardSkeleton } from '../components/FormulaCard';
import { TierFilter } from '../components/TierBadge';
import { DeploymentWizard, DeploymentConfig } from '../components/DeploymentWizard';
import { apiService } from '../services/api';
import type { Formula } from '../types';

const DOMAINS = [
  'structural_engineering',
  'concrete_technology',
  'thermal_analysis',
  'financial_metrics',
  'energy_systems',
  'manufacturing',
  'fluid_dynamics'
];

export const FormulaCatalog: React.FC = () => {
  const [formulas, setFormulas] = useState<Formula[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Search and filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [selectedTiers, setSelectedTiers] = useState<number[]>([]);
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'confidence' | 'executions' | 'tier'>('tier');

  // Deployment wizard state
  const [selectedFormula, setSelectedFormula] = useState<Formula | null>(null);
  const [isDeploymentWizardOpen, setIsDeploymentWizardOpen] = useState(false);

  // View mode
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');

  // Fetch formulas
  useEffect(() => {
    fetchFormulas();
  }, []);

  const fetchFormulas = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getFormulas();
      setFormulas(data.formulas || data || []);
    } catch (err: any) {
      setError(err.message || 'Failed to load formulas');
      console.error('Error fetching formulas:', err);
    } finally {
      setLoading(false);
    }
  };

  // Filter and search logic
  const filteredFormulas = useMemo(() => {
    let filtered = [...formulas];

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (f) =>
          f.name.toLowerCase().includes(query) ||
          f.domain.toLowerCase().includes(query) ||
          f.equation.toLowerCase().includes(query) ||
          f.description?.toLowerCase().includes(query)
      );
    }

    // Domain filter
    if (selectedDomains.length > 0) {
      filtered = filtered.filter((f) => selectedDomains.includes(f.domain));
    }

    // Tier filter
    if (selectedTiers.length > 0) {
      filtered = filtered.filter((f) => selectedTiers.includes(f.tier));
    }

    // Status filter
    if (selectedStatus !== 'all') {
      filtered = filtered.filter((f) => f.status === selectedStatus);
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'confidence':
          return b.confidence_score - a.confidence_score;
        case 'executions':
          return b.execution_count - a.execution_count;
        case 'tier':
          return b.tier - a.tier;
        default:
          return 0;
      }
    });

    return filtered;
  }, [formulas, searchQuery, selectedDomains, selectedTiers, selectedStatus, sortBy]);

  // Tier counts for filter
  const tierCounts = useMemo(() => {
    const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };
    formulas.forEach((f) => {
      counts[f.tier] = (counts[f.tier] || 0) + 1;
    });
    return counts;
  }, [formulas]);

  // Handlers
  const handleDomainToggle = (domain: string) => {
    setSelectedDomains((prev) =>
      prev.includes(domain) ? prev.filter((d) => d !== domain) : [...prev, domain]
    );
  };

  const handleTierToggle = (tier: number) => {
    setSelectedTiers((prev) =>
      prev.includes(tier) ? prev.filter((t) => t !== tier) : [...prev, tier]
    );
  };

  const handleDeploy = (formula: Formula) => {
    setSelectedFormula(formula);
    setIsDeploymentWizardOpen(true);
  };

  const handleDeploymentSubmit = async (config: DeploymentConfig) => {
    console.log('Deploying formula with config:', config);
    // In a real app, this would make an API call to deploy the formula
    await new Promise((resolve) => setTimeout(resolve, 1500)); // Simulate API call
    alert(`Formula deployed successfully to ${config.target_environment}!`);
  };

  const handleExecute = (formula: Formula) => {
    // Navigate to execution page or open execution modal
    console.log('Execute formula:', formula);
    alert(`Execution interface for ${formula.name} would open here`);
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setSelectedDomains([]);
    setSelectedTiers([]);
    setSelectedStatus('all');
  };

  const hasActiveFilters =
    searchQuery.trim() !== '' ||
    selectedDomains.length > 0 ||
    selectedTiers.length > 0 ||
    selectedStatus !== 'all';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <span className="text-4xl">📚</span>
                Formula Catalog
              </h1>
              <p className="text-gray-600 mt-1">
                Browse and deploy validated mathematical formulas
              </p>
            </div>

            <div className="flex items-center gap-3">
              {/* View Mode Toggle */}
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`
                    px-3 py-1.5 rounded text-sm font-medium transition-all
                    ${viewMode === 'grid'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                    }
                  `}
                >
                  Grid
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`
                    px-3 py-1.5 rounded text-sm font-medium transition-all
                    ${viewMode === 'list'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                    }
                  `}
                >
                  List
                </button>
              </div>

              {/* Refresh Button */}
              <button
                onClick={fetchFormulas}
                disabled={loading}
                className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-sm"
              >
                <svg
                  className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                Refresh
              </button>
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by name, domain, equation, or description..."
              className="w-full px-4 py-3 pl-12 pr-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900 placeholder-gray-400"
            />
            <svg
              className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-8">
          {/* Sidebar - Filters */}
          <div className="w-64 flex-shrink-0 space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 space-y-4 sticky top-32">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-bold text-gray-900 uppercase">Filters</h3>
                {hasActiveFilters && (
                  <button
                    onClick={handleClearFilters}
                    className="text-xs text-purple-600 hover:text-purple-700 font-medium"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {/* Tier Filter */}
              <TierFilter
                selectedTiers={selectedTiers}
                onTierToggle={handleTierToggle}
                counts={tierCounts}
              />

              {/* Status Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active</option>
                  <option value="testing">Testing</option>
                  <option value="deprecated">Deprecated</option>
                </select>
              </div>

              {/* Domain Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Domain</label>
                <div className="space-y-1.5 max-h-64 overflow-y-auto">
                  {DOMAINS.map((domain) => (
                    <label key={domain} className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={selectedDomains.includes(domain)}
                        onChange={() => handleDomainToggle(domain)}
                        className="w-4 h-4 text-purple-600 rounded focus:ring-purple-500"
                      />
                      <span className="text-sm text-gray-700 capitalize group-hover:text-purple-600 transition-colors">
                        {domain.replace('_', ' ')}
                      </span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Sort By */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm"
                >
                  <option value="tier">Tier Level</option>
                  <option value="name">Name</option>
                  <option value="confidence">Confidence Score</option>
                  <option value="executions">Executions</option>
                </select>
              </div>
            </div>
          </div>

          {/* Formula Grid/List */}
          <div className="flex-1">
            {/* Results Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="text-sm text-gray-600">
                {loading ? (
                  'Loading formulas...'
                ) : (
                  <>
                    Showing <span className="font-semibold text-gray-900">{filteredFormulas.length}</span> of{' '}
                    <span className="font-semibold text-gray-900">{formulas.length}</span> formulas
                    {hasActiveFilters && (
                      <span className="ml-1 text-purple-600">(filtered)</span>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Error State */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
                <div className="text-4xl mb-3">⚠️</div>
                <h3 className="text-lg font-semibold text-red-900 mb-2">Failed to Load Formulas</h3>
                <p className="text-sm text-red-700 mb-4">{error}</p>
                <button
                  onClick={fetchFormulas}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Try Again
                </button>
              </div>
            )}

            {/* Loading State */}
            {loading && !error && (
              <div
                className={`
                  ${viewMode === 'grid'
                    ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6'
                    : 'space-y-4'
                  }
                `}
              >
                {[...Array(6)].map((_, i) => (
                  <FormulaCardSkeleton key={i} />
                ))}
              </div>
            )}

            {/* Empty State */}
            {!loading && !error && filteredFormulas.length === 0 && (
              <div className="bg-white border border-gray-200 rounded-lg p-12 text-center">
                <div className="text-6xl mb-4">🔍</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">No Formulas Found</h3>
                <p className="text-gray-600 mb-6">
                  {hasActiveFilters
                    ? 'Try adjusting your filters or search query'
                    : 'No formulas available yet'}
                </p>
                {hasActiveFilters && (
                  <button
                    onClick={handleClearFilters}
                    className="px-6 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
                  >
                    Clear Filters
                  </button>
                )}
              </div>
            )}

            {/* Formula Grid/List */}
            {!loading && !error && filteredFormulas.length > 0 && (
              <AnimatePresence mode="popLayout">
                <div
                  className={`
                    ${viewMode === 'grid'
                      ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6'
                      : 'space-y-4'
                    }
                  `}
                >
                  {filteredFormulas.map((formula) => (
                    <FormulaCard
                      key={formula.id}
                      formula={formula}
                      onDeploy={handleDeploy}
                      onExecute={handleExecute}
                    />
                  ))}
                </div>
              </AnimatePresence>
            )}
          </div>
        </div>
      </div>

      {/* Deployment Wizard Modal */}
      <DeploymentWizard
        formula={selectedFormula}
        isOpen={isDeploymentWizardOpen}
        onClose={() => {
          setIsDeploymentWizardOpen(false);
          setSelectedFormula(null);
        }}
        onDeploy={handleDeploymentSubmit}
      />
    </div>
  );
};

export default FormulaCatalog;
