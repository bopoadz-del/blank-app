import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Eye, TrendingUp, Clock, CheckCircle, XCircle, AlertTriangle,
  FileText, Activity, Search, Filter, Calendar, Shield
} from 'lucide-react';
import { apiService } from '../services/api';
import { toast } from 'sonner';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const AuditorDashboard: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [timeline, setTimeline] = useState<any>(null);
  const [tierDistribution, setTierDistribution] = useState<any>(null);
  const [auditLogs, setAuditLogs] = useState<any[]>([]);
  const [selectedLog, setSelectedLog] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState(30);
  const [filterAction, setFilterAction] = useState('');
  const [filterEntityType, setFilterEntityType] = useState('');

  useEffect(() => {
    fetchDashboardData();
  }, [timeRange]);

  const fetchDashboardData = async () => {
    setIsLoading(true);
    try {
      const [statsData, timelineData, tierData, logsData] = await Promise.all([
        apiService.getAuditorDashboardStats(timeRange),
        apiService.getCorrectionsTimeline(timeRange),
        apiService.getFormulaTierDistribution(),
        apiService.getAuditLogs({ days: timeRange, limit: 50 })
      ]);

      setStats(statsData);
      setTimeline(timelineData.timeline);
      setTierDistribution(tierData.distribution);
      setAuditLogs(logsData);
    } catch (error: any) {
      console.error('Error fetching dashboard data:', error);
      toast.error('Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const filterAuditLogs = async () => {
    try {
      const logsData = await apiService.getAuditLogs({
        action: filterAction || undefined,
        entity_type: filterEntityType || undefined,
        days: timeRange,
        limit: 100
      });
      setAuditLogs(logsData);
    } catch (error) {
      toast.error('Failed to filter audit logs');
    }
  };

  const viewExecutionTrail = async (executionId: number) => {
    try {
      const trail = await apiService.getExecutionTrail(executionId);
      setSelectedLog(trail);
      toast.success('Execution trail loaded');
    } catch (error) {
      toast.error('Failed to load execution trail');
    }
  };

  // Prepare chart data
  const chartData = timeline ? {
    labels: timeline.map((t: any) => t.date),
    datasets: [
      {
        label: 'Total Corrections',
        data: timeline.map((t: any) => t.total),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
      },
      {
        label: 'Approved',
        data: timeline.map((t: any) => t.approved),
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.2)',
      },
      {
        label: 'Rejected',
        data: timeline.map((t: any) => t.rejected),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
      }
    ]
  } : null;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Eye className="h-8 w-8 text-blue-600" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Auditor Dashboard</h1>
            <p className="text-gray-600">Complete read-only view of system audit trail</p>
          </div>
        </div>

        {/* Time Range Selector */}
        <div className="flex items-center gap-4 mt-4">
          <Calendar className="h-5 w-5 text-gray-500" />
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(parseInt(e.target.value))}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Executions</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.total_executions || 0}</p>
            </div>
            <Activity className="h-10 w-10 text-blue-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Corrections</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.total_corrections || 0}</p>
            </div>
            <FileText className="h-10 w-10 text-orange-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Pending Review</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.pending_corrections || 0}</p>
            </div>
            <Clock className="h-10 w-10 text-yellow-600" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tier 1 Formulas</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.tier_1_formulas || 0}</p>
            </div>
            <Shield className="h-10 w-10 text-green-600" />
          </div>
        </motion.div>
      </div>

      {/* Corrections Timeline Chart */}
      {chartData && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Corrections Timeline</h2>
          <Line data={chartData} options={{
            responsive: true,
            plugins: {
              legend: {
                position: 'top' as const,
              },
              title: {
                display: false,
              },
            },
          }} />
        </div>
      )}

      {/* Tier Distribution */}
      {tierDistribution && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Formula Tier Distribution</h2>
          <div className="grid grid-cols-4 gap-4">
            {Object.entries(tierDistribution).map(([tier, count]: [string, any]) => {
              const tierNum = parseInt(tier.split('_')[1]);
              const tierColors = {
                1: 'bg-green-100 text-green-700 border-green-300',
                2: 'bg-blue-100 text-blue-700 border-blue-300',
                3: 'bg-yellow-100 text-yellow-700 border-yellow-300',
                4: 'bg-red-100 text-red-700 border-red-300'
              };
              return (
                <div
                  key={tier}
                  className={`p-4 rounded-lg border-2 ${tierColors[tierNum as keyof typeof tierColors]}`}
                >
                  <div className="text-2xl font-bold">{count}</div>
                  <div className="text-sm">Tier {tierNum}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Audit Logs */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">Audit Trail</h2>
          <div className="flex gap-3">
            <input
              type="text"
              placeholder="Filter by action..."
              value={filterAction}
              onChange={(e) => setFilterAction(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
            />
            <input
              type="text"
              placeholder="Filter by entity type..."
              value={filterEntityType}
              onChange={(e) => setFilterEntityType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm"
            />
            <button
              onClick={filterAuditLogs}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
            >
              <Filter className="h-4 w-4" />
              Filter
            </button>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entity</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">User</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {auditLogs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900 font-mono">
                    {new Date(log.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700">
                      {log.action}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {log.entity_type}
                    {log.entity_id && <span className="text-gray-500 ml-1">#{log.entity_id}</span>}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900">
                    {log.user_id || 'System'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {log.success ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {log.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AuditorDashboard;
