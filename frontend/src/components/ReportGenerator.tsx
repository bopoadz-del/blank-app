import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Download, Trash2, X, FileSpreadsheet, FileType } from 'lucide-react';
import api from '../services/api';
import { toast } from 'sonner';

interface Report {
  id: string;
  title: string;
  type: string;
  format: string;
  fileSize: number;
  createdAt: string;
}

interface ReportGeneratorProps {
  isOpen: boolean;
  onClose: () => void;
  resourceType?: 'conversation' | 'project';
  resourceId?: number;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({
  isOpen,
  onClose,
  resourceType = 'conversation',
  resourceId
}) => {
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<'pdf' | 'excel' | 'csv'>('pdf');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingReports, setIsLoadingReports] = useState(false);

  useEffect(() => {
    if (isOpen) {
      fetchReports();
    }
  }, [isOpen]);

  const fetchReports = async () => {
    setIsLoadingReports(true);
    try {
      const response = await api.get('/reports', {
        params: { limit: 20 }
      });
      setReports(response.data);
    } catch (error) {
      console.error('Error fetching reports:', error);
      toast.error('Failed to load reports');
    } finally {
      setIsLoadingReports(false);
    }
  };

  const generateReport = async () => {
    if (!resourceId) {
      toast.error('No resource selected');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await api.post('/reports/generate', {
        type: resourceType,
        format: selectedFormat,
        resourceId: resourceId
      });

      toast.success('Report generated successfully!');
      fetchReports(); // Refresh the list

      // Auto-download the report
      downloadReport(response.data.id);
    } catch (error: any) {
      console.error('Error generating report:', error);
      const message = error.response?.data?.detail || 'Failed to generate report';
      toast.error(message);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadReport = async (reportId: string) => {
    try {
      const response = await api.get(`/reports/${reportId}/download`, {
        responseType: 'blob'
      });

      // Get filename from content-disposition header or use default
      const contentDisposition = response.headers['content-disposition'];
      let filename = 'report';
      if (contentDisposition) {
        const match = contentDisposition.match(/filename="?(.+)"?/);
        if (match) filename = match[1];
      }

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);

      toast.success('Report downloaded');
    } catch (error) {
      console.error('Error downloading report:', error);
      toast.error('Failed to download report');
    }
  };

  const deleteReport = async (reportId: string) => {
    if (!confirm('Are you sure you want to delete this report?')) {
      return;
    }

    try {
      await api.delete(`/reports/${reportId}`);
      setReports(prev => prev.filter(r => r.id !== reportId));
      toast.success('Report deleted');
    } catch (error) {
      console.error('Error deleting report:', error);
      toast.error('Failed to delete report');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf':
        return <FileText className="h-5 w-5 text-red-500" />;
      case 'excel':
        return <FileSpreadsheet className="h-5 w-5 text-green-500" />;
      case 'csv':
        return <FileType className="h-5 w-5 text-blue-500" />;
      default:
        return <FileText className="h-5 w-5 text-gray-500" />;
    }
  };

  const formatTypes = [
    { value: 'pdf', label: 'PDF Document', icon: FileText, color: 'text-red-600' },
    { value: 'excel', label: 'Excel Spreadsheet', icon: FileSpreadsheet, color: 'text-green-600' },
    { value: 'csv', label: 'CSV File', icon: FileType, color: 'text-blue-600' }
  ];

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
            <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] flex flex-col">
              {/* Header */}
              <div className="p-6 border-b border-gray-200 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <FileText className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">Report Generator</h2>
                    <p className="text-gray-600 text-sm mt-1">
                      Generate and download reports
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
                {/* Generate New Report */}
                <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
                  <h3 className="font-semibold text-gray-900 mb-4">Generate New Report</h3>

                  {/* Format Selection */}
                  <div className="space-y-3 mb-4">
                    {formatTypes.map((format) => {
                      const Icon = format.icon;
                      return (
                        <button
                          key={format.value}
                          onClick={() => setSelectedFormat(format.value as any)}
                          className={`w-full flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                            selectedFormat === format.value
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <Icon className={`h-6 w-6 ${format.color}`} />
                          <span className="font-medium text-gray-900">{format.label}</span>
                          {selectedFormat === format.value && (
                            <div className="ml-auto h-5 w-5 rounded-full bg-blue-500 flex items-center justify-center">
                              <div className="h-2 w-2 bg-white rounded-full" />
                            </div>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {/* Generate Button */}
                  <button
                    onClick={generateReport}
                    disabled={isGenerating || !resourceId}
                    className="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    <FileText className="h-5 w-5" />
                    {isGenerating ? 'Generating Report...' : 'Generate Report'}
                  </button>
                </div>

                {/* Previous Reports */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-4">Recent Reports</h3>
                  {isLoadingReports ? (
                    <div className="text-center py-8 text-gray-500">
                      Loading reports...
                    </div>
                  ) : reports.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <FileText className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                      <p>No reports generated yet</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {reports.map((report) => (
                        <motion.div
                          key={report.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
                        >
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            {getFormatIcon(report.format)}
                            <div className="flex-1 min-w-0">
                              <h4 className="font-medium text-gray-900 truncate">
                                {report.title}
                              </h4>
                              <p className="text-sm text-gray-500">
                                {formatFileSize(report.fileSize)} • {new Date(report.createdAt).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => downloadReport(report.id)}
                              className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                              title="Download"
                            >
                              <Download className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => deleteReport(report.id)}
                              className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                              title="Delete"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Footer */}
              <div className="p-6 border-t border-gray-200 flex justify-end">
                <button
                  onClick={onClose}
                  className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  Close
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default ReportGenerator;
