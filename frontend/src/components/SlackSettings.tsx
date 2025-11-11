import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { MessageSquare, Save, Trash2, TestTube, ExternalLink, Check } from 'lucide-react';
import api from '../services/api';
import { toast } from 'sonner';

interface SlackIntegration {
  id: string;
  workspaceName: string;
  webhookUrl: string;
  channel: string;
  enabled: boolean;
  createdAt: string;
}

const SlackSettings: React.FC = () => {
  const [integration, setIntegration] = useState<SlackIntegration | null>(null);
  const [formData, setFormData] = useState({
    workspaceName: '',
    webhookUrl: '',
    channel: ''
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  useEffect(() => {
    fetchIntegration();
  }, []);

  const fetchIntegration = async () => {
    setIsLoading(true);
    try {
      const response = await api.get('/slack/integration');
      if (response.data) {
        setIntegration(response.data);
        setFormData({
          workspaceName: response.data.workspaceName,
          webhookUrl: response.data.webhookUrl,
          channel: response.data.channel
        });
      }
    } catch (error) {
      console.error('Error fetching Slack integration:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveIntegration = async () => {
    if (!formData.workspaceName || !formData.webhookUrl || !formData.channel) {
      toast.error('Please fill in all fields');
      return;
    }

    setIsSaving(true);
    try {
      const response = await api.post('/slack/integration', formData);
      setIntegration(response.data);
      toast.success('Slack integration saved successfully');
    } catch (error) {
      console.error('Error saving Slack integration:', error);
      toast.error('Failed to save Slack integration');
    } finally {
      setIsSaving(false);
    }
  };

  const testWebhook = async () => {
    if (!formData.webhookUrl) {
      toast.error('Please enter a webhook URL');
      return;
    }

    setIsTesting(true);
    try {
      const response = await api.post('/slack/test', formData);
      if (response.data.success) {
        toast.success('Test message sent! Check your Slack channel.');
      } else {
        toast.error('Test failed. Please check your webhook URL.');
      }
    } catch (error) {
      console.error('Error testing webhook:', error);
      toast.error('Failed to send test message');
    } finally {
      setIsTesting(false);
    }
  };

  const toggleIntegration = async () => {
    if (!integration) return;

    try {
      await api.patch('/slack/toggle', {
        enabled: !integration.enabled
      });
      setIntegration(prev => prev ? { ...prev, enabled: !prev.enabled } : null);
      toast.success(`Slack integration ${!integration.enabled ? 'enabled' : 'disabled'}`);
    } catch (error) {
      console.error('Error toggling integration:', error);
      toast.error('Failed to toggle integration');
    }
  };

  const deleteIntegration = async () => {
    if (!confirm('Are you sure you want to delete this Slack integration?')) {
      return;
    }

    try {
      await api.delete('/slack/integration');
      setIntegration(null);
      setFormData({
        workspaceName: '',
        webhookUrl: '',
        channel: ''
      });
      toast.success('Slack integration deleted');
    } catch (error) {
      console.error('Error deleting integration:', error);
      toast.error('Failed to delete integration');
    }
  };

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading Slack integration...</p>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <MessageSquare className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Slack Integration</h2>
                <p className="text-gray-600 mt-1">
                  Connect your Slack workspace to receive notifications
                </p>
              </div>
            </div>
            {integration && (
              <div className="flex items-center gap-2">
                <span className={`text-sm ${integration.enabled ? 'text-green-600' : 'text-gray-500'}`}>
                  {integration.enabled ? 'Active' : 'Inactive'}
                </span>
                <button
                  onClick={toggleIntegration}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    integration.enabled ? 'bg-green-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      integration.enabled ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Setup Instructions */}
        <div className="p-6 bg-blue-50 border-b border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-2">How to set up:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-blue-800">
            <li>Go to your Slack workspace settings</li>
            <li>Create an Incoming Webhook for your desired channel</li>
            <li>Copy the webhook URL and paste it below</li>
            <li>Save and test the connection</li>
          </ol>
          <a
            href="https://api.slack.com/messaging/webhooks"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-700 mt-2"
          >
            Learn more about Slack webhooks
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>

        {/* Form */}
        <div className="p-6 space-y-4">
          {/* Workspace Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Workspace Name
            </label>
            <input
              type="text"
              value={formData.workspaceName}
              onChange={(e) => setFormData({ ...formData, workspaceName: e.target.value })}
              placeholder="e.g., My Company Workspace"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          {/* Webhook URL */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Webhook URL
            </label>
            <input
              type="url"
              value={formData.webhookUrl}
              onChange={(e) => setFormData({ ...formData, webhookUrl: e.target.value })}
              placeholder="https://hooks.slack.com/services/..."
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent font-mono text-sm"
            />
          </div>

          {/* Channel */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Channel
            </label>
            <input
              type="text"
              value={formData.channel}
              onChange={(e) => setFormData({ ...formData, channel: e.target.value })}
              placeholder="e.g., #notifications"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>

          {/* Test Connection */}
          <div className="pt-2">
            <button
              onClick={testWebhook}
              disabled={isTesting || !formData.webhookUrl}
              className="w-full px-4 py-2 border border-purple-300 text-purple-600 rounded-lg hover:bg-purple-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              <TestTube className="h-4 w-4" />
              {isTesting ? 'Sending test message...' : 'Send Test Message'}
            </button>
          </div>
        </div>

        {/* Integration Status */}
        {integration && (
          <div className="px-6 pb-4">
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-start gap-3"
            >
              <Check className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="font-medium text-green-900">Integration Active</h4>
                <p className="text-sm text-green-700 mt-1">
                  Connected to <strong>{integration.workspaceName}</strong> on{' '}
                  {new Date(integration.createdAt).toLocaleDateString()}
                </p>
              </div>
            </motion.div>
          </div>
        )}

        {/* Actions */}
        <div className="p-6 border-t border-gray-200 flex justify-between">
          <div>
            {integration && (
              <button
                onClick={deleteIntegration}
                className="px-4 py-2 border border-red-300 text-red-600 rounded-lg hover:bg-red-50 transition-colors flex items-center gap-2"
              >
                <Trash2 className="h-4 w-4" />
                Delete Integration
              </button>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={fetchIntegration}
              className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={saveIntegration}
              disabled={isSaving}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              {isSaving ? 'Saving...' : 'Save Integration'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SlackSettings;
