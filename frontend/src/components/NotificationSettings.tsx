import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Bell, Mail, Smartphone, MessageSquare, Save } from 'lucide-react';
import api from '../services/api';
import { toast } from 'sonner';

interface NotificationPreferences {
  emailEnabled: boolean;
  pushEnabled: boolean;
  slackEnabled: boolean;
  inAppEnabled: boolean;
}

const NotificationSettings: React.FC = () => {
  const [preferences, setPreferences] = useState<NotificationPreferences>({
    emailEnabled: true,
    pushEnabled: false,
    slackEnabled: false,
    inAppEnabled: true
  });
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    fetchPreferences();
  }, []);

  const fetchPreferences = async () => {
    setIsLoading(true);
    try {
      const response = await api.get('/notifications/preferences');
      setPreferences(response.data);
    } catch (error) {
      console.error('Error fetching preferences:', error);
      toast.error('Failed to load notification preferences');
    } finally {
      setIsLoading(false);
    }
  };

  const updatePreferences = async () => {
    setIsSaving(true);
    try {
      await api.patch('/notifications/preferences', preferences);
      toast.success('Notification preferences updated');
    } catch (error) {
      console.error('Error updating preferences:', error);
      toast.error('Failed to update preferences');
    } finally {
      setIsSaving(false);
    }
  };

  const togglePreference = (key: keyof NotificationPreferences) => {
    setPreferences(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const notificationChannels = [
    {
      key: 'inAppEnabled' as keyof NotificationPreferences,
      label: 'In-App Notifications',
      description: 'Show notifications within the application',
      icon: Bell,
      color: 'text-blue-600'
    },
    {
      key: 'emailEnabled' as keyof NotificationPreferences,
      label: 'Email Notifications',
      description: 'Receive notifications via email',
      icon: Mail,
      color: 'text-green-600'
    },
    {
      key: 'pushEnabled' as keyof NotificationPreferences,
      label: 'Push Notifications',
      description: 'Get push notifications on your device',
      icon: Smartphone,
      color: 'text-purple-600'
    },
    {
      key: 'slackEnabled' as keyof NotificationPreferences,
      label: 'Slack Notifications',
      description: 'Send notifications to your Slack workspace',
      icon: MessageSquare,
      color: 'text-pink-600'
    }
  ];

  if (isLoading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent mx-auto"></div>
        <p className="mt-4 text-gray-600">Loading preferences...</p>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Bell className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Notification Settings</h2>
              <p className="text-gray-600 mt-1">
                Manage how you receive notifications
              </p>
            </div>
          </div>
        </div>

        {/* Notification Channels */}
        <div className="p-6 space-y-4">
          {notificationChannels.map((channel) => {
            const Icon = channel.icon;
            return (
              <motion.div
                key={channel.key}
                whileHover={{ scale: 1.01 }}
                className="flex items-center justify-between p-4 rounded-lg border border-gray-200 hover:border-blue-300 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className={`p-2 bg-gray-50 rounded-lg ${channel.color}`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-medium text-gray-900">{channel.label}</h3>
                    <p className="text-sm text-gray-600 mt-1">{channel.description}</p>
                  </div>
                </div>

                {/* Toggle Switch */}
                <button
                  onClick={() => togglePreference(channel.key)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    preferences[channel.key] ? 'bg-blue-600' : 'bg-gray-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      preferences[channel.key] ? 'translate-x-6' : 'translate-x-1'
                    }`}
                  />
                </button>
              </motion.div>
            );
          })}
        </div>

        {/* Note */}
        <div className="px-6 pb-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> To receive Slack notifications, you need to configure your Slack integration in the Integrations section.
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="p-6 border-t border-gray-200 flex justify-end gap-3">
          <button
            onClick={fetchPreferences}
            className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
          <button
            onClick={updatePreferences}
            disabled={isSaving}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Save className="h-4 w-4" />
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default NotificationSettings;
