import React, { useState } from 'react';
import {
  FiFolder,
  FiPlus,
  FiEdit2,
  FiTrash2,
  FiChevronDown,
  FiChevronRight,
  FiMessageSquare,
} from 'react-icons/fi';
import type { Project } from '../types';

interface ProjectManagerProps {
  projects: Project[];
  currentProjectId: string | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (name: string, description?: string) => void;
  onDeleteProject: (projectId: string) => void;
  onRenameProject: (projectId: string, newName: string) => void;
  onSelectConversation: (conversationId: string) => void;
}

export const ProjectManager: React.FC<ProjectManagerProps> = ({
  projects,
  currentProjectId,
  onSelectProject,
  onCreateProject,
  onDeleteProject,
  onRenameProject,
  onSelectConversation,
}) => {
  const [expandedProjects, setExpandedProjects] = useState<Set<string>>(new Set());
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingProjectId, setEditingProjectId] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');

  const toggleProject = (projectId: string) => {
    const newExpanded = new Set(expandedProjects);
    if (newExpanded.has(projectId)) {
      newExpanded.delete(projectId);
    } else {
      newExpanded.add(projectId);
    }
    setExpandedProjects(newExpanded);
  };

  const handleCreateProject = () => {
    if (newProjectName.trim()) {
      onCreateProject(newProjectName.trim(), newProjectDescription.trim() || undefined);
      setNewProjectName('');
      setNewProjectDescription('');
      setShowCreateModal(false);
    }
  };

  const handleRename = (projectId: string) => {
    const project = projects.find((p) => p.id === projectId);
    if (project && newProjectName.trim()) {
      onRenameProject(projectId, newProjectName.trim());
      setEditingProjectId(null);
      setNewProjectName('');
    }
  };

  const getProjectColor = (color?: string) => {
    const colors: Record<string, string> = {
      blue: 'bg-blue-500',
      green: 'bg-green-500',
      purple: 'bg-purple-500',
      red: 'bg-red-500',
      yellow: 'bg-yellow-500',
      pink: 'bg-pink-500',
    };
    return colors[color || 'blue'] || 'bg-blue-500';
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold text-gray-700 uppercase">Projects</h2>
          <button
            onClick={() => setShowCreateModal(true)}
            className="p-1 hover:bg-gray-100 rounded transition-colors"
            title="Create new project"
          >
            <FiPlus className="w-5 h-5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Projects List */}
      <div className="flex-1 overflow-y-auto">
        {projects.length === 0 ? (
          <div className="p-4 text-center text-gray-500 text-sm">
            <FiFolder className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p>No projects yet</p>
            <p className="text-xs mt-1">Create your first project to get started</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {projects.map((project) => (
              <div key={project.id} className="group">
                <div
                  className={`flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors ${
                    currentProjectId === project.id
                      ? 'bg-primary-50 text-primary-700'
                      : 'hover:bg-gray-100'
                  }`}
                  onClick={() => onSelectProject(project.id)}
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleProject(project.id);
                    }}
                    className="p-1 hover:bg-gray-200 rounded"
                  >
                    {expandedProjects.has(project.id) ? (
                      <FiChevronDown className="w-4 h-4" />
                    ) : (
                      <FiChevronRight className="w-4 h-4" />
                    )}
                  </button>

                  <div className={`w-2 h-2 rounded-full ${getProjectColor(project.color)}`} />

                  {editingProjectId === project.id ? (
                    <input
                      type="text"
                      value={newProjectName}
                      onChange={(e) => setNewProjectName(e.target.value)}
                      onBlur={() => handleRename(project.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRename(project.id);
                        if (e.key === 'Escape') setEditingProjectId(null);
                      }}
                      className="flex-1 px-2 py-1 text-sm border border-primary-300 rounded focus:outline-none focus:ring-2 focus:ring-primary-500"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span className="flex-1 text-sm font-medium truncate">{project.name}</span>
                  )}

                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingProjectId(project.id);
                        setNewProjectName(project.name);
                      }}
                      className="p-1 hover:bg-gray-200 rounded"
                      title="Rename"
                    >
                      <FiEdit2 className="w-3 h-3 text-gray-600" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete project "${project.name}"?`)) {
                          onDeleteProject(project.id);
                        }
                      }}
                      className="p-1 hover:bg-red-100 rounded"
                      title="Delete"
                    >
                      <FiTrash2 className="w-3 h-3 text-red-600" />
                    </button>
                  </div>
                </div>

                {/* Conversations */}
                {expandedProjects.has(project.id) && (
                  <div className="ml-8 mt-1 space-y-1">
                    {project.conversations.length === 0 ? (
                      <div className="px-2 py-1 text-xs text-gray-400">No conversations</div>
                    ) : (
                      project.conversations.map((conv) => (
                        <button
                          key={conv.id}
                          onClick={() => onSelectConversation(conv.id)}
                          className="w-full flex items-center gap-2 px-2 py-1.5 text-left text-xs text-gray-600 hover:bg-gray-100 rounded transition-colors"
                        >
                          <FiMessageSquare className="w-3 h-3 flex-shrink-0" />
                          <span className="truncate">{conv.title}</span>
                          <span className="text-xs text-gray-400 ml-auto">
                            {conv.messages.length}
                          </span>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
            <div className="p-6">
              <h3 className="text-xl font-semibold text-gray-900 mb-4">Create New Project</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Name
                  </label>
                  <input
                    type="text"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    placeholder="My Project"
                    className="input-field"
                    autoFocus
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description (optional)
                  </label>
                  <textarea
                    value={newProjectDescription}
                    onChange={(e) => setNewProjectDescription(e.target.value)}
                    placeholder="Project description..."
                    rows={3}
                    className="input-field"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 mt-6">
                <button
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewProjectName('');
                    setNewProjectDescription('');
                  }}
                  className="btn-secondary flex-1"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateProject}
                  disabled={!newProjectName.trim()}
                  className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Create Project
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
