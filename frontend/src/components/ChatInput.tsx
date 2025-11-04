import React, { useState, useRef, useEffect } from 'react';
import {
  FiSend,
  FiPaperclip,
  FiCamera,
  FiMic,
  FiGlobe,
  FiX,
  FiImage,
  FiFile,
} from 'react-icons/fi';
import { CameraCapture } from './CameraCapture';
import { AudioRecorder } from './AudioRecorder';
import toast from 'react-hot-toast';

interface ChatInputProps {
  onSendMessage: (message: string, files?: File[], useInternet?: boolean) => void;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [showCamera, setShowCamera] = useState(false);
  const [showAudioRecorder, setShowAudioRecorder] = useState(false);
  const [showFileOptions, setShowFileOptions] = useState(false);
  const [useInternet, setUseInternet] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    adjustTextareaHeight();
  }, [message]);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() && attachedFiles.length === 0) return;

    onSendMessage(message, attachedFiles.length > 0 ? attachedFiles : undefined, useInternet);
    setMessage('');
    setAttachedFiles([]);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleCameraCapture = (file: File, type: 'photo' | 'video') => {
    setAttachedFiles((prev) => [...prev, file]);
    setShowCamera(false);
    toast.success(`${type === 'photo' ? 'Photo' : 'Video'} captured`);
  };

  const handleAudioRecord = (file: File) => {
    setAttachedFiles((prev) => [...prev, file]);
    setShowAudioRecorder(false);
    toast.success('Audio recorded');
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setAttachedFiles((prev) => [...prev, ...files]);
      toast.success(`${files.length} file(s) attached`);
    }
    setShowFileOptions(false);
  };

  const removeFile = (index: number) => {
    setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      setAttachedFiles((prev) => [...prev, ...files]);
      toast.success(`${files.length} file(s) attached`);
    }
  };

  const getFilePreview = (file: File) => {
    if (file.type.startsWith('image/')) {
      return URL.createObjectURL(file);
    }
    return null;
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('image/')) return <FiImage className="w-4 h-4" />;
    return <FiFile className="w-4 h-4" />;
  };

  return (
    <>
      <div className="border-t border-gray-200 bg-white">
        <div className="max-w-4xl mx-auto p-4">
          {/* Attached Files Preview */}
          {attachedFiles.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {attachedFiles.map((file, index) => {
                const preview = getFilePreview(file);
                return (
                  <div
                    key={index}
                    className="relative group bg-gray-100 rounded-lg p-2 flex items-center gap-2 max-w-xs"
                  >
                    {preview ? (
                      <img
                        src={preview}
                        alt={file.name}
                        className="w-12 h-12 object-cover rounded"
                      />
                    ) : (
                      <div className="w-12 h-12 bg-gray-200 rounded flex items-center justify-center text-gray-500">
                        {getFileIcon(file)}
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium text-gray-700 truncate">{file.name}</p>
                      <p className="text-xs text-gray-500">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <FiX className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {/* Internet Toggle */}
          <div className="mb-2 flex items-center gap-2">
            <button
              onClick={() => setUseInternet(!useInternet)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                useInternet
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <FiGlobe className="w-4 h-4" />
              {useInternet ? 'Internet: ON' : 'Internet: OFF'}
            </button>
            {useInternet && (
              <span className="text-xs text-gray-500">
                AI can search the web for information
              </span>
            )}
          </div>

          {/* Input Area */}
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative rounded-2xl border-2 transition-all ${
              isDragging
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-200 bg-gray-50'
            }`}
          >
            {isDragging && (
              <div className="absolute inset-0 flex items-center justify-center bg-primary-50/90 rounded-2xl z-10">
                <div className="text-center">
                  <FiPaperclip className="w-12 h-12 mx-auto mb-2 text-primary-600" />
                  <p className="text-lg font-medium text-primary-700">Drop files here</p>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex items-end gap-2 p-2">
              {/* Action Buttons */}
              <div className="flex items-center gap-1 pb-2">
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setShowFileOptions(!showFileOptions)}
                    className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
                    disabled={disabled}
                  >
                    <FiPaperclip className="w-5 h-5" />
                  </button>

                  {showFileOptions && (
                    <div className="absolute bottom-full left-0 mb-2 bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-[180px] z-20">
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-50 flex items-center gap-2"
                      >
                        <FiPaperclip className="w-4 h-4" />
                        Upload Files
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setShowCamera(true);
                          setShowFileOptions(false);
                        }}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-50 flex items-center gap-2"
                      >
                        <FiCamera className="w-4 h-4" />
                        Take Photo/Video
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setShowAudioRecorder(true);
                          setShowFileOptions(false);
                        }}
                        className="w-full px-4 py-2 text-left text-sm hover:bg-gray-50 flex items-center gap-2"
                      >
                        <FiMic className="w-4 h-4" />
                        Record Audio
                      </button>
                    </div>
                  )}
                </div>

                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                  accept="*/*"
                />

                <button
                  type="button"
                  onClick={() => setShowCamera(true)}
                  className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
                  disabled={disabled}
                >
                  <FiCamera className="w-5 h-5" />
                </button>

                <button
                  type="button"
                  onClick={() => setShowAudioRecorder(true)}
                  className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
                  disabled={disabled}
                >
                  <FiMic className="w-5 h-5" />
                </button>
              </div>

              {/* Text Input */}
              <textarea
                ref={textareaRef}
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message... (Shift+Enter for new line)"
                disabled={disabled}
                className="flex-1 bg-transparent border-none resize-none outline-none px-2 py-2 text-gray-900 placeholder-gray-400 max-h-[200px]"
                rows={1}
              />

              {/* Send Button */}
              <button
                type="submit"
                disabled={disabled || (!message.trim() && attachedFiles.length === 0)}
                className="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors mb-1"
              >
                <FiSend className="w-5 h-5" />
              </button>
            </form>
          </div>

          <p className="text-xs text-gray-400 mt-2 text-center">
            Supports text, images, videos, audio, documents, and more
          </p>
        </div>
      </div>

      {/* Modals */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}

      {showAudioRecorder && (
        <AudioRecorder
          onRecord={handleAudioRecord}
          onCancel={() => setShowAudioRecorder(false)}
        />
      )}
    </>
  );
};
