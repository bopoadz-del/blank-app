import React, { useState } from 'react';
import {
  FiFile,
  FiImage,
  FiVideo,
  FiMusic,
  FiFileText,
  FiArchive,
  FiDownload,
  FiMaximize2,
  FiX,
} from 'react-icons/fi';

interface FilePreviewProps {
  file: File;
  onClose: () => void;
}

export const FilePreview: React.FC<FilePreviewProps> = ({ file, onClose }) => {
  const [isFullscreen, setIsFullscreen] = useState(false);

  const getFileType = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    const type = file.type.split('/')[0];

    // Zipped files
    if (
      ext === 'zip' ||
      ext === 'rar' ||
      ext === '7z' ||
      ext === 'tar' ||
      ext === 'gz' ||
      file.type.includes('zip') ||
      file.type.includes('compressed')
    ) {
      return 'archive';
    }

    // XER files (Primavera P6)
    if (ext === 'xer' || ext === 'xml' || ext === 'mpp') {
      return 'project';
    }

    // CAD files
    if (
      ext === 'dwg' ||
      ext === 'dxf' ||
      ext === 'dwf' ||
      ext === 'dgn' ||
      ext === 'rvt' ||
      ext === 'ifc' ||
      ext === 'step' ||
      ext === 'stp' ||
      ext === 'iges' ||
      ext === 'igs' ||
      ext === 'stl' ||
      ext === '3dm'
    ) {
      return 'cad';
    }

    // Standard types
    if (type === 'image') return 'image';
    if (type === 'video') return 'video';
    if (type === 'audio') return 'audio';
    if (ext === 'pdf') return 'pdf';
    if (
      ext === 'doc' ||
      ext === 'docx' ||
      ext === 'txt' ||
      ext === 'rtf' ||
      type === 'text'
    ) {
      return 'document';
    }

    return 'unknown';
  };

  const getFileIcon = (fileType: string) => {
    switch (fileType) {
      case 'image':
        return <FiImage className="w-16 h-16" />;
      case 'video':
        return <FiVideo className="w-16 h-16" />;
      case 'audio':
        return <FiMusic className="w-16 h-16" />;
      case 'archive':
        return <FiArchive className="w-16 h-16" />;
      case 'document':
      case 'pdf':
        return <FiFileText className="w-16 h-16" />;
      default:
        return <FiFile className="w-16 h-16" />;
    }
  };

  const fileType = getFileType(file);
  const fileUrl = URL.createObjectURL(file);

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = fileUrl;
    a.download = file.name;
    a.click();
  };

  const renderFileContent = () => {
    switch (fileType) {
      case 'image':
        return (
          <img
            src={fileUrl}
            alt={file.name}
            className="max-w-full max-h-[600px] object-contain mx-auto"
          />
        );

      case 'video':
        return (
          <video
            src={fileUrl}
            controls
            className="max-w-full max-h-[600px] mx-auto"
          />
        );

      case 'audio':
        return (
          <div className="flex flex-col items-center gap-4 p-8">
            <FiMusic className="w-24 h-24 text-primary-600" />
            <audio src={fileUrl} controls className="w-full max-w-md" />
          </div>
        );

      case 'pdf':
        return (
          <iframe
            src={fileUrl}
            className="w-full h-[600px] border-0"
            title={file.name}
          />
        );

      case 'archive':
        return (
          <div className="flex flex-col items-center gap-4 p-8 text-center">
            <FiArchive className="w-24 h-24 text-yellow-600" />
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Compressed Archive
              </h3>
              <p className="text-gray-600 mb-4">
                {file.name}
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Size: {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md">
                <p className="text-sm text-blue-800">
                  📦 Supported formats: ZIP, RAR, 7Z, TAR, GZ
                </p>
                <p className="text-xs text-blue-600 mt-2">
                  Download to extract and view contents
                </p>
              </div>
            </div>
          </div>
        );

      case 'project':
        return (
          <div className="flex flex-col items-center gap-4 p-8 text-center">
            <div className="w-24 h-24 bg-green-100 rounded-lg flex items-center justify-center">
              <span className="text-3xl font-bold text-green-600">XER</span>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Project Schedule File
              </h3>
              <p className="text-gray-600 mb-4">
                {file.name}
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Size: {(file.size / 1024).toFixed(2)} KB
              </p>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-md">
                <p className="text-sm text-green-800 font-medium mb-2">
                  🗓️ Primavera P6 / MS Project File
                </p>
                <p className="text-xs text-green-600">
                  Supported formats: XER, XML, MPP
                </p>
                <p className="text-xs text-green-600 mt-1">
                  Send to AI for schedule analysis, critical path identification, and resource optimization
                </p>
              </div>
            </div>
          </div>
        );

      case 'cad':
        return (
          <div className="flex flex-col items-center gap-4 p-8 text-center">
            <div className="w-24 h-24 bg-purple-100 rounded-lg flex items-center justify-center">
              <span className="text-3xl font-bold text-purple-600">CAD</span>
            </div>
            <div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                CAD Drawing File
              </h3>
              <p className="text-gray-600 mb-4">
                {file.name}
              </p>
              <p className="text-sm text-gray-500 mb-4">
                Size: {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 max-w-md">
                <p className="text-sm text-purple-800 font-medium mb-2">
                  📐 Supported CAD Formats:
                </p>
                <div className="text-xs text-purple-600 space-y-1">
                  <p>• AutoCAD: DWG, DXF, DWF</p>
                  <p>• Revit: RVT, IFC</p>
                  <p>• MicroStation: DGN</p>
                  <p>• 3D Models: STEP, IGES, STL, 3DM</p>
                </div>
                <p className="text-xs text-purple-600 mt-3">
                  Send to AI for design analysis, clash detection, and quantity takeoff
                </p>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="flex flex-col items-center gap-4 p-8">
            {getFileIcon(fileType)}
            <div className="text-center">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {file.name}
              </h3>
              <p className="text-sm text-gray-500">
                {file.type || 'Unknown type'} • {(file.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
        );
    }
  };

  return (
    <div
      className={`fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 ${
        isFullscreen ? 'p-0' : ''
      }`}
    >
      <div
        className={`bg-white rounded-lg shadow-2xl overflow-hidden ${
          isFullscreen ? 'w-full h-full' : 'max-w-4xl w-full max-h-[90vh]'
        }`}
      >
        {/* Header */}
        <div className="bg-gray-100 px-4 py-3 flex items-center justify-between border-b">
          <h2 className="text-lg font-semibold text-gray-900 truncate flex-1 mr-4">
            {file.name}
          </h2>
          <div className="flex items-center gap-2">
            <button
              onClick={handleDownload}
              className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
              title="Download"
            >
              <FiDownload className="w-5 h-5 text-gray-700" />
            </button>
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
              title="Toggle fullscreen"
            >
              <FiMaximize2 className="w-5 h-5 text-gray-700" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-200 rounded-lg transition-colors"
              title="Close"
            >
              <FiX className="w-5 h-5 text-gray-700" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className={`overflow-auto ${isFullscreen ? 'h-[calc(100vh-60px)]' : 'max-h-[calc(90vh-60px)]'}`}>
          {renderFileContent()}
        </div>
      </div>
    </div>
  );
};
