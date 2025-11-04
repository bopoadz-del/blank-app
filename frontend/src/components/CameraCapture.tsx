import React, { useRef, useState, useEffect } from 'react';
import { FiCamera, FiVideo, FiX } from 'react-icons/fi';
import toast from 'react-hot-toast';

interface CameraCaptureProps {
  onCapture: (file: File, type: 'photo' | 'video') => void;
  onClose: () => void;
}

export const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, onClose }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mode, setMode] = useState<'photo' | 'video'>('photo');

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 1280, height: 720 },
        audio: mode === 'video',
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      toast.error('Failed to access camera. Please check permissions.');
      console.error('Camera error:', error);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');

    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0);
      canvas.toBlob((blob) => {
        if (blob) {
          const file = new File([blob], `photo-${Date.now()}.jpg`, { type: 'image/jpeg' });
          onCapture(file, 'photo');
          stopCamera();
          onClose();
        }
      }, 'image/jpeg', 0.95);
    }
  };

  const startVideoRecording = () => {
    if (!stream) return;

    try {
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm',
      });

      mediaRecorderRef.current = mediaRecorder;
      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const file = new File([blob], `video-${Date.now()}.webm`, { type: 'video/webm' });
        onCapture(file, 'video');
        stopCamera();
        onClose();
      };

      mediaRecorder.start();
      setIsRecording(true);
      toast.success('Recording started');
    } catch (error) {
      toast.error('Failed to start recording');
      console.error('Recording error:', error);
    }
  };

  const stopVideoRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const switchMode = async () => {
    stopCamera();
    setMode(mode === 'photo' ? 'video' : 'photo');
    await startCamera();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gradient-to-b from-black/50 to-transparent p-4">
        <div className="flex items-center justify-between">
          <button
            onClick={onClose}
            className="p-2 rounded-full bg-white/10 hover:bg-white/20 transition-colors"
          >
            <FiX className="w-6 h-6 text-white" />
          </button>
          <div className="flex items-center gap-2 bg-white/10 rounded-full px-4 py-2">
            <button
              onClick={switchMode}
              className={`px-3 py-1 rounded-full transition-colors ${
                mode === 'photo' ? 'bg-white text-black' : 'text-white'
              }`}
            >
              Photo
            </button>
            <button
              onClick={switchMode}
              className={`px-3 py-1 rounded-full transition-colors ${
                mode === 'video' ? 'bg-white text-black' : 'text-white'
              }`}
            >
              Video
            </button>
          </div>
          <div className="w-10" />
        </div>
      </div>

      {/* Video Preview */}
      <div className="flex-1 flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted={mode === 'photo'}
          className="max-w-full max-h-full"
        />
      </div>

      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-8">
        <div className="flex items-center justify-center gap-8">
          {mode === 'photo' ? (
            <button
              onClick={capturePhoto}
              className="w-16 h-16 rounded-full bg-white hover:bg-gray-200 transition-colors flex items-center justify-center"
            >
              <FiCamera className="w-8 h-8 text-black" />
            </button>
          ) : (
            <button
              onClick={isRecording ? stopVideoRecording : startVideoRecording}
              className={`w-16 h-16 rounded-full transition-colors flex items-center justify-center ${
                isRecording
                  ? 'bg-red-500 hover:bg-red-600'
                  : 'bg-white hover:bg-gray-200'
              }`}
            >
              {isRecording ? (
                <div className="w-6 h-6 bg-white rounded-sm" />
              ) : (
                <FiVideo className="w-8 h-8 text-black" />
              )}
            </button>
          )}
        </div>
        {isRecording && (
          <div className="text-center mt-4">
            <span className="text-white text-sm bg-red-500 px-3 py-1 rounded-full">
              Recording...
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
