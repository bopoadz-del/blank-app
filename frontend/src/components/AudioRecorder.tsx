import React, { useState, useRef, useEffect } from 'react';
import { FiMic, FiSquare, FiTrash2, FiCheck } from 'react-icons/fi';
import toast from 'react-hot-toast';

interface AudioRecorderProps {
  onRecord: (file: File) => void;
  onCancel: () => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onRecord, onCancel }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioURL, setAudioURL] = useState<string>('');
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        setAudioBlob(blob);
      };

      mediaRecorder.start();
      setIsRecording(true);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      toast.success('Recording started');
    } catch (error) {
      toast.error('Failed to access microphone. Please check permissions.');
      console.error('Microphone error:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        timerRef.current = setInterval(() => {
          setRecordingTime((prev) => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }
      }
    }
  };

  const deleteRecording = () => {
    setAudioURL('');
    setAudioBlob(null);
    setRecordingTime(0);
    chunksRef.current = [];
  };

  const confirmRecording = () => {
    if (audioBlob) {
      const file = new File([audioBlob], `audio-${Date.now()}.webm`, { type: 'audio/webm' });
      onRecord(file);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md">
        <h3 className="text-xl font-semibold mb-6 text-center">Audio Recording</h3>

        {/* Waveform Visualization */}
        <div className="mb-6 h-24 bg-gradient-to-r from-primary-100 to-primary-200 rounded-lg flex items-center justify-center">
          {isRecording && !isPaused ? (
            <div className="flex items-center gap-1">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="w-1 bg-primary-600 rounded-full animate-pulse"
                  style={{
                    height: `${Math.random() * 60 + 20}px`,
                    animationDelay: `${i * 0.05}s`,
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="text-gray-400">
              {audioURL ? 'Recording complete' : 'Ready to record'}
            </div>
          )}
        </div>

        {/* Timer */}
        <div className="text-center mb-6">
          <div className="text-3xl font-mono font-semibold text-gray-900">
            {formatTime(recordingTime)}
          </div>
          {isRecording && (
            <div className="text-sm text-gray-500 mt-2">
              {isPaused ? 'Paused' : 'Recording...'}
            </div>
          )}
        </div>

        {/* Audio Playback */}
        {audioURL && !isRecording && (
          <div className="mb-6">
            <audio src={audioURL} controls className="w-full" />
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-4">
          {!isRecording && !audioURL && (
            <>
              <button onClick={onCancel} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={startRecording}
                className="btn-primary flex items-center gap-2"
              >
                <FiMic className="w-5 h-5" />
                Start Recording
              </button>
            </>
          )}

          {isRecording && (
            <>
              <button
                onClick={pauseRecording}
                className="p-4 rounded-full bg-yellow-500 hover:bg-yellow-600 text-white transition-colors"
              >
                {isPaused ? (
                  <FiMic className="w-6 h-6" />
                ) : (
                  <div className="w-6 h-6 flex items-center justify-center">
                    <div className="w-4 h-4 bg-white rounded-sm" />
                  </div>
                )}
              </button>
              <button
                onClick={stopRecording}
                className="p-4 rounded-full bg-red-500 hover:bg-red-600 text-white transition-colors"
              >
                <FiSquare className="w-6 h-6" />
              </button>
            </>
          )}

          {audioURL && !isRecording && (
            <>
              <button
                onClick={deleteRecording}
                className="p-4 rounded-full bg-gray-200 hover:bg-gray-300 transition-colors"
              >
                <FiTrash2 className="w-6 h-6 text-gray-700" />
              </button>
              <button onClick={onCancel} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={confirmRecording}
                className="btn-primary flex items-center gap-2"
              >
                <FiCheck className="w-5 h-5" />
                Use Recording
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
