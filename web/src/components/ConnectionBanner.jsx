import React from 'react';
import { RefreshCw, Wifi, WifiOff, AlertCircle } from 'lucide-react';

export default function ConnectionBanner({ backendStatus, onRetry, isChecking }) {
  if (!backendStatus) return null;

  if (backendStatus.success && backendStatus.data?.model_loaded) {
    return (
      <div className="flex items-center gap-2 text-xs text-neutral-400">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span>
        <span className="hidden sm:inline font-medium text-neutral-300">
          MiniGPT V3 Online
        </span>
        <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-neutral-800 text-neutral-300 border border-neutral-700/60">
          {backendStatus.data.device || 'CUDA'}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="relative flex h-2 w-2">
        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
      </span>
      <span className="text-red-400 font-medium">
        Backend Offline
      </span>
      <button
        onClick={onRetry}
        disabled={isChecking}
        className="p-1 rounded hover:bg-neutral-800 text-neutral-400 hover:text-white transition-colors disabled:opacity-50"
        title="Retry connection"
      >
        <RefreshCw className={`w-3 h-3 ${isChecking ? 'animate-spin' : ''}`} />
      </button>
    </div>
  );
}

