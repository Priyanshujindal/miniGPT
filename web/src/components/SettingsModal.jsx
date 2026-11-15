import React from 'react';
import { X, RotateCcw, Sliders, Cpu, Info } from 'lucide-react';

export const DEFAULT_SETTINGS = {
  model_version: 'v4',
  temperature: 0.7,
  max_new_tokens: 80,
  top_k: 40,
  repetition_penalty: 1.15,
};

export default function SettingsModal({ isOpen, onClose, settings, onSave, onReset }) {
  if (!isOpen) return null;

  const handleChange = (key, value) => {
    onSave({
      ...settings,
      [key]: value,
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
      <div
        className="w-full max-w-md bg-[#161618] border border-neutral-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-neutral-800/80 bg-[#1c1c20]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-neutral-800 flex items-center justify-center text-neutral-300">
              <Sliders className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-neutral-100">Generation Parameters</h2>
              <p className="text-xs text-neutral-400">MiniGPT V3 Autoregressive Sampling</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Sliders body */}
        <div className="p-5 space-y-5 overflow-y-auto max-h-[75vh]">
          {/* Model Selector */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 text-neutral-300 font-medium">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span>Target Model Architecture</span>
              </div>
              <span className="text-[11px] text-neutral-400 font-mono">
                {settings.model_version === 'v4' ? '2053 Vocab' : '2049 Vocab'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 p-1 bg-neutral-900/90 border border-neutral-800 rounded-xl">
              <button
                type="button"
                onClick={() => handleChange('model_version', 'v4')}
                className={`flex flex-col items-center justify-center py-2 px-3 rounded-lg text-xs font-medium transition-all ${
                  settings.model_version === 'v4' || !settings.model_version
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
                }`}
              >
                <span className="font-semibold">MiniGPT V4</span>
                <span className="text-[10px] text-neutral-400">Conversational Q&A</span>
              </button>

              <button
                type="button"
                onClick={() => handleChange('model_version', 'v3')}
                className={`flex flex-col items-center justify-center py-2 px-3 rounded-lg text-xs font-medium transition-all ${
                  settings.model_version === 'v3'
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
                }`}
              >
                <span className="font-semibold">MiniGPT V3</span>
                <span className="text-[10px] text-neutral-400">TinyStories Storyteller</span>
              </button>
            </div>
            <p className="text-[11px] text-neutral-400">
              {settings.model_version === 'v4' || !settings.model_version
                ? 'MiniGPT V4 is instruction-tuned with dedicated conversational tokens to answer questions and follow prompts.'
                : 'MiniGPT V3 generates continuous creative children stories from prompts.'}
            </p>
          </div>

          {/* Temperature */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <label className="font-medium text-neutral-200">Temperature</label>
              <span className="font-mono text-emerald-400 font-semibold">{settings.temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              value={settings.temperature}
              onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
            />
            <p className="text-[11px] text-neutral-400">
              Controls randomness. Lower values (0.2) make output predictable; higher values (1.2) increase creativity.
            </p>
          </div>

          {/* Max New Tokens */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <label className="font-medium text-neutral-200">Maximum New Tokens</label>
              <span className="font-mono text-emerald-400 font-semibold">{settings.max_new_tokens}</span>
            </div>
            <input
              type="range"
              min="20"
              max="128"
              step="4"
              value={settings.max_new_tokens}
              onChange={(e) => handleChange('max_new_tokens', parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
            />
            <p className="text-[11px] text-neutral-400">
              Maximum tokens to generate (20–128). Context window total limit is 128 tokens.
            </p>
          </div>

          {/* Top-K */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <label className="font-medium text-neutral-200">Top-K Filtering</label>
              <span className="font-mono text-emerald-400 font-semibold">{settings.top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="100"
              step="1"
              value={settings.top_k}
              onChange={(e) => handleChange('top_k', parseInt(e.target.value, 10))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
            />
            <p className="text-[11px] text-neutral-400">
              Restricts sampling to the top K most probable tokens at each autoregressive step.
            </p>
          </div>

          {/* Repetition Penalty */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center text-xs">
              <label className="font-medium text-neutral-200">Repetition Penalty</label>
              <span className="font-mono text-emerald-400 font-semibold">{settings.repetition_penalty.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="1.0"
              max="1.5"
              step="0.05"
              value={settings.repetition_penalty}
              onChange={(e) => handleChange('repetition_penalty', parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-neutral-800 rounded-lg appearance-none"
            />
            <p className="text-[11px] text-neutral-400">
              Penalizes previously generated tokens to prevent repetitive loops (1.0 = no penalty, 1.15 = balanced).
            </p>
          </div>

          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 flex gap-2.5 text-xs text-amber-300">
            <Info className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
            <p className="leading-relaxed text-[11px]">
              MiniGPT V3 was trained on the TinyStories dataset. Temperature between 0.7–0.9 and repetition penalty around 1.15 produce the most coherent stories.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3.5 border-t border-neutral-800/80 bg-[#19191d]">
          <button
            onClick={onReset}
            className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-200 transition-colors py-1.5 px-3 rounded-lg hover:bg-neutral-800"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>
          <button
            onClick={onClose}
            className="text-xs font-medium bg-neutral-100 hover:bg-white text-neutral-900 px-4 py-2 rounded-lg transition-colors font-semibold"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

