import React, { useState } from 'react';
import { Bot, User, Copy, Check, RotateCw, AlertCircle } from 'lucide-react';
import MarkdownRenderer from './MarkdownRenderer.jsx';

export default function ChatMessage({
  message,
  onRegenerate,
  isLastAssistantMessage = false,
  isGenerating = false,
}) {
  const [copied, setCopied] = useState(false);
  const isAssistant = message.role === 'assistant';

  const handleCopy = () => {
    if (!message.content) return;
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formattedTime = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div
      className={`py-5 px-4 sm:px-6 w-full transition-colors ${
        isAssistant ? 'bg-[#121214]/60' : 'bg-transparent'
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div className="shrink-0 mt-0.5">
          {isAssistant ? (
            <div className="w-8 h-8 rounded-xl bg-neutral-800 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm">
              <Bot className="w-4 h-4" />
            </div>
          ) : (
            <div className="w-8 h-8 rounded-xl bg-neutral-700/60 border border-neutral-600/40 flex items-center justify-center text-neutral-300">
              <User className="w-4 h-4" />
            </div>
          )}
        </div>

        {/* Content area */}
        <div className="flex-1 min-w-0 space-y-1.5">
          {/* Header */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-neutral-200">
              {isAssistant ? (message.model || 'MiniGPT V4') : 'You'}
            </span>
            {isAssistant && (
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-400 border border-neutral-700/50">
                checkpoint: best.pt
              </span>
            )}
            {formattedTime && (
              <span className="text-[11px] text-neutral-500 ml-auto">
                {formattedTime}
              </span>
            )}
          </div>

          {/* Error Message */}
          {message.error ? (
            <div className="p-3.5 rounded-xl bg-red-950/40 border border-red-800/60 text-red-200 text-xs flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold text-red-300">Generation Error</p>
                <p className="leading-relaxed opacity-90">{message.error}</p>
              </div>
            </div>
          ) : isAssistant ? (
            /* Assistant Markdown Message */
            <div className="prose-invert">
              <MarkdownRenderer content={message.content} />
            </div>
          ) : (
            /* User Message text */
            <p className="text-[14.5px] leading-relaxed text-neutral-100 whitespace-pre-wrap">
              {message.content}
            </p>
          )}

          {/* Action buttons (only for assistant responses) */}
          {isAssistant && !message.error && (
            <div className="flex items-center gap-2 pt-2 text-xs text-neutral-400">
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 py-1 px-2 rounded-md hover:bg-neutral-800 hover:text-neutral-200 transition-colors"
                title="Copy response"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400 text-[11px]">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span className="text-[11px]">Copy</span>
                  </>
                )}
              </button>

              {isLastAssistantMessage && onRegenerate && (
                <button
                  onClick={onRegenerate}
                  disabled={isGenerating}
                  className="flex items-center gap-1.5 py-1 px-2 rounded-md hover:bg-neutral-800 hover:text-neutral-200 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  title="Regenerate with current settings"
                >
                  <RotateCw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
                  <span className="text-[11px]">Regenerate</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

