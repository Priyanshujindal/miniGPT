import React, { useRef, useEffect } from 'react';
import { ArrowUp, Loader2, Sliders } from 'lucide-react';

export default function MessageComposer({
  input,
  setInput,
  onSend,
  isGenerating,
  onOpenSettings,
  settings,
}) {
  const textareaRef = useRef(null);

  // Auto-resize textarea based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && input.trim()) {
        onSend();
      }
    }
  };

  const isSendDisabled = isGenerating || !input.trim();

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4">
      <div className="relative rounded-2xl bg-[#17171a] border border-neutral-800/80 focus-within:border-neutral-700 shadow-xl transition-all">
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Message MiniGPT V3 (e.g., 'Once upon a time...')"
          rows={1}
          disabled={isGenerating}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-10 text-sm text-neutral-100 placeholder:text-neutral-500 focus:outline-none disabled:opacity-50 min-h-[52px] max-h-[180px]"
        />

        {/* Bottom controls row inside composer */}
        <div className="absolute bottom-2.5 left-3.5 right-2.5 flex items-center justify-between pointer-events-none">
          {/* Quick info / settings button */}
          <div className="pointer-events-auto flex items-center gap-2">
            <button
              onClick={onOpenSettings}
              className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/80 transition-colors"
              title="Open Generation Settings"
            >
              <Sliders className="w-3 h-3 text-emerald-400" />
              <span>T: {settings.temperature.toFixed(2)}</span>
              <span className="text-neutral-600">•</span>
              <span>{settings.max_new_tokens}t</span>
            </button>
          </div>

          {/* Send / Stop button */}
          <div className="pointer-events-auto">
            <button
              onClick={onSend}
              disabled={isSendDisabled}
              className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
                isSendDisabled
                  ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
                  : 'bg-white text-black hover:bg-neutral-200 shadow-sm cursor-pointer scale-100 active:scale-95'
              }`}
              title={isGenerating ? 'Generating...' : 'Send message (Enter)'}
            >
              {isGenerating ? (
                <Loader2 className="w-4 h-4 animate-spin text-neutral-400" />
              ) : (
                <ArrowUp className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between px-2 pt-2 text-[11px] text-neutral-500">
        <span>Press <kbd className="px-1 py-0.5 rounded bg-neutral-800/80 text-neutral-400 font-mono text-[10px]">Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded bg-neutral-800/80 text-neutral-400 font-mono text-[10px]">Shift+Enter</kbd> for new line</span>
        <span className="font-mono">MiniGPT V3 • TinyStories</span>
      </div>
    </div>
  );
}

