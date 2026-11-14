import React from 'react';
import { Sparkles, BookOpen, Compass, Heart, Box, Bot } from 'lucide-react';

const SUGGESTIONS = [
  {
    icon: BookOpen,
    title: 'A Little Boy',
    prompt: 'Write a short story about a little boy.',
    description: 'A tale of curiosity and adventure',
  },
  {
    icon: Compass,
    title: 'Mysterious Forest',
    prompt: 'A dog discovers a mysterious forest.',
    description: 'Exploration through ancient woods',
  },
  {
    icon: Heart,
    title: 'Story About Friendship',
    prompt: 'Write a story about friendship.',
    description: 'Heartwarming bond between companions',
  },
  {
    icon: Box,
    title: 'The Magical Box',
    prompt: 'A little girl finds a magical box.',
    description: 'A discovery with an unexpected surprise',
  },
];

export default function WelcomeScreen({ onSelectPrompt }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto my-auto select-none">
      {/* Icon Badge */}
      <div className="relative mb-4 group">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-emerald-500/20 via-neutral-800 to-emerald-500/10 border border-emerald-500/30 flex items-center justify-center shadow-lg shadow-emerald-500/5 group-hover:scale-105 transition-transform duration-300">
          <Bot className="w-8 h-8 text-emerald-400" />
        </div>
        <div className="absolute -top-1 -right-1 flex h-4 w-4">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-40"></span>
          <span className="relative inline-flex rounded-full h-4 w-4 bg-emerald-500 text-[9px] font-bold text-black items-center justify-center">
            3
          </span>
        </div>
      </div>

      {/* Main Title */}
      <h1 className="text-2xl font-bold tracking-tight text-neutral-100 sm:text-3xl mb-2">
        MiniGPT <span className="text-emerald-400 font-mono">V3</span>
      </h1>

      <p className="text-sm text-neutral-400 max-w-md leading-relaxed mb-8">
        Custom 4-layer autoregressive GPT transformer trained from scratch on TinyStories. Enter a prompt or choose a starter below to generate.
      </p>

      {/* Suggested Prompts Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
        {SUGGESTIONS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              className="group p-4 rounded-xl bg-[#141417]/80 hover:bg-[#1c1c22] border border-neutral-800/80 hover:border-neutral-700/80 transition-all duration-200 cursor-pointer flex flex-col justify-between text-left shadow-sm hover:shadow-md"
            >
              <div className="flex items-center gap-2.5 mb-2">
                <div className="p-1.5 rounded-lg bg-neutral-800/80 group-hover:bg-emerald-500/10 transition-colors text-neutral-400 group-hover:text-emerald-400">
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-xs font-semibold text-neutral-200 group-hover:text-white transition-colors">
                  {item.title}
                </span>
              </div>
              <p className="text-xs text-neutral-400 group-hover:text-neutral-300 leading-snug font-medium line-clamp-2">
                "{item.prompt}"
              </p>
            </button>
          );
        })}
      </div>

      {/* Architecture Specs Pill */}
      <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-[11px] text-neutral-500">
        <span className="px-2 py-0.5 rounded-md bg-neutral-900 border border-neutral-800">256 Dim</span>
        <span>•</span>
        <span className="px-2 py-0.5 rounded-md bg-neutral-900 border border-neutral-800">4 Layers / 4 Heads</span>
        <span>•</span>
        <span className="px-2 py-0.5 rounded-md bg-neutral-900 border border-neutral-800">128 Context Window</span>
        <span>•</span>
        <span className="px-2 py-0.5 rounded-md bg-neutral-900 border border-neutral-800">2049 Vocab</span>
      </div>
    </div>
  );
}

