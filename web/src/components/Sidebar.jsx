import React, { useState } from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  Sliders,
  Info,
  ChevronLeft,
  ChevronRight,
  Bot,
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';

export default function Sidebar({
  isOpen,
  onToggle,
  conversations,
  activeChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onOpenSettings,
  backendStatus,
}) {
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);

  const handleDeleteClick = (e, chatId) => {
    e.stopPropagation();
    if (deleteConfirmId === chatId) {
      onDeleteChat(chatId);
      setDeleteConfirmId(null);
    } else {
      setDeleteConfirmId(chatId);
      setTimeout(() => {
        setDeleteConfirmId((current) => (current === chatId ? null : current));
      }, 3000);
    }
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden backdrop-blur-xs transition-opacity"
          onClick={onToggle}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col bg-[#111113] border-r border-neutral-800/80 transition-all duration-300 ease-in-out ${
          isOpen ? 'w-64 sm:w-72 translate-x-0' : 'w-0 -translate-x-full md:w-0 md:translate-x-0 overflow-hidden'
        }`}
      >
        <div className="w-64 sm:w-72 flex flex-col h-full">
          {/* Top Header: Logo + Collapse */}
          <div className="flex items-center justify-between p-3.5 border-b border-neutral-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-500/20 to-neutral-800 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <Bot className="w-4 h-4" />
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-sm tracking-tight text-neutral-100 flex items-center gap-1.5">
                  MiniGPT <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">V3</span>
                </span>
                <span className="text-[11px] text-neutral-500">Local Transformer</span>
              </div>
            </div>

            <button
              onClick={onToggle}
              className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
              title="Close sidebar"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
          </div>

          {/* New Chat Button */}
          <div className="p-3">
            <button
              onClick={onNewChat}
              className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-[#1c1c20] hover:bg-[#25252b] border border-neutral-800 hover:border-neutral-700 text-xs font-semibold text-neutral-200 transition-all shadow-sm group cursor-pointer"
            >
              <div className="flex items-center gap-2">
                <Plus className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
                <span>New Story / Chat</span>
              </div>
              <kbd className="text-[10px] font-mono text-neutral-500 px-1.5 py-0.5 rounded bg-neutral-800/80">
                Alt+N
              </kbd>
            </button>
          </div>

          {/* Conversations List */}
          <div className="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
            <div className="px-2.5 py-1 text-[11px] font-semibold text-neutral-500 uppercase tracking-wider">
              Recent Chats
            </div>

            {conversations.length === 0 ? (
              <div className="px-3 py-6 text-center text-xs text-neutral-500">
                No conversations yet.
              </div>
            ) : (
              conversations.map((chat) => {
                const isActive = chat.id === activeChatId;
                const isConfirmingDelete = deleteConfirmId === chat.id;

                return (
                  <div
                    key={chat.id}
                    onClick={() => onSelectChat(chat.id)}
                    className={`group relative flex items-center justify-between px-2.5 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
                      isActive
                        ? 'bg-[#1e1e24] text-neutral-100 font-medium'
                        : 'text-neutral-400 hover:bg-[#161619] hover:text-neutral-200'
                    }`}
                  >
                    <div className="flex items-center gap-2 min-w-0 flex-1 pr-1">
                      <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-emerald-400' : 'text-neutral-500'}`} />
                      <span className="truncate">{chat.title || 'Untitled Conversation'}</span>
                    </div>

                    {/* Delete action button */}
                    <button
                      onClick={(e) => handleDeleteClick(e, chat.id)}
                      className={`shrink-0 p-1 rounded-md transition-all ${
                        isConfirmingDelete
                          ? 'bg-red-500/20 text-red-400 opacity-100'
                          : 'opacity-0 group-hover:opacity-100 text-neutral-500 hover:text-red-400 hover:bg-neutral-800'
                      }`}
                      title={isConfirmingDelete ? 'Click again to confirm delete' : 'Delete chat'}
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* Model Specs Card */}
          <div className="p-3 border-t border-neutral-800/80 bg-[#0e0e10]">
            <div className="p-2.5 rounded-xl bg-[#141417] border border-neutral-800/80 space-y-2 text-[11px]">
              <div className="flex items-center justify-between text-neutral-300 font-medium">
                <span className="flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-emerald-400" />
                  Model Specs
                </span>
                <span className="font-mono text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded">
                  {backendStatus?.data?.device || 'CUDA'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-1 text-[10px] text-neutral-400 pt-0.5">
                <div>Layers: <span className="text-neutral-200 font-mono">4</span></div>
                <div>Heads: <span className="text-neutral-200 font-mono">4</span></div>
                <div>Embed: <span className="text-neutral-200 font-mono">256</span></div>
                <div>Ctx: <span className="text-neutral-200 font-mono">128</span></div>
              </div>
              <div className="text-[10px] text-neutral-500 pt-1 border-t border-neutral-800/60 truncate">
                Weights: checkpoints_tokenizer_v3/best.pt
              </div>
            </div>
          </div>

          {/* Bottom Settings Button */}
          <div className="p-2.5 border-t border-neutral-800/80 bg-[#111113]">
            <button
              onClick={onOpenSettings}
              className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs text-neutral-300 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-neutral-400" />
                <span>Settings</span>
              </div>
              <span className="text-[10px] text-neutral-500">Sampling</span>
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

