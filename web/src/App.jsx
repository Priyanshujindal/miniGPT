import React, { useState, useEffect, useRef } from 'react';
import { Menu, Sliders, Bot, Plus, Loader2 } from 'lucide-react';
import Sidebar from './components/Sidebar.jsx';
import WelcomeScreen from './components/WelcomeScreen.jsx';
import ChatMessage from './components/ChatMessage.jsx';
import MessageComposer from './components/MessageComposer.jsx';
import SettingsModal, { DEFAULT_SETTINGS } from './components/SettingsModal.jsx';
import ConnectionBanner from './components/ConnectionBanner.jsx';
import { getHealthStatus, generateCompletion } from './services/api.js';
import { useLocalStorage } from './hooks/useLocalStorage.js';

export default function App() {
  // Local storage state
  const [conversations, setConversations] = useLocalStorage('minigpt_conversations', []);
  const [activeChatId, setActiveChatId] = useLocalStorage('minigpt_active_chat_id', null);
  const [settings, setSettings] = useLocalStorage('minigpt_settings', DEFAULT_SETTINGS);

  // UI state
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const [isCheckingBackend, setIsCheckingBackend] = useState(false);

  const messagesEndRef = useRef(null);

  // Check backend health on mount and every 30 seconds
  const checkBackend = async () => {
    setIsCheckingBackend(true);
    const status = await getHealthStatus();
    setBackendStatus(status);
    setIsCheckingBackend(false);
  };

  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  // Keyboard shortcut Alt+N for new chat
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.altKey && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [conversations]);

  // Current active conversation
  const currentChat = conversations.find((c) => c.id === activeChatId) || null;
  const messages = currentChat ? currentChat.messages : [];

  // Scroll to bottom when messages change
  const scrollToBottom = (behavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isGenerating]);

  // Create new conversation
  const handleNewChat = () => {
    const newChat = {
      id: `chat_${Date.now()}`,
      title: 'New Story',
      createdAt: new Date().toISOString(),
      messages: [],
    };
    setConversations([newChat, ...conversations]);
    setActiveChatId(newChat.id);
    setInput('');
  };

  // Delete a conversation
  const handleDeleteChat = (chatId) => {
    const updated = conversations.filter((c) => c.id !== chatId);
    setConversations(updated);
    if (activeChatId === chatId) {
      setActiveChatId(updated.length > 0 ? updated[0].id : null);
    }
  };

  // Send a prompt to the model
  const handleSendMessage = async (customPrompt = null) => {
    const promptToSend = customPrompt || input.trim();
    if (!promptToSend || isGenerating) return;

    let targetChatId = activeChatId;
    let targetChat = currentChat;

    // Create a new conversation if none exists
    if (!targetChatId || !targetChat) {
      targetChatId = `chat_${Date.now()}`;
      targetChat = {
        id: targetChatId,
        title: promptToSend.slice(0, 32) + (promptToSend.length > 32 ? '...' : ''),
        createdAt: new Date().toISOString(),
        messages: [],
      };
      setConversations([targetChat, ...conversations]);
      setActiveChatId(targetChatId);
    } else if (targetChat.messages.length === 0) {
      // Update title from the first message
      targetChat.title = promptToSend.slice(0, 32) + (promptToSend.length > 32 ? '...' : '');
    }

    const userMessage = {
      id: `msg_user_${Date.now()}`,
      role: 'user',
      content: promptToSend,
      timestamp: new Date().toISOString(),
    };

    // Optimistically add user message
    const updatedMessages = [...targetChat.messages, userMessage];
    setConversations((prev) =>
      prev.map((c) => (c.id === targetChatId ? { ...c, messages: updatedMessages, title: targetChat.title } : c))
    );

    if (!customPrompt) {
      setInput('');
    }

    setIsGenerating(true);

    // Call backend API
    const result = await generateCompletion({
      message: promptToSend,
      max_new_tokens: settings.max_new_tokens,
      temperature: settings.temperature,
      top_k: settings.top_k,
      repetition_penalty: settings.repetition_penalty,
    });

    setIsGenerating(false);

    let assistantMessage;
    if (result.success) {
      assistantMessage = {
        id: `msg_assistant_${Date.now()}`,
        role: 'assistant',
        content: result.response,
        model: result.model,
        timestamp: new Date().toISOString(),
      };
    } else {
      assistantMessage = {
        id: `msg_assistant_${Date.now()}`,
        role: 'assistant',
        content: '',
        error: result.error,
        timestamp: new Date().toISOString(),
      };
    }

    setConversations((prev) =>
      prev.map((c) =>
        c.id === targetChatId
          ? { ...c, messages: [...updatedMessages, assistantMessage] }
          : c
      )
    );
  };

  // Regenerate the last assistant response
  const handleRegenerate = async () => {
    if (!currentChat || isGenerating) return;

    // Find the last user message
    const lastUserMessage = [...currentChat.messages].reverse().find((m) => m.role === 'user');
    if (!lastUserMessage) return;

    // Pop the last assistant message and resend
    const filteredMessages = currentChat.messages.slice(0, -1);
    setConversations((prev) =>
      prev.map((c) => (c.id === currentChat.id ? { ...c, messages: filteredMessages } : c))
    );

    setIsGenerating(true);

    const result = await generateCompletion({
      message: lastUserMessage.content,
      max_new_tokens: settings.max_new_tokens,
      temperature: settings.temperature,
      top_k: settings.top_k,
      repetition_penalty: settings.repetition_penalty,
    });

    setIsGenerating(false);

    let assistantMessage;
    if (result.success) {
      assistantMessage = {
        id: `msg_assistant_${Date.now()}`,
        role: 'assistant',
        content: result.response,
        model: result.model,
        timestamp: new Date().toISOString(),
      };
    } else {
      assistantMessage = {
        id: `msg_assistant_${Date.now()}`,
        role: 'assistant',
        content: '',
        error: result.error,
        timestamp: new Date().toISOString(),
      };
    }

    setConversations((prev) =>
      prev.map((c) =>
        c.id === currentChat.id
          ? { ...c, messages: [...filteredMessages, assistantMessage] }
          : c
      )
    );
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0d0d0f] text-neutral-100 antialiased">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        conversations={conversations}
        activeChatId={activeChatId}
        onSelectChat={(id) => setActiveChatId(id)}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onOpenSettings={() => setSettingsOpen(true)}
        backendStatus={backendStatus}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-[#0e0e11] relative">
        {/* Top Navbar */}
        <header className="h-14 border-b border-neutral-800/80 px-4 flex items-center justify-between shrink-0 bg-[#101014]/90 backdrop-blur-md z-20">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
                title="Open sidebar"
              >
                <Menu className="w-5 h-5" />
              </button>
            )}

            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-neutral-200">
                {currentChat?.title || 'MiniGPT V3'}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-neutral-700/50">
                TinyStories
              </span>
            </div>
          </div>

          {/* Right navbar items: Connection banner + Settings toggle */}
          <div className="flex items-center gap-3">
            <ConnectionBanner
              backendStatus={backendStatus}
              onRetry={checkBackend}
              isChecking={isCheckingBackend}
            />

            <button
              onClick={() => setSettingsOpen(true)}
              className="p-1.5 rounded-lg text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800 transition-colors"
              title="Generation Parameters"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Chat / Messages scrollable container */}
        <main className="flex-1 overflow-y-auto flex flex-col">
          {messages.length === 0 ? (
            <WelcomeScreen onSelectPrompt={(prompt) => handleSendMessage(prompt)} />
          ) : (
            <div className="flex-1 divide-y divide-neutral-800/40 pb-6">
              {messages.map((msg, index) => {
                const isLastAssistant =
                  msg.role === 'assistant' && index === messages.length - 1;
                return (
                  <ChatMessage
                    key={msg.id}
                    message={msg}
                    isLastAssistantMessage={isLastAssistant}
                    onRegenerate={handleRegenerate}
                    isGenerating={isGenerating}
                  />
                );
              })}

              {/* Generating loading indicator */}
              {isGenerating && (
                <div className="py-5 px-4 sm:px-6 w-full bg-[#121214]/60">
                  <div className="max-w-3xl mx-auto flex gap-4">
                    <div className="w-8 h-8 rounded-xl bg-neutral-800 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm shrink-0">
                      <Bot className="w-4 h-4 animate-pulse" />
                    </div>
                    <div className="flex items-center gap-2 text-xs text-neutral-400 pt-1.5">
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-400" />
                      <span>MiniGPT V3 is generating tokens autoregressively...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </main>

        {/* Message Composer */}
        <footer className="shrink-0 bg-gradient-to-t from-[#0e0e11] via-[#0e0e11] to-transparent pt-2">
          <MessageComposer
            input={input}
            setInput={setInput}
            onSend={() => handleSendMessage()}
            isGenerating={isGenerating}
            onOpenSettings={() => setSettingsOpen(true)}
            settings={settings}
          />
        </footer>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSave={(newSettings) => setSettings(newSettings)}
        onReset={() => setSettings(DEFAULT_SETTINGS)}
      />
    </div>
  );
}

