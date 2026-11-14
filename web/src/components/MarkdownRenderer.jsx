import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-lg overflow-hidden border border-neutral-800 bg-[#141416]">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-[#1a1a1e] border-b border-neutral-800 text-xs text-neutral-400">
        <span className="font-mono uppercase tracking-wider text-[11px] font-medium text-neutral-300">
          {language || 'text'}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-neutral-400 hover:text-neutral-200 transition-colors py-0.5 px-2 rounded hover:bg-neutral-800"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy code</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto text-xs leading-relaxed text-neutral-200 font-mono">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export default function MarkdownRenderer({ content }) {
  if (!content) return null;

  // Split into code blocks vs text blocks
  const parts = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({
        type: 'text',
        content: content.slice(lastIndex, match.index),
      });
    }
    parts.push({
      type: 'code',
      language: match[1] || 'text',
      code: match[2].trimEnd(),
    });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push({
      type: 'text',
      content: content.slice(lastIndex),
    });
  }

  const renderInlineFormatted = (text) => {
    // Process inline elements: inline code, bold, italic
    const tokens = [];
    const inlineRegex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
    let idx = 0;
    let m;

    while ((m = inlineRegex.exec(text)) !== null) {
      if (m.index > idx) {
        tokens.push(text.slice(idx, m.index));
      }
      const raw = m[0];
      if (raw.startsWith('`') && raw.endsWith('`')) {
        tokens.push(
          <code
            key={m.index}
            className="px-1.5 py-0.5 mx-0.5 rounded bg-neutral-800/80 border border-neutral-700/60 text-emerald-300 font-mono text-[0.88em]"
          >
            {raw.slice(1, -1)}
          </code>
        );
      } else if (raw.startsWith('**') && raw.endsWith('**')) {
        tokens.push(
          <strong key={m.index} className="font-semibold text-neutral-100">
            {raw.slice(2, -2)}
          </strong>
        );
      } else if (raw.startsWith('*') && raw.endsWith('*')) {
        tokens.push(
          <em key={m.index} className="italic text-neutral-200">
            {raw.slice(1, -1)}
          </em>
        );
      }
      idx = m.index + raw.length;
    }

    if (idx < text.length) {
      tokens.push(text.slice(idx));
    }

    return tokens;
  };

  const renderTextParagraphs = (rawText, blockIndex) => {
    const lines = rawText.split('\n');
    return lines.map((line, lineIndex) => {
      // Empty line
      if (line.trim() === '') {
        return <div key={`${blockIndex}-${lineIndex}`} className="h-2" />;
      }

      // Blockquote
      if (line.startsWith('> ')) {
        return (
          <blockquote
            key={`${blockIndex}-${lineIndex}`}
            className="border-l-2 border-emerald-500/70 pl-3 my-1.5 text-neutral-300 italic"
          >
            {renderInlineFormatted(line.slice(2))}
          </blockquote>
        );
      }

      // Heading 3
      if (line.startsWith('### ')) {
        return (
          <h3
            key={`${blockIndex}-${lineIndex}`}
            className="font-semibold text-base text-neutral-100 mt-2 mb-1"
          >
            {renderInlineFormatted(line.slice(4))}
          </h3>
        );
      }

      // Heading 2
      if (line.startsWith('## ')) {
        return (
          <h2
            key={`${blockIndex}-${lineIndex}`}
            className="font-bold text-lg text-neutral-50 mt-3 mb-1"
          >
            {renderInlineFormatted(line.slice(3))}
          </h2>
        );
      }

      // Bullet lists
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        return (
          <div
            key={`${blockIndex}-${lineIndex}`}
            className="flex items-start gap-2 my-0.5 ml-2"
          >
            <span className="text-neutral-500 mt-1.5 text-xs">•</span>
            <span className="flex-1">{renderInlineFormatted(line.trim().slice(2))}</span>
          </div>
        );
      }

      // Regular paragraph line
      return (
        <p
          key={`${blockIndex}-${lineIndex}`}
          className="leading-relaxed text-[14.5px] text-neutral-200"
        >
          {renderInlineFormatted(line)}
        </p>
      );
    });
  };

  return (
    <div className="space-y-1 text-neutral-200 text-sm">
      {parts.map((part, index) => {
        if (part.type === 'code') {
          return (
            <CodeBlock
              key={index}
              language={part.language}
              code={part.code}
            />
          );
        }
        return (
          <div key={index} className="space-y-1">
            {renderTextParagraphs(part.content, index)}
          </div>
        );
      })}
    </div>
  );
}

