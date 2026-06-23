"use client";

import { useState, useRef, useEffect } from "react";

const EMOTION_COLORS = {
  happy: "#FBBF24",
  excited: "#F97316",
  sad: "#60A5FA",
  worried: "#A78BFA",
  angry: "#EF4444",
  surprised: "#34D399",
  thinking: "#94A3B8",
  calm: "#818CF8",
  love: "#F472B6",
  shy: "#FB923C",
};

function EmotionBadge({ emotion }) {
  const color = EMOTION_COLORS[emotion] || EMOTION_COLORS.calm;
  return (
    <span
      className="inline-block text-xs px-2 py-0.5 rounded-full mr-2 font-medium"
      style={{ backgroundColor: color + "22", color, border: `1px solid ${color}44` }}
    >
      {emotion}
    </span>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 px-4 py-3">
      <div className="w-8 h-8 rounded-full bg-sion-primary flex items-center justify-center text-sm font-bold shrink-0">
        시
      </div>
      <div className="bg-sion-card rounded-2xl rounded-tl-sm px-4 py-3 flex gap-1">
        <span className="typing-dot w-2 h-2 bg-sion-muted rounded-full inline-block" />
        <span className="typing-dot w-2 h-2 bg-sion-muted rounded-full inline-block" />
        <span className="typing-dot w-2 h-2 bg-sion-muted rounded-full inline-block" />
      </div>
    </div>
  );
}

function MessageBubble({ role, text, emotion }) {
  const isSion = role === "assistant";
  return (
    <div className={`flex items-end gap-2 px-4 py-1 ${isSion ? "" : "flex-row-reverse"}`}>
      {isSion && (
        <div className="w-8 h-8 rounded-full bg-sion-primary flex items-center justify-center text-sm font-bold shrink-0">
          시
        </div>
      )}
      <div
        className={`max-w-[75%] px-4 py-2.5 text-sm leading-relaxed ${
          isSion
            ? "bg-sion-card rounded-2xl rounded-tl-sm"
            : "bg-sion-primary rounded-2xl rounded-tr-sm"
        }`}
      >
        {isSion && emotion && <EmotionBadge emotion={emotion} />}
        <span className="whitespace-pre-wrap">{text}</span>
      </div>
    </div>
  );
}

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "안녕! 나는 시온이야~ AI DJ VTuber! 뭐든 편하게 물어봐!",
      emotion: "excited",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(null); // null=unknown, true/false
  const bottomRef = useRef(null);

  // 서버 상태 체크
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setConnected(d.status === "ok"))
      .catch(() => setConnected(false));
  }, []);

  // 자동 스크롤
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();

      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: "연결에 문제가 있어... 잠시 후 다시 시도해줘!", emotion: "worried" },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: data.reply, emotion: data.emotion || "calm" },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "서버에 연결할 수 없어ㅠㅠ 나중에 다시 와줘!", emotion: "sad" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <main className="flex flex-col h-screen max-w-2xl mx-auto">
      {/* 헤더 */}
      <header className="flex items-center gap-3 px-4 py-3 border-b border-white/10">
        <div className="w-10 h-10 rounded-full bg-sion-primary flex items-center justify-center text-lg font-bold">
          시
        </div>
        <div className="flex-1">
          <h1 className="text-base font-semibold">시온 (sion)</h1>
          <p className="text-xs text-sion-muted">AI DJ VTuber</p>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              connected === true ? "bg-green-400" : connected === false ? "bg-red-400" : "bg-yellow-400"
            }`}
          />
          <span className="text-xs text-sion-muted">
            {connected === true ? "온라인" : connected === false ? "오프라인" : "확인 중..."}
          </span>
        </div>
      </header>

      {/* 채팅 영역 */}
      <div className="flex-1 overflow-y-auto py-4 space-y-1">
        {messages.map((msg, i) => (
          <MessageBubble key={i} {...msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* 입력 영역 */}
      <div className="px-4 py-3 border-t border-white/10">
        <div className="flex gap-2 items-end">
          <textarea
            className="flex-1 bg-sion-input rounded-xl px-4 py-2.5 text-sm resize-none outline-none focus:ring-2 focus:ring-sion-primary/50 placeholder:text-sion-muted/60 max-h-32"
            rows={1}
            placeholder="시온에게 메시지 보내기..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            className="bg-sion-primary hover:bg-sion-primary/80 disabled:opacity-40 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors shrink-0"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            전송
          </button>
        </div>
        <p className="text-center text-xs text-sion-muted/50 mt-2">
          시온은 AI입니다. 답변이 부정확할 수 있어요.
        </p>
      </div>
    </main>
  );
}
