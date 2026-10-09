"use client";

import { useState } from "react";
import {
  Send,
  Bot,
  User,
  Sparkles,
  AlertCircle,
  FileText,
  Clock,
  Layers,
  CheckCircle,
} from "lucide-react";
import { api, ApiError } from "../../lib/api";
import { AgentRunResponse } from "../../types/api";
import { CitationCard } from "../../components/CitationCard";

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  responseMeta?: AgentRunResponse;
}

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "assistant",
      text: "Hello! I am your StudentAI Orchestrator. I can help you understand academic concepts, generate multi-day study plans, analyze job descriptions against your resume, and prepare for project interviews. How can I assist you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setError(null);
    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: query,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.runAgent(query);
      const assistantMsg: Message = {
        id: (Date.now() + 1).toString(),
        sender: "assistant",
        text: res.answer,
        responseMeta: res,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      setError(err instanceof ApiError ? err.detail : "Agent orchestration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Header Info */}
      <div className="pb-4 border-b border-slate-200 shrink-0">
        <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
          <Bot className="w-5 h-5 text-indigo-600" /> AI Assistant & Orchestrator
        </h2>
        <p className="text-xs text-slate-500 mt-0.5">
          Orchestrates academic notes, project repositories, and career documentation into grounded answers.
        </p>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto py-6 space-y-6">
        {messages.map((m) => {
          const isUser = m.sender === "user";
          return (
            <div
              key={m.id}
              className={`flex gap-3 max-w-3xl ${
                isUser ? "ml-auto flex-row-reverse" : "mr-auto"
              }`}
            >
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-white ${
                  isUser ? "bg-indigo-600" : "bg-slate-800"
                }`}
              >
                {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              <div className="space-y-3 flex-1 min-w-0">
                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    isUser
                      ? "bg-indigo-600 text-white rounded-tr-none shadow-sm font-normal"
                      : "bg-white border border-slate-200 text-slate-800 rounded-tl-none shadow-sm whitespace-pre-wrap"
                  }`}
                >
                  {m.text}
                </div>

                {/* Assistant Metadata (Execution Trace & Sources) */}
                {m.responseMeta && (
                  <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs space-y-3">
                    {/* Workflow & Classification */}
                    <div className="flex flex-wrap items-center gap-2 pb-2 border-b border-slate-200 text-slate-600">
                      <span className="font-semibold text-slate-800">
                        Workflow: {m.responseMeta.workflow}
                      </span>
                      <span>•</span>
                      <span>Category: {m.responseMeta.classification}</span>
                      <span>•</span>
                      <span className="inline-flex items-center gap-1 text-slate-500">
                        <Clock className="w-3 h-3" />
                        {m.responseMeta.total_duration_ms.toFixed(0)} ms
                      </span>
                      {m.responseMeta.citations_verified && (
                        <span className="ml-auto inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          <CheckCircle className="w-3 h-3 text-emerald-600" />
                          Citations Verified
                        </span>
                      )}
                    </div>

                    {/* Execution Trace Steps */}
                    {m.responseMeta.execution_trace.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider mb-1.5 flex items-center gap-1">
                          <Layers className="w-3 h-3" /> Tools Executed
                        </h4>
                        <div className="space-y-1">
                          {m.responseMeta.execution_trace.map((tr) => (
                            <div
                              key={tr.step}
                              className="flex items-center justify-between p-1.5 rounded bg-white border border-slate-100 text-[11px]"
                            >
                              <span className="font-medium text-slate-800">
                                Step {tr.step}: <code className="text-indigo-600">{tr.tool_name}</code>
                              </span>
                              <span
                                className={`px-1.5 py-0.5 rounded font-bold text-[9px] ${
                                  tr.status === "success"
                                    ? "bg-emerald-50 text-emerald-700"
                                    : "bg-rose-50 text-rose-700"
                                }`}
                              >
                                {tr.status.toUpperCase()} ({tr.duration_ms.toFixed(0)}ms)
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Grounded Sources */}
                    {m.responseMeta.sources.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-slate-700 text-[11px] uppercase tracking-wider mb-1.5 flex items-center gap-1">
                          <FileText className="w-3 h-3" /> Cited Sources
                        </h4>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                          {m.responseMeta.sources.slice(0, 4).map((s, idx) => (
                            <CitationCard key={idx} source={s} index={idx + 1} />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Limitations & Warnings */}
                    {m.responseMeta.limitations.length > 0 && (
                      <div className="p-2 bg-amber-50 rounded border border-amber-200 text-amber-800 text-[11px] space-y-0.5">
                        {m.responseMeta.limitations.map((lim, idx) => (
                          <div key={idx} className="flex items-start gap-1">
                            <span>*</span>
                            <span>{lim}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="flex gap-3 max-w-xl mr-auto">
            <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center text-white shrink-0">
              <Bot className="w-4 h-4" />
            </div>
            <div className="p-4 rounded-2xl bg-white border border-slate-200 text-slate-600 text-sm shadow-sm flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-600 animate-spin" />
              <span>Orchestrating tools & synthesizing answer...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="p-3 bg-rose-50 text-rose-700 rounded-xl text-xs border border-rose-200 flex items-center gap-2 max-w-xl">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="pt-4 border-t border-slate-200 shrink-0">
        <div className="flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder="Ask a question or describe your goal (e.g., 'Help me prepare for DBMS with a 7-day study plan'). Press Enter to send."
            rows={2}
            className="flex-1 p-3 bg-white border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 resize-none"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold text-sm rounded-xl transition-colors flex items-center justify-center gap-1.5 shrink-0 self-end h-11"
          >
            <Send className="w-4 h-4" />
            <span>Send</span>
          </button>
        </div>
      </form>
    </div>
  );
}
