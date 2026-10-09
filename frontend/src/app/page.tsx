"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Bot,
  GraduationCap,
  CalendarCheck,
  Briefcase,
  FolderGit2,
  FileUp,
  AlertTriangle,
  Sparkles,
  ArrowRight,
  TrendingDown,
  BookOpen,
} from "lucide-react";
import { api } from "../lib/api";
import { LearningProfileResponse, LearningRecommendation } from "../types/api";

export default function OverviewPage() {
  const [profile, setProfile] = useState<LearningProfileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getLearningProfile()
      .then((data) => {
        setProfile(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.detail || "Could not load learning profile.");
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-2xl p-8 text-white shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-200 border border-indigo-400/30 mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            AI-Powered Student Companion
          </span>
          <h2 className="text-2xl font-bold tracking-tight">Welcome to StudentAI</h2>
          <p className="text-slate-300 text-sm mt-1 max-w-xl">
            Your personalized knowledge, learning, and career orchestration platform. Grounded in your
            course notes, resume, and software projects.
          </p>
        </div>
        <Link
          href="/assistant"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white text-indigo-900 font-semibold text-sm shadow-sm hover:bg-indigo-50 transition-colors shrink-0"
        >
          <Bot className="w-4 h-4 text-indigo-600" />
          Ask Assistant
        </Link>
      </div>

      {/* Quick Actions Grid */}
      <div>
        <h3 className="text-sm font-semibold text-slate-800 uppercase tracking-wider mb-4">
          Quick Actions
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link
            href="/assistant"
            className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-400 hover:shadow-sm transition-all group flex flex-col justify-between"
          >
            <div className="w-10 h-10 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3 group-hover:bg-indigo-600 group-hover:text-white transition-colors">
              <Bot className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 text-sm">Orchestrated AI</h4>
              <p className="text-xs text-slate-500 mt-1">
                Multi-step research across learning, career, and projects.
              </p>
            </div>
          </Link>

          <Link
            href="/learning"
            className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-400 hover:shadow-sm transition-all group flex flex-col justify-between"
          >
            <div className="w-10 h-10 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center mb-3 group-hover:bg-emerald-600 group-hover:text-white transition-colors">
              <GraduationCap className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 text-sm">Academic RAG</h4>
              <p className="text-xs text-slate-500 mt-1">
                Study notes, ask concept questions with citations.
              </p>
            </div>
          </Link>

          <Link
            href="/planner"
            className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-400 hover:shadow-sm transition-all group flex flex-col justify-between"
          >
            <div className="w-10 h-10 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center mb-3 group-hover:bg-amber-600 group-hover:text-white transition-colors">
              <CalendarCheck className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 text-sm">Study Planner</h4>
              <p className="text-xs text-slate-500 mt-1">
                Generate deterministic multi-day revision schedules.
              </p>
            </div>
          </Link>

          <Link
            href="/documents"
            className="p-5 bg-white rounded-xl border border-slate-200 hover:border-indigo-400 hover:shadow-sm transition-all group flex flex-col justify-between"
          >
            <div className="w-10 h-10 rounded-lg bg-purple-50 text-purple-600 flex items-center justify-center mb-3 group-hover:bg-purple-600 group-hover:text-white transition-colors">
              <FileUp className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 text-sm">Upload Notes</h4>
              <p className="text-xs text-slate-500 mt-1">
                Index PDF, Markdown, and text files into Qdrant.
              </p>
            </div>
          </Link>
        </div>
      </div>

      {/* Learning Status & Weaknesses Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Topic Mastery Overview */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-6">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100">
            <div>
              <h3 className="font-bold text-slate-900 text-base">Personalized Learning Profile</h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Aggregated from recent quiz outcomes and academic interactions
              </p>
            </div>
            <Link
              href="/learning"
              className="text-xs font-semibold text-indigo-600 hover:text-indigo-700 inline-flex items-center gap-1"
            >
              View Details <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>

          {loading ? (
            <div className="py-12 text-center text-slate-400 text-sm">Loading learning analytics...</div>
          ) : error ? (
            <div className="py-8 text-center text-amber-700 bg-amber-50 rounded-lg text-xs border border-amber-200 mt-4">
              {error}
            </div>
          ) : !profile || profile.topics.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-sm flex flex-col items-center">
              <BookOpen className="w-8 h-8 text-slate-300 mb-2" />
              <p className="font-medium text-slate-700">No learning history recorded yet</p>
              <p className="text-xs text-slate-400 mt-1 max-w-sm">
                Start asking academic questions in the Learning tab or take a quiz to begin tracking your mastery.
              </p>
            </div>
          ) : (
            <div className="mt-4 space-y-4">
              {/* Weak Topics Callout */}
              {profile.weak_topics.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-rose-700 uppercase tracking-wider flex items-center gap-1.5 mb-2">
                    <TrendingDown className="w-3.5 h-3.5" /> High-Priority Revision Areas
                  </h4>
                  <div className="space-y-2">
                    {profile.weak_topics.slice(0, 3).map((w, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 bg-rose-50/50 rounded-lg border border-rose-100 text-xs"
                      >
                        <div>
                          <span className="font-semibold text-slate-800">{w.topic}</span>
                          <span className="text-slate-500 ml-1.5">/ {w.subtopic}</span>
                        </div>
                        <span className="px-2 py-0.5 rounded font-bold text-[10px] bg-rose-100 text-rose-800">
                          {w.weakness_level} WEAKNESS
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Strong Topics */}
              {profile.strong_topics.length > 0 && (
                <div>
                  <h4 className="text-xs font-semibold text-emerald-700 uppercase tracking-wider mb-2">
                    Demonstrated Strengths
                  </h4>
                  <div className="space-y-2">
                    {profile.strong_topics.slice(0, 2).map((s, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 bg-emerald-50/40 rounded-lg border border-emerald-100 text-xs"
                      >
                        <div>
                          <span className="font-semibold text-slate-800">{s.topic}</span>
                          <span className="text-slate-500 ml-1.5">/ {s.subtopic}</span>
                        </div>
                        <span className="px-2 py-0.5 rounded font-medium text-[10px] bg-emerald-100 text-emerald-800">
                          Proficient
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Recommendations */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-slate-900 text-base">Actionable Recommendations</h3>
            <p className="text-xs text-slate-500 mt-0.5">What to study next based on your data</p>

            {loading ? (
              <div className="py-8 text-center text-slate-400 text-sm">Loading recommendations...</div>
            ) : !profile || profile.recommendations.length === 0 ? (
              <div className="py-8 text-center text-slate-400 text-xs italic">
                No active recommendations. Complete practice quizzes to generate targeted suggestions.
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                {profile.recommendations.slice(0, 3).map((r, idx) => (
                  <div key={idx} className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-slate-900">{r.topic}</span>
                      <span className="text-[10px] font-semibold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                        {r.priority}
                      </span>
                    </div>
                    <p className="text-slate-600 text-[11px]">{r.recommended_action}</p>
                    {r.source_documents.length > 0 && (
                      <p className="text-[10px] text-slate-400 mt-1">
                        Sources: {r.source_documents.join(", ")}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <Link
            href="/planner"
            className="mt-6 w-full py-2.5 px-4 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-semibold text-xs rounded-lg text-center transition-colors flex items-center justify-center gap-1.5"
          >
            Generate Study Schedule <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
}
