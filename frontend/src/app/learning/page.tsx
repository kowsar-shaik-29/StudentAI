"use client";

import React, { useState, useEffect } from "react";
import { api, ApiError } from "@/lib/api";
import { LearningAskResponse, LearningProfileResponse } from "@/types/api";
import CitationCard from "@/components/CitationCard";
import {
  GraduationCap,
  HelpCircle,
  BookOpen,
  Search,
  Sparkles,
  BarChart2,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
} from "lucide-react";

export default function LearningPage() {
  const [query, setQuery] = useState("");
  const [intent, setIntent] = useState<string>("concept_explanation");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LearningAskResponse | null>(null);

  // Profile data for stats
  const [profile, setProfile] = useState<LearningProfileResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);

  // Quiz interactive state
  const [revealedAnswers, setRevealedAnswers] = useState<Record<number, boolean>>({});

  useEffect(() => {
    async function fetchProfile() {
      try {
        const data = await api.getLearningProfile();
        setProfile(data);
      } catch {
        // Backend might be offline
      } finally {
        setProfileLoading(false);
      }
    }
    fetchProfile();
  }, []);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setRevealedAnswers({});

    try {
      const resp = await api.askLearning(query, 5, intent || undefined);
      setResult(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Failed to get learning response from backend.");
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleAnswer = (idx: number) => {
    setRevealedAnswers((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="space-y-8">
      {/* Page Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <GraduationCap className="w-8 h-8 text-primary" />
            Learning & Knowledge RAG
          </h1>
          <p className="text-muted text-sm mt-1">
            Grounded curriculum learning, automatic topic detection, citations, and interactive quiz generation.
          </p>
        </div>
      </div>

      {/* Main Grid: Question Form & Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Ask Input & Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Query Box Card */}
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <form onSubmit={handleAsk} className="space-y-4">
              <div>
                <label htmlFor="learning-query" className="block text-sm font-semibold text-foreground mb-2">
                  What do you want to learn?
                </label>
                <div className="relative">
                  <input
                    id="learning-query"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. Explain binary search trees, or Generate quiz questions on normalization..."
                    className="w-full bg-background border border-border rounded-lg pl-4 pr-10 py-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                  />
                  <Search className="w-5 h-5 text-muted absolute right-3 top-3.5" />
                </div>
              </div>

              {/* Learning Intent Selection */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs font-semibold text-muted mr-1">Intent mode:</span>
                {[
                  { id: "concept_explanation", label: "Concept Explanation" },
                  { id: "quiz_generation", label: "Quiz / Practice" },
                  { id: "debugging", label: "Troubleshooting" },
                  { id: "revision", label: "Revision" },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setIntent(item.id)}
                    className={`text-xs px-3 py-1.5 rounded-full font-medium transition-colors ${
                      intent === item.id
                        ? "bg-primary text-white shadow-sm"
                        : "bg-surface border border-border text-muted hover:text-foreground hover:bg-slate-100"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>

              {/* Submit Button */}
              <div className="flex justify-end pt-2">
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Retrieving & Explaining...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Ask Learning Assistant
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
              <div>
                <span className="font-semibold">Request failed:</span> {error}
              </div>
            </div>
          )}

          {/* Results Display */}
          {result && (
            <div className="space-y-6">
              {/* Answer Card */}
              <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-border">
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-5 h-5 text-primary" />
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      {result.topic} &bull; {result.subtopic}
                    </span>
                  </div>
                  <span className="text-xs px-2.5 py-1 bg-primary/10 text-primary font-medium rounded-full">
                    Intent: {result.intent}
                  </span>
                </div>

                <div className="prose prose-slate max-w-none text-foreground text-sm leading-relaxed whitespace-pre-line">
                  {result.answer}
                </div>
              </div>

              {/* Interactive Quiz Section if available */}
              {result.questions && result.questions.length > 0 && (
                <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
                  <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
                    <HelpCircle className="w-5 h-5 text-amber-500" />
                    Practice Quiz ({result.questions.length} questions)
                  </h3>

                  <div className="space-y-4">
                    {result.questions.map((q, idx) => (
                      <div
                        key={idx}
                        className="bg-surface border border-border rounded-lg p-4 space-y-3"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <p className="text-sm font-medium text-foreground">
                            <span className="font-bold text-primary mr-2">Q{idx + 1}.</span>
                            {q.question}
                          </p>
                          {q.difficulty && (
                            <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-200 text-slate-700 uppercase">
                              {q.difficulty}
                            </span>
                          )}
                        </div>

                        {q.answer && (
                          <div>
                            <button
                              type="button"
                              onClick={() => toggleAnswer(idx)}
                              className="text-xs font-medium text-primary hover:underline flex items-center gap-1"
                            >
                              <Lightbulb className="w-3.5 h-3.5" />
                              {revealedAnswers[idx] ? "Hide Answer" : "Show Answer"}
                            </button>
                            {revealedAnswers[idx] && (
                              <div className="mt-2 text-xs bg-emerald-50 text-emerald-800 border border-emerald-200 rounded p-3">
                                <span className="font-semibold">Explanation: </span>
                                {q.answer}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Sources & Citations */}
              <CitationCard sources={result.sources} title="Academic Sources & Course Notes" />
            </div>
          )}
        </div>

        {/* Right Column (1 Col): Learning Stats & Analytics */}
        <div className="space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-primary" />
              Topic Knowledge Stats
            </h3>

            {profileLoading ? (
              <div className="text-xs text-muted animate-pulse py-4 text-center">
                Loading learning profile...
              </div>
            ) : profile && profile.topics && profile.topics.length > 0 ? (
              <div className="space-y-3">
                {profile.topics.map((t, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-surface rounded-lg border border-border flex flex-col gap-1.5"
                  >
                    <div className="flex items-center justify-between text-xs font-semibold text-foreground">
                      <span>{t.topic}</span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                          t.weakness_level === "HIGH"
                            ? "bg-red-500/10 text-red-600"
                            : t.weakness_level === "MEDIUM"
                            ? "bg-amber-500/10 text-amber-600"
                            : "bg-emerald-500/10 text-emerald-600"
                        }`}
                      >
                        {t.weakness_level}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-muted">
                      <span>{t.interactions} interactions</span>
                      <span>
                        Accuracy:{" "}
                        {t.quiz_accuracy !== null
                          ? `${(t.quiz_accuracy * 100).toFixed(0)}%`
                          : "N/A"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-muted py-4 text-center">
                No topic activity logged yet. Ask questions above to start building your learning profile!
              </div>
            )}
          </div>

          {/* Quick study tips */}
          <div className="bg-primary/5 border border-primary/20 rounded-xl p-5 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-primary flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4" />
              Pro Tip
            </h4>
            <p className="text-xs text-foreground/80 leading-relaxed">
              When reviewing tough concepts, select the &ldquo;Quiz / Practice&rdquo; intent mode to test your retention. The system logs quiz outcomes to personalize future study plans.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
