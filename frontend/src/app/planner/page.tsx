"use client";

import React, { useState } from "react";
import { api, ApiError } from "@/lib/api";
import { StudyPlanResponse } from "@/types/api";
import {
  Calendar,
  Clock,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  BookOpen,
  HelpCircle,
  Layers,
  ArrowRight,
} from "lucide-react";

export default function PlannerPage() {
  const [subject, setSubject] = useState("Computer Science");
  const [days, setDays] = useState(7);
  const [dailyMinutes, setDailyMinutes] = useState(60);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<StudyPlanResponse | null>(null);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!subject.trim() || loading) return;

    setLoading(true);
    setError(null);

    try {
      const resp = await api.createStudyPlan(subject, days, dailyMinutes);
      setPlan(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Failed to generate study plan. Please ensure backend is reachable.");
      }
    } finally {
      setLoading(false);
    }
  };

  const getActivityIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "concept_review":
      case "review":
      case "theory":
        return <BookOpen className="w-4 h-4 text-blue-500" />;
      case "quiz_practice":
      case "practice":
      case "quiz":
        return <HelpCircle className="w-4 h-4 text-amber-500" />;
      default:
        return <Layers className="w-4 h-4 text-primary" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Calendar className="w-8 h-8 text-primary" />
            Study Planner & Revision Engine
          </h1>
          <p className="text-muted text-sm mt-1">
            Generate personalized, adaptive daily study schedules grounded in your weak topics and course materials.
          </p>
        </div>
      </div>

      {/* Configuration Form Card */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
        <form onSubmit={handleGenerate} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label htmlFor="subject-input" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                Target Subject / Area
              </label>
              <input
                id="subject-input"
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                placeholder="e.g. Operating Systems, DBMS, Machine Learning"
                className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                required
              />
            </div>

            <div>
              <label htmlFor="days-select" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                Duration (Days)
              </label>
              <select
                id="days-select"
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
                className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
              >
                <option value={3}>3 Days (Quick Revision)</option>
                <option value={5}>5 Days (Sprint)</option>
                <option value={7}>7 Days (1 Week Full)</option>
                <option value={14}>14 Days (2 Weeks In-depth)</option>
              </select>
            </div>

            <div>
              <label htmlFor="minutes-select" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                Daily Study Time (Minutes)
              </label>
              <select
                id="minutes-select"
                value={dailyMinutes}
                onChange={(e) => setDailyMinutes(Number(e.target.value))}
                className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
              >
                <option value={30}>30 mins / day</option>
                <option value={45}>45 mins / day</option>
                <option value={60}>60 mins (1 hour) / day</option>
                <option value={90}>90 mins (1.5 hours) / day</option>
                <option value={120}>120 mins (2 hours) / day</option>
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <span className="text-xs text-muted">
              Total estimated study commitment: <span className="font-semibold text-foreground">{((days * dailyMinutes) / 60).toFixed(1)} hours</span>
            </span>
            <button
              type="submit"
              disabled={loading || !subject.trim()}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Generating Plan...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Generate Personalized Plan
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Error State */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
          <div>
            <span className="font-semibold">Planning error:</span> {error}
          </div>
        </div>
      )}

      {/* Plan Output */}
      {plan && (
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 bg-surface border border-border rounded-xl">
            <div className="space-y-1">
              <span className="text-xs font-bold uppercase tracking-wider text-primary">
                Generated Schedule
              </span>
              <h2 className="text-lg font-bold text-foreground">
                {plan.subject} &bull; {plan.days}-Day Personalized Curriculum
              </h2>
            </div>
            <div className="flex items-center gap-3 text-xs font-medium text-muted">
              <span className="flex items-center gap-1.5 px-3 py-1.5 bg-card border border-border rounded-lg">
                <Clock className="w-4 h-4 text-primary" />
                {plan.daily_minutes} mins / day
              </span>
              <span className="flex items-center gap-1.5 px-3 py-1.5 bg-card border border-border rounded-lg">
                <Calendar className="w-4 h-4 text-primary" />
                {plan.plan.length} Scheduled Days
              </span>
            </div>
          </div>

          {/* Daily Schedule Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {plan.plan.map((day) => (
              <div
                key={day.day}
                className="bg-card border border-border rounded-xl p-5 shadow-sm flex flex-col justify-between hover:border-border-hover transition-colors"
              >
                <div className="space-y-4">
                  <div className="flex items-center justify-between border-b border-border pb-3">
                    <span className="text-sm font-bold text-primary">Day {day.day}</span>
                    <span className="text-xs px-2.5 py-0.5 rounded-full bg-surface border border-border font-medium text-muted">
                      {day.estimated_minutes} mins
                    </span>
                  </div>

                  {/* Focus Topics */}
                  {day.focus && day.focus.length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                        Topic Focus:
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {day.focus.map((f, fIdx) => (
                          <span
                            key={fIdx}
                            className={`text-xs px-2 py-0.5 rounded font-medium ${
                              f.priority === "HIGH"
                                ? "bg-red-500/10 text-red-600"
                                : "bg-primary/10 text-primary"
                            }`}
                          >
                            {f.subtopic || f.topic}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Daily Activities List */}
                  <div className="space-y-2 pt-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
                      Tasks:
                    </span>
                    <div className="space-y-2">
                      {day.activities.map((act, aIdx) => (
                        <div
                          key={aIdx}
                          className="bg-surface border border-border rounded-lg p-2.5 text-xs flex items-start gap-2.5"
                        >
                          <div className="mt-0.5 flex-shrink-0">
                            {getActivityIcon(act.type)}
                          </div>
                          <div className="flex-1">
                            <p className="font-medium text-foreground leading-snug">
                              {act.description}
                            </p>
                            <span className="text-[11px] text-muted font-mono mt-0.5 inline-block">
                              {act.minutes}m &bull; {act.type.replace(/_/g, " ")}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="pt-4 mt-4 border-t border-border flex items-center justify-between text-xs text-muted">
                  <span className="flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-500" /> Complete daily targets
                  </span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
