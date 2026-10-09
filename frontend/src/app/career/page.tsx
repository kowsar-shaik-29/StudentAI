"use client";

import React, { useState } from "react";
import { api, ApiError } from "@/lib/api";
import { CareerAskResponse, JobFitResponse } from "@/types/api";
import CitationCard from "@/components/CitationCard";
import {
  Briefcase,
  Search,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  ShieldAlert,
  FileText,
  Target,
} from "lucide-react";

export default function CareerPage() {
  const [activeTab, setActiveTab] = useState<"qa" | "match">("match");

  // Career QA State
  const [query, setQuery] = useState("");
  const [docType, setDocType] = useState<string>("resume");
  const [qaLoading, setQaLoading] = useState(false);
  const [qaError, setQaError] = useState<string | null>(null);
  const [qaResult, setQaResult] = useState<CareerAskResponse | null>(null);

  // Job Fit Match State
  const [resumeId, setResumeId] = useState("resume_sample");
  const [jobId, setJobId] = useState("job_sample");
  const [matchLoading, setMatchLoading] = useState(false);
  const [matchError, setMatchError] = useState<string | null>(null);
  const [matchResult, setMatchResult] = useState<JobFitResponse | null>(null);

  const handleAskCareer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || qaLoading) return;

    setQaLoading(true);
    setQaError(null);
    setQaResult(null);

    try {
      const resp = await api.askCareer(query, 5, docType || undefined);
      setQaResult(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setQaError(err.detail);
      } else {
        setQaError("Failed to query career RAG.");
      }
    } finally {
      setQaLoading(false);
    }
  };

  const handleJobFit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resumeId.trim() || !jobId.trim() || matchLoading) return;

    setMatchLoading(true);
    setMatchError(null);
    setMatchResult(null);

    try {
      const resp = await api.analyzeJobFit(resumeId, jobId);
      setMatchResult(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setMatchError(err.detail);
      } else {
        setMatchError("Job fit analysis failed. Ensure document IDs exist.");
      }
    } finally {
      setMatchLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Briefcase className="w-8 h-8 text-primary" />
            Career RAG & Job Matching
          </h1>
          <p className="text-muted text-sm mt-1">
            Grounded resume intelligence, deterministic skill gap analysis, and honest ATS capability assessment.
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex items-center gap-1 bg-surface border border-border p-1 rounded-lg">
          <button
            onClick={() => setActiveTab("match")}
            className={`text-xs font-semibold px-4 py-2 rounded-md transition-colors ${
              activeTab === "match"
                ? "bg-card text-primary shadow-sm"
                : "text-muted hover:text-foreground"
            }`}
          >
            Job-Fit Matcher
          </button>
          <button
            onClick={() => setActiveTab("qa")}
            className={`text-xs font-semibold px-4 py-2 rounded-md transition-colors ${
              activeTab === "qa"
                ? "bg-card text-primary shadow-sm"
                : "text-muted hover:text-foreground"
            }`}
          >
            Career Q&amp;A
          </button>
        </div>
      </div>

      {activeTab === "match" && (
        <div className="space-y-6">
          {/* Form */}
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <form onSubmit={handleJobFit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="resume-id" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                    Resume Document ID or Name
                  </label>
                  <input
                    id="resume-id"
                    type="text"
                    value={resumeId}
                    onChange={(e) => setResumeId(e.target.value)}
                    placeholder="e.g. resume.pdf or resume_sample"
                    className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                    required
                  />
                  <p className="text-[11px] text-muted mt-1.5">
                    Select a parsed candidate resume indexed in StudentAI.
                  </p>
                </div>

                <div>
                  <label htmlFor="job-id" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                    Job Description ID or Name
                  </label>
                  <input
                    id="job-id"
                    type="text"
                    value={jobId}
                    onChange={(e) => setJobId(e.target.value)}
                    placeholder="e.g. job_description.pdf or job_sample"
                    className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                    required
                  />
                  <p className="text-[11px] text-muted mt-1.5">
                    Target job posting or vacancy requirement document.
                  </p>
                </div>
              </div>

              <div className="flex justify-end pt-2 border-t border-border">
                <button
                  type="submit"
                  disabled={matchLoading || !resumeId.trim() || !jobId.trim()}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
                >
                  {matchLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing Alignment...
                    </>
                  ) : (
                    <>
                      <Target className="w-4 h-4" />
                      Run Job-Fit Analysis
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {matchError && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
              <div>
                <span className="font-semibold">Analysis failed:</span> {matchError}
              </div>
            </div>
          )}

          {matchResult && (
            <div className="space-y-6">
              {/* Coverage KPI Header */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Skill Coverage
                  </span>
                  <div className="text-3xl font-extrabold text-primary mt-2">
                    {matchResult.match.coverage_percentage.toFixed(1)}%
                  </div>
                  <div className="w-full bg-surface h-2 rounded-full mt-3 overflow-hidden">
                    <div
                      className="bg-primary h-full rounded-full"
                      style={{ width: `${Math.min(matchResult.match.coverage_percentage, 100)}%` }}
                    />
                  </div>
                </div>

                <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
                  <span className="text-xs font-semibold uppercase tracking-wider text-emerald-600">
                    Exact Matches
                  </span>
                  <div className="text-3xl font-extrabold text-foreground mt-2">
                    {matchResult.match.matched_count}
                  </div>
                  <span className="text-xs text-muted mt-1 block">
                    Verified in resume evidence
                  </span>
                </div>

                <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
                  <span className="text-xs font-semibold uppercase tracking-wider text-amber-600">
                    Partial Matches
                  </span>
                  <div className="text-3xl font-extrabold text-foreground mt-2">
                    {matchResult.match.partial_count}
                  </div>
                  <span className="text-xs text-muted mt-1 block">
                    Semantically related skills
                  </span>
                </div>

                <div className="bg-card border border-border rounded-xl p-5 shadow-sm">
                  <span className="text-xs font-semibold uppercase tracking-wider text-red-600">
                    Missing Gaps
                  </span>
                  <div className="text-3xl font-extrabold text-foreground mt-2">
                    {matchResult.match.missing_count}
                  </div>
                  <span className="text-xs text-muted mt-1 block">
                    Areas to bridge before applying
                  </span>
                </div>
              </div>

              {/* Skills Breakdown Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Matched Skills */}
                <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
                  <h3 className="text-sm font-bold text-foreground flex items-center gap-2 pb-2 border-b border-border">
                    <CheckCircle className="w-4 h-4 text-emerald-500" />
                    Matched Skills ({matchResult.match.matched_skills.length})
                  </h3>
                  <div className="space-y-3">
                    {matchResult.match.matched_skills.length === 0 ? (
                      <p className="text-xs text-muted py-2">No direct exact matches found.</p>
                    ) : (
                      matchResult.match.matched_skills.map((s, idx) => (
                        <div key={idx} className="bg-surface border border-border p-3 rounded-lg text-xs space-y-1">
                          <div className="font-semibold text-foreground">{s.requirement}</div>
                          <div className="text-[11px] text-emerald-700 font-mono">
                            Matched: {s.matched_resume_skill}
                          </div>
                          {s.resume_evidence && (
                            <p className="text-[11px] text-muted italic">&ldquo;{s.resume_evidence}&rdquo;</p>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Partial Matches */}
                <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
                  <h3 className="text-sm font-bold text-foreground flex items-center gap-2 pb-2 border-b border-border">
                    <Clock className="w-4 h-4 text-amber-500" />
                    Partial Matches ({matchResult.match.partial_matches.length})
                  </h3>
                  <div className="space-y-3">
                    {matchResult.match.partial_matches.length === 0 ? (
                      <p className="text-xs text-muted py-2">No partial matches.</p>
                    ) : (
                      matchResult.match.partial_matches.map((p, idx) => (
                        <div key={idx} className="bg-surface border border-border p-3 rounded-lg text-xs space-y-1">
                          <div className="font-semibold text-foreground">{p.requirement}</div>
                          <div className="text-[11px] text-amber-700">
                            Mention: {p.candidate_mention}
                          </div>
                          {p.resume_evidence && (
                            <p className="text-[11px] text-muted italic">&ldquo;{p.resume_evidence}&rdquo;</p>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>

                {/* Missing Skills */}
                <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
                  <h3 className="text-sm font-bold text-foreground flex items-center gap-2 pb-2 border-b border-border">
                    <XCircle className="w-4 h-4 text-red-500" />
                    Missing Requirements ({matchResult.match.missing_skills.length})
                  </h3>
                  <div className="space-y-3">
                    {matchResult.match.missing_skills.length === 0 ? (
                      <p className="text-xs text-muted py-2">No missing requirements found!</p>
                    ) : (
                      matchResult.match.missing_skills.map((m, idx) => (
                        <div key={idx} className="bg-surface border border-border p-3 rounded-lg text-xs space-y-1">
                          <div className="font-semibold text-foreground">{m.requirement}</div>
                          <span className="inline-block text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-600 font-semibold uppercase">
                            {m.skill_type}
                          </span>
                          {m.job_evidence && (
                            <p className="text-[11px] text-muted italic">&ldquo;{m.job_evidence}&rdquo;</p>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* Realistic ATS Disclaimer Banner */}
              {matchResult.match.limitations && matchResult.match.limitations.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-5 space-y-2 text-xs text-amber-900">
                  <div className="font-bold flex items-center gap-2 text-amber-700">
                    <ShieldAlert className="w-4 h-4" />
                    ATS &amp; Matching Limitations Disclaimer
                  </div>
                  <ul className="list-disc pl-5 space-y-1 text-foreground/80">
                    {matchResult.match.limitations.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === "qa" && (
        <div className="space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <form onSubmit={handleAskCareer} className="space-y-4">
              <div>
                <label htmlFor="career-query" className="block text-sm font-semibold text-foreground mb-2">
                  Ask questions about your resume or job descriptions
                </label>
                <div className="relative">
                  <input
                    id="career-query"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. What distributed systems projects are listed on my resume? or What are the required qualifications?"
                    className="w-full bg-background border border-border rounded-lg pl-4 pr-10 py-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                  />
                  <Search className="w-5 h-5 text-muted absolute right-3 top-3.5" />
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-4 pt-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-muted">Scope:</span>
                  {["resume", "job_description", "all"].map((type) => (
                    <button
                      key={type}
                      type="button"
                      onClick={() => setDocType(type === "all" ? "" : type)}
                      className={`text-xs px-3 py-1.5 rounded-full font-medium transition-colors ${
                        (type === "all" && !docType) || docType === type
                          ? "bg-primary text-white shadow-sm"
                          : "bg-surface border border-border text-muted hover:text-foreground"
                      }`}
                    >
                      {type === "all" ? "All Documents" : type.replace(/_/g, " ")}
                    </button>
                  ))}
                </div>

                <button
                  type="submit"
                  disabled={qaLoading || !query.trim()}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
                >
                  {qaLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing Documents...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Ask Career Assistant
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {qaError && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
              <div>
                <span className="font-semibold">Career query failed:</span> {qaError}
              </div>
            </div>
          )}

          {qaResult && (
            <div className="space-y-6">
              <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
                <div className="flex items-center gap-2 pb-3 border-b border-border">
                  <FileText className="w-5 h-5 text-primary" />
                  <span className="text-xs font-bold uppercase tracking-wider text-muted">
                    Career Grounded Response
                  </span>
                </div>
                <div className="prose prose-slate max-w-none text-foreground text-sm leading-relaxed whitespace-pre-line">
                  {qaResult.answer}
                </div>
              </div>

              <CitationCard sources={qaResult.sources} title="Resume & Job Description Sources" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
