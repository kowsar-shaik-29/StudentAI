"use client";

import React, { useState, useEffect } from "react";
import { api, ApiError } from "@/lib/api";
import { ProjectListResponse, ProjectAskResponse } from "@/types/api";
import CitationCard from "@/components/CitationCard";
import {
  FolderGit2,
  Search,
  Sparkles,
  AlertTriangle,
  Folder,
  FileCode,
  Layers,
  HelpCircle,
} from "lucide-react";

export default function ProjectsPage() {
  const [projectsData, setProjectsData] = useState<ProjectListResponse | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const [selectedProject, setSelectedProject] = useState<string>("");

  // Project Q&A state
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProjectAskResponse | null>(null);

  useEffect(() => {
    async function loadProjects() {
      try {
        const data = await api.listProjects();
        setProjectsData(data);
        if (data.projects.length > 0) {
          setSelectedProject(data.projects[0].project_id);
        }
      } catch {
        // Backend offline or no projects
      } finally {
        setLoadingProjects(false);
      }
    }
    loadProjects();
  }, []);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const resp = await api.askProject(query, selectedProject || undefined);
      setResult(resp);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Failed to query project assistant.");
      }
    } finally {
      setLoading(false);
    }
  };

  const setPresetQuery = (presetText: string) => {
    setQuery(presetText);
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <FolderGit2 className="w-8 h-8 text-primary" />
            Project RAG &amp; Interview Prep
          </h1>
          <p className="text-muted text-sm mt-1">
            Grounded architectural deep-dives, design trade-off explanations, and behavioral/technical interview rehearsal.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Project Explorer & Query */}
        <div className="lg:col-span-2 space-y-6">
          {/* Query Form */}
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
            <form onSubmit={handleAsk} className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label htmlFor="project-select" className="text-xs font-semibold uppercase tracking-wider text-muted">
                    Target Project Context
                  </label>
                  {projectsData && (
                    <span className="text-xs text-muted">
                      {projectsData.total} projects indexed
                    </span>
                  )}
                </div>

                <select
                  id="project-select"
                  value={selectedProject}
                  onChange={(e) => setSelectedProject(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg px-4 py-2.5 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary mb-4"
                >
                  <option value="">Auto-detect project from query</option>
                  {projectsData?.projects.map((p) => (
                    <option key={p.project_id} value={p.project_id}>
                      {p.project_name} ({p.document_count} docs)
                    </option>
                  ))}
                </select>

                <label htmlFor="project-query" className="block text-sm font-semibold text-foreground mb-2">
                  Ask about project architecture, technical decisions, or interview questions
                </label>
                <div className="relative">
                  <input
                    id="project-query"
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g. Generate 3 technical interview questions for this project, or Explain the database choices..."
                    className="w-full bg-background border border-border rounded-lg pl-4 pr-10 py-3 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                  />
                  <Search className="w-5 h-5 text-muted absolute right-3 top-3.5" />
                </div>
              </div>

              {/* Quick Prompt Starters */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs font-semibold text-muted mr-1">Starters:</span>
                {[
                  "Generate system design interview questions",
                  "Explain the key architectural trade-offs",
                  "What were the hardest technical challenges?",
                  "Summarize this project in 3 bullets",
                ].map((item, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setPresetQuery(item)}
                    className="text-xs px-2.5 py-1 rounded bg-surface border border-border text-muted hover:text-foreground hover:bg-slate-100 transition-colors"
                  >
                    {item}
                  </button>
                ))}
              </div>

              <div className="flex justify-end pt-2">
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Analyzing Architecture...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Ask Project Assistant
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
              <div>
                <span className="font-semibold">Query failed:</span> {error}
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-6">
              <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-border">
                  <div className="flex items-center gap-2">
                    <Layers className="w-5 h-5 text-primary" />
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      {result.project_id ? `Project: ${result.project_id}` : "Grounded Project Synthesis"}
                    </span>
                  </div>
                  {result.ambiguous && (
                    <span className="text-xs px-2.5 py-1 bg-amber-500/10 text-amber-700 font-medium rounded-full">
                      Ambiguous project scope
                    </span>
                  )}
                </div>

                <div className="prose prose-slate max-w-none text-foreground text-sm leading-relaxed whitespace-pre-line">
                  {result.answer}
                </div>
              </div>

              <CitationCard sources={result.sources} title="Project Docs & Code Artifacts" />
            </div>
          )}
        </div>

        {/* Right Column (1 Col): Indexed Project Repository List */}
        <div className="space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <Folder className="w-5 h-5 text-primary" />
              Available Projects
            </h3>

            {loadingProjects ? (
              <div className="text-xs text-muted animate-pulse py-4 text-center">
                Loading projects list...
              </div>
            ) : projectsData && projectsData.projects.length > 0 ? (
              <div className="space-y-3">
                {projectsData.projects.map((p) => (
                  <div
                    key={p.project_id}
                    onClick={() => setSelectedProject(p.project_id)}
                    className={`p-3 rounded-lg border text-xs cursor-pointer transition-colors ${
                      selectedProject === p.project_id
                        ? "bg-primary/5 border-primary"
                        : "bg-surface border border-border hover:border-border-hover"
                    }`}
                  >
                    <div className="flex items-center justify-between font-semibold text-foreground">
                      <span className="flex items-center gap-1.5">
                        <FileCode className="w-4 h-4 text-primary" />
                        {p.project_name}
                      </span>
                      <span className="text-[10px] bg-card px-2 py-0.5 rounded border border-border text-muted">
                        {p.document_count} doc{p.document_count !== 1 ? "s" : ""}
                      </span>
                    </div>

                    {p.filenames && p.filenames.length > 0 && (
                      <div className="mt-2 text-[11px] text-muted space-y-0.5 truncate">
                        {p.filenames.map((fn, fIdx) => (
                          <div key={fIdx} className="truncate">
                            &bull; {fn}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-muted py-4 text-center space-y-2">
                <p>No project documents indexed yet.</p>
                <p className="text-[11px] text-muted">
                  Upload project documentation or design specs in the Documents tab under category &ldquo;project&rdquo;.
                </p>
              </div>
            )}
          </div>

          <div className="bg-primary/5 border border-primary/20 rounded-xl p-5 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-primary flex items-center gap-1.5">
              <HelpCircle className="w-4 h-4" />
              Interview Preparation
            </h4>
            <p className="text-xs text-foreground/80 leading-relaxed">
              Use this workspace to simulate tough hiring manager inquiries: why you chose specific libraries, how you handled state management, or how the architecture scales.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
