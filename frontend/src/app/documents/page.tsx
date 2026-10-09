"use client";

import React, { useState, useRef } from "react";
import { api, ApiError } from "@/lib/api";
import { DocumentUploadResponse } from "@/types/api";
import {
  UploadCloud,
  FileText,
  CheckCircle,
  AlertTriangle,
  FolderOpen,
  Info,
  Layers,
} from "lucide-react";

export default function DocumentsPage() {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("academic");
  const [subject, setSubject] = useState("Computer Science");
  const [docType, setDocType] = useState("lecture_notes");

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<DocumentUploadResponse | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (selectedFile: File) => {
    setError(null);
    setSuccess(null);

    // Extension check (.pdf, .md, .txt)
    const validExts = [".pdf", ".md", ".txt"];
    const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf(".")).toLowerCase();
    if (!validExts.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Please upload a PDF (.pdf), Markdown (.md), or Text (.txt) file.`);
      setFile(null);
      return;
    }

    // Size check: 10MB limit
    const maxSize = 10 * 1024 * 1024;
    if (selectedFile.size > maxSize) {
      setError(`File is too large (${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 10 MB.`);
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || uploading) return;

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const resp = await api.uploadDocument(file, category, subject, docType);
      setSuccess(resp);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError("Document upload failed. Ensure the backend FastAPI server is running.");
      }
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <UploadCloud className="w-8 h-8 text-primary" />
            Document Ingestion Manager
          </h1>
          <p className="text-muted text-sm mt-1">
            Upload course notes, syllabi, resumes, job specs, and project docs for page-aware chunking and vector indexing.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2 Cols): Upload Form */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <form onSubmit={handleUpload} className="space-y-6">
              {/* Drag and Drop Zone */}
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                  dragActive
                    ? "border-primary bg-primary/5"
                    : file
                    ? "border-emerald-500 bg-emerald-50/20"
                    : "border-border hover:border-primary/50 hover:bg-surface"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.md,.txt"
                  onChange={handleFileInput}
                  className="hidden"
                />

                <div className="flex flex-col items-center justify-center gap-2">
                  <div className="p-3 bg-surface rounded-full text-primary border border-border">
                    <FolderOpen className="w-6 h-6" />
                  </div>
                  {file ? (
                    <div>
                      <p className="font-semibold text-foreground text-sm flex items-center justify-center gap-1.5">
                        <FileText className="w-4 h-4 text-emerald-600" />
                        {file.name}
                      </p>
                      <p className="text-xs text-muted mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB &bull; Click or drag another file to replace
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="font-semibold text-foreground text-sm">
                        Click to select or drag and drop a file
                      </p>
                      <p className="text-xs text-muted mt-0.5">
                        Supported: PDF (.pdf), Markdown (.md), Plain text (.txt) &bull; Max 10MB
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Metadata Fields */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label htmlFor="category-select" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                    Category
                  </label>
                  <select
                    id="category-select"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                  >
                    <option value="academic">Academic / Learning</option>
                    <option value="career">Career / Jobs</option>
                    <option value="project">Project / Portfolio</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="subject-input" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                    Subject / Domain
                  </label>
                  <input
                    id="subject-input"
                    type="text"
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    placeholder="e.g. Operating Systems, General"
                    className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                    required
                  />
                </div>

                <div>
                  <label htmlFor="doctype-select" className="block text-xs font-semibold uppercase tracking-wider text-muted mb-2">
                    Document Type
                  </label>
                  <select
                    id="doctype-select"
                    value={docType}
                    onChange={(e) => setDocType(e.target.value)}
                    className="w-full bg-background border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary"
                  >
                    <option value="lecture_notes">Lecture Notes</option>
                    <option value="textbook">Textbook / Readings</option>
                    <option value="syllabus">Syllabus</option>
                    <option value="resume">Resume</option>
                    <option value="job_description">Job Description</option>
                    <option value="project_doc">Project Documentation</option>
                    <option value="notes">General Notes</option>
                  </select>
                </div>
              </div>

              {/* Upload Button */}
              <div className="flex justify-end pt-2 border-t border-border">
                <button
                  type="submit"
                  disabled={uploading || !file}
                  className="inline-flex items-center gap-2 px-6 py-2.5 bg-primary hover:bg-primary-hover disabled:bg-primary/50 text-white font-medium text-sm rounded-lg transition-colors shadow-sm"
                >
                  {uploading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Parsing &amp; Embedding...
                    </>
                  ) : (
                    <>
                      <UploadCloud className="w-4 h-4" />
                      Upload &amp; Index Document
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Feedback states */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-sm text-red-600 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5 text-red-500" />
              <div>
                <span className="font-semibold">Upload error:</span> {error}
              </div>
            </div>
          )}

          {success && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-5 space-y-3">
              <div className="flex items-center gap-2 text-emerald-800 font-bold text-sm">
                <CheckCircle className="w-5 h-5 text-emerald-600" />
                Document successfully ingested and indexed!
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs">
                <div className="bg-card border border-border p-2.5 rounded-lg">
                  <span className="text-muted block text-[10px] uppercase font-semibold">Document ID</span>
                  <span className="font-mono text-foreground truncate block">{success.document_id}</span>
                </div>
                <div className="bg-card border border-border p-2.5 rounded-lg">
                  <span className="text-muted block text-[10px] uppercase font-semibold">Pages</span>
                  <span className="font-semibold text-foreground block">{success.pages}</span>
                </div>
                <div className="bg-card border border-border p-2.5 rounded-lg">
                  <span className="text-muted block text-[10px] uppercase font-semibold">Chunks</span>
                  <span className="font-semibold text-foreground block">{success.chunks}</span>
                </div>
                <div className="bg-card border border-border p-2.5 rounded-lg">
                  <span className="text-muted block text-[10px] uppercase font-semibold">Embeddings</span>
                  <span className="font-semibold text-foreground block">{success.embeddings}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Ingestion Pipeline Explanation */}
        <div className="space-y-6">
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
            <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
              <Layers className="w-5 h-5 text-primary" />
              Ingestion Pipeline
            </h3>
            <p className="text-xs text-muted leading-relaxed">
              Every document you upload goes through the complete StudentAI pipeline without external cloud dependencies:
            </p>

            <div className="space-y-3 text-xs">
              <div className="p-3 bg-surface rounded-lg border border-border">
                <div className="font-semibold text-foreground">1. Page-Aware Extraction</div>
                <p className="text-muted mt-0.5">
                  PyPDF extracts text page by page preserving metadata for exact citation grounding.
                </p>
              </div>

              <div className="p-3 bg-surface rounded-lg border border-border">
                <div className="font-semibold text-foreground">2. Semantic Chunking</div>
                <p className="text-muted mt-0.5">
                  Chunks are built with optimal sliding window overlap to keep context boundaries intact.
                </p>
              </div>

              <div className="p-3 bg-surface rounded-lg border border-border">
                <div className="font-semibold text-foreground">3. Dual Indexing</div>
                <p className="text-muted mt-0.5">
                  Embeddings stored in Qdrant; BM25 inverted index created for keyword search and Reciprocal Rank Fusion.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-surface border border-border rounded-xl p-5 space-y-2 text-xs text-muted">
            <div className="font-semibold text-foreground flex items-center gap-1.5">
              <Info className="w-4 h-4 text-primary" /> Note on Document Types
            </div>
            <p>
              Tagging documents accurately with category and document type ensures queries are correctly routed to the respective Learning, Career, or Project RAG subsystems.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
