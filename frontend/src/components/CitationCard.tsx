import { FileText, Bookmark } from "lucide-react";
import { SourceChunk } from "../types/api";

export interface CitationCardProps {
  source?: SourceChunk;
  sources?: SourceChunk[];
  index?: number;
  title?: string;
}

export function CitationCard({ source, sources, index = 1, title }: CitationCardProps) {
  // If a list of sources is provided with an optional title
  if (sources) {
    if (sources.length === 0) return null;
    return (
      <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3">
        <h3 className="text-xs font-bold uppercase tracking-wider text-muted flex items-center gap-1.5 pb-2 border-b border-border">
          <FileText className="w-4 h-4 text-primary" />
          {title || `Cited Sources (${sources.length})`}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {sources.map((src, idx) => (
            <CitationCard key={idx} source={src} index={idx + 1} />
          ))}
        </div>
      </div>
    );
  }

  if (!source) return null;

  return (
    <div className="p-3 bg-slate-50 hover:bg-slate-100/80 transition-colors border border-slate-200/80 rounded-lg text-xs flex flex-col gap-1.5">
      <div className="flex items-center justify-between font-semibold text-slate-800">
        <span className="flex items-center gap-1.5 truncate">
          <FileText className="w-3.5 h-3.5 text-primary shrink-0" />
          <span className="truncate">{source.file_name}</span>
        </span>
        <span className="inline-flex items-center gap-1 bg-white px-2 py-0.5 rounded text-[11px] font-medium text-slate-600 border border-slate-200">
          <Bookmark className="w-3 h-3 text-slate-400" />
          Page {source.page_number}
        </span>
      </div>
      {source.text && (
        <p className="text-slate-600 line-clamp-3 text-[11px] leading-relaxed italic bg-white/60 p-2 rounded border border-slate-100">
          &ldquo;{source.text.trim()}&rdquo;
        </p>
      )}
      <div className="flex items-center gap-3 text-[10px] text-slate-400 font-medium mt-0.5">
        {source.category && <span>Category: {source.category}</span>}
        {source.score !== undefined && <span>Score: {source.score.toFixed(4)}</span>}
      </div>
    </div>
  );
}

export default CitationCard;

