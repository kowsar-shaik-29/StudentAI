"use client";

import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { CheckCircle2, AlertCircle } from "lucide-react";

export function Header() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .getHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 shrink-0">
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-slate-800">Knowledge & Learning Workspace</h1>
      </div>
      <div className="flex items-center gap-4">
        {online === null ? (
          <span className="inline-flex items-center gap-1.5 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-slate-300 animate-pulse" />
            Connecting backend...
          </span>
        ) : online ? (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full border border-emerald-100">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            FastAPI Connected (Port 8000)
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 text-xs font-medium text-amber-700 bg-amber-50 px-2.5 py-1 rounded-full border border-amber-200">
            <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
            Backend Disconnected
          </span>
        )}
      </div>
    </header>
  );
}
