"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "../lib/api";
import { useAuth } from "../context/AuthContext";
import { CheckCircle2, AlertCircle, User, LogOut, LogIn } from "lucide-react";

export function Header() {
  const router = useRouter();
  const { user, logout, loading: authLoading } = useAuth();
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .getHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-8 shrink-0">
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-slate-800">Knowledge & Learning Workspace</h1>
      </div>
      <div className="flex items-center gap-5">
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

        <div className="h-4 w-px bg-slate-200" />

        {!authLoading && (
          <div>
            {user ? (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 bg-slate-100 py-1 px-2.5 rounded-full border border-slate-200">
                  <User className="w-3.5 h-3.5 text-slate-500" />
                  <span>{user.username}</span>
                </div>
                <button
                  onClick={handleLogout}
                  title="Sign out"
                  className="text-slate-400 hover:text-rose-600 p-1 rounded transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-xs">
                <Link
                  href="/login"
                  className="flex items-center gap-1 text-slate-600 hover:text-slate-900 font-medium py-1 px-2.5 rounded-lg border border-slate-200 hover:bg-slate-50 transition-colors"
                >
                  <LogIn className="w-3.5 h-3.5" />
                  Sign In
                </Link>
                <Link
                  href="/register"
                  className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-1 px-2.5 rounded-lg transition-colors"
                >
                  Register
                </Link>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
