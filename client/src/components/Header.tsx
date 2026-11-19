/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Navigation
   Matches landing.html fixed header: logo left, links right.
   60px height, 0.5px ink border, nothing else.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { useApp } from "@/lib/store";
import { ENERGY_LABELS } from "@/lib/types";
import type { EnergyLevel } from "@/lib/types";
import { logoutUser } from "@/app/actions/auth";

export default function Header({ study = false, dashboardDemo = false }: { study?: boolean; dashboardDemo?: boolean }) {
  const pathname = usePathname();
  const { state, dispatch } = useApp();
  const [theme, setTheme] = useState<"light" | "dark" | "system">("system");

  useEffect(() => {
    if (dashboardDemo) return;
    const saved = localStorage.getItem("axiom-theme");
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
    }
  }, [dashboardDemo]);

  useEffect(() => {
    if (dashboardDemo) return;
    // Sync theme to root element on mount and change
    if (theme !== "system") {
      document.documentElement.setAttribute("data-theme", theme);
      localStorage.setItem("axiom-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
      localStorage.removeItem("axiom-theme");
    }
  }, [theme, dashboardDemo]);

  const toggleTheme = () => {
    if (theme === "system") {
      setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark");
    } else if (theme === "dark") {
      setTheme("light");
    } else {
      setTheme("system"); // cycle back
    }
  };

  if (study || dashboardDemo) {
    return (
      <>
        <header className="nav" data-axiom-demo={dashboardDemo ? "" : undefined}>
          <div className="nav-inner">
            <Link href={dashboardDemo ? "/dashboard/demo" : "/learn"} className="logo">Axiom</Link>
            {dashboardDemo ? (
              <nav aria-label="Dashboard">
                <div style={{ display: "flex", gap: "8px 18px", flexWrap: "wrap" }}>
                  <Link href="/dashboard/demo" className={`nav-link ${pathname === "/dashboard/demo" ? "active" : ""}`}>Dashboard</Link>
                  <Link href="/dashboard/demo/roadmap" className={`nav-link ${pathname === "/dashboard/demo/roadmap" ? "active" : ""}`}>Roadmap</Link>
                  <Link href="/dashboard/demo/discover" className={`nav-link ${pathname === "/dashboard/demo/discover" ? "active" : ""}`}>Discover</Link>
                  <Link href="/dashboard/demo/study" className={`nav-link ${pathname === "/dashboard/demo/study" ? "active" : ""}`}>Study</Link>
                  <Link href="/dashboard/demo#study-history" className="nav-link">Study history</Link>
                </div>
              </nav>
            ) : (
              <div style={{ display: "flex", gap: 24 }}>
                <a href="#study" className="nav-link">Study</a>
                <a href="#schedule" className="nav-link">Schedule</a>
                <a href="#history" className="nav-link">History</a>
              </div>
            )}
          </div>
        </header>
        <div className="mobile-spacer" style={{ height: 60 }} />
      </>
    );
  }

  return (
    <>
      <header className="nav">
        <div className="nav-inner">
          <Link href="/" className="logo">Axiom</Link>

          <div className="nav-links-container" style={{ display: "flex", alignItems: "center", gap: 24 }}>
            {/* Energy indicator — compact */}
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="meta-text">{ENERGY_LABELS[state.energyLevel]}</span>
              <div style={{ display: "flex", gap: 2 }}>
                {([-2, -1, 0, 1, 2] as EnergyLevel[]).map((level) => (
                  <button
                    key={level}
                    onClick={() => {
                      dispatch({ type: "SET_ENERGY", payload: level });
                      dispatch({ type: "ADAPTIVE_RESCHEDULE", payload: { reason: "energy_changed" } });
                    }}
                    style={{
                      width: 14,
                      height: 6,
                      background: state.energyLevel >= level ? "var(--ink)" : "var(--rule)",
                      border: "none",
                      cursor: "pointer",
                      display: "block",
                    }}
                    title={ENERGY_LABELS[level]}
                  />
                ))}
              </div>
            </div>

            {/* Primary nav links */}
            <Link href="/dashboard" className={`nav-link ${pathname === '/dashboard' ? 'active' : ''}`}>Dashboard</Link>
            <Link href="/dashboard/tasks" className={`nav-link ${pathname === '/dashboard/tasks' ? 'active' : ''}`}>Tasks</Link>
            <Link href="/dashboard/roadmap" className={`nav-link ${pathname === '/dashboard/roadmap' ? 'active' : ''}`}>Roadmap</Link>
            <Link href="/dashboard/trends" className={`nav-link ${pathname === '/dashboard/trends' ? 'active' : ''}`}>Trends</Link>
            <Link href="/dashboard/report" className={`nav-link ${pathname === '/dashboard/report' ? 'active' : ''}`}>Report</Link>
            
            {/* Secondary links in a compact group */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginLeft: 8, paddingLeft: 16, borderLeft: "1px solid var(--rule)" }}>
              <Link href="/dashboard/feedback" className={`nav-link ${pathname === '/dashboard/feedback' ? 'active' : ''}`}>Feedback</Link>
              <Link href="/dashboard/tutorial" className={`nav-link ${pathname === '/dashboard/tutorial' ? 'active' : ''}`}>Guide</Link>
              <Link href="/dashboard/profile" className={`nav-link ${pathname === '/dashboard/profile' ? 'active' : ''}`}>Profile</Link>
            </div>

            {/* Action buttons */}
            <button
              className="btn btn-sm"
              onClick={() => dispatch({ type: "TOGGLE_ADD_TASK" })}
            >
              + Task
            </button>
            <button
              onClick={toggleTheme}
              className="nav-link"
              style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center" }}
              title="Toggle Theme"
            >
              {theme === "system" ? "🌓" : theme === "dark" ? "🌙" : "☀️"}
            </button>
            <button
              onClick={() => logoutUser()}
              className="nav-link"
              style={{ background: "none", border: "none", cursor: "pointer", color: "var(--vermillion)" }}
            >
              Logout
            </button>
          </div>
        </div>
      </header>
      {/* Spacer for fixed nav */}
      <div className="mobile-spacer" style={{ height: 60 }} />
    </>
  );
}
