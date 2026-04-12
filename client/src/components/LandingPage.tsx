/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Landing Page
   Personalised Learning OS. Swiss editorial aesthetic,
   biological awareness, content elasticity, trend intelligence.
   ═══════════════════════════════════════════════════════════ */

"use client";
 
import Link from "next/link";
import { useEffect } from "react";
import { useApp } from "@/lib/store";
 
export default function LandingPage() {
  const { state, dispatch } = useApp();

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
    if (savedTheme) {
      dispatch({ type: "SET_THEME", payload: savedTheme });
      document.documentElement.setAttribute("data-theme", savedTheme);
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
       dispatch({ type: "SET_THEME", payload: "dark" });
       document.documentElement.setAttribute("data-theme", "dark");
    }
  }, [dispatch]);

  const toggleTheme = () => {
    const nextTheme = state.theme === "light" ? "dark" : "light";
    dispatch({ type: "SET_THEME", payload: nextTheme });
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme", nextTheme);
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      {/* ── Navigation ── */}
      <header className="nav">
        <div className="nav-inner">
          <Link href="/" className="logo">
            Axiom
          </Link>
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <button 
              onClick={toggleTheme}
              className="btn btn-sm"
              style={{ padding: "4px 8px", fontSize: 9, border: '0.5px solid var(--rule)' }}
            >
              {state.theme === "light" ? "Dark" : "Light"}
            </button>
            <Link
              href="/login"
              className="nav-link"
              style={{ paddingBottom: 2 }}
            >
              Login
            </Link>
            <Link
              href="/onboarding"
              className="nav-link"
              style={{ borderBottom: "0.5px solid var(--ink)", paddingBottom: 2 }}
            >
              Initialize OS
            </Link>
          </div>
        </div>
      </header>
      <div className="mobile-spacer" style={{ height: 60 }} />

      <main>
        {/* ── HERO (WHITE) ── */}
        <section
          className="container landing-hero-grid"
          style={{
            paddingTop: 140,
            paddingBottom: 100,
            display: "grid",
            gridTemplateColumns: "1.2fr 1fr",
            gap: 60,
            alignItems: "center",
            background: 'var(--bg)'
          }}
        >
          <div>
            <div className="meta-text" style={{ marginBottom: 24, color: "var(--vermillion)" }}>
              Version 1.0 — Cognitive Learning OS
            </div>
            <h1 style={{ fontSize: 64, lineHeight: 1.05, marginBottom: 32 }}>
              The first system that knows <em style={{ fontWeight: 400, color: "var(--muted)" }}>how you learn.</em>
            </h1>
            <p
              style={{
                maxWidth: 480,
                color: "var(--muted)",
                marginBottom: 40,
                fontSize: 16
              }}
            >
              Most platforms tell you what to study. Most calendars tell you when you&apos;re free. Axiom is the first system that models your brain to know what to teach, how to format it, and exactly when you&apos;re ready to absorb it.
            </p>
            <div style={{ display: "flex", gap: 24 }}>
                <Link href="/onboarding" className="nav-link" style={{ fontSize: 13, borderBottom: '1.5px solid var(--ink)' }}>
                    Start Learning
                </Link>
                <Link href="#pillars" className="nav-link" style={{ fontSize: 13, color: 'var(--muted)' }}>
                    See Methodology
                </Link>
            </div>
          </div>

          <div style={{ 
            padding: 40, 
            border: '0.5px solid var(--ink)', 
            background: 'var(--bg-card)', 
            position: 'relative' 
          }}>
            <div style={{
                position: 'absolute',
                top: -10,
                left: 20,
                background: 'var(--bg)',
                padding: '0 10px',
                fontSize: 9,
                letterSpacing: '0.1em'
            }}>COGNITIVE STATE: MEASURED</div>
            
            <div className="chart-wrapper" style={{ height: 160 }}>
              <svg
                style={{ width: "100%", height: "100%", overflow: "visible" }}
                viewBox="0 0 1000 140"
                preserveAspectRatio="none"
              >
                <line x1="333" y1="0" x2="333" y2="120" className="chart-grid" />
                <line x1="666" y1="0" x2="666" y2="120" className="chart-grid" />
                <line x1="0" y1="15" x2="1000" y2="15" stroke="#C0392B" strokeWidth="1" strokeDasharray="6 4" />
                <path
                  d="M0,120 L166,120 L250,100 L333,80 L375,50 L416,25 L437,15 L458,25 L500,60 L541,70 L583,40 L604,15 L625,50 L666,80 L708,100 L750,110 L1000,120 Z"
                  className="chart-fill"
                />
                <path d="M424.4,15 L437,0 L449.6,15 Z" className="chart-fill-danger" />
                <path
                  d="M0,120 L166,120 L250,100 L333,80 L375,50 L416,25 L437,0 L458,25 L500,60 L541,70 L583,40 L604,0 L625,50 L666,80 L708,100 L750,110 L1000,120"
                  className="chart-line"
                />
                <text x="430" y="-5" style={{ fontFamily: "var(--mono)", fontSize: 9, fill: "var(--vermillion)", fontWeight: 700 }}>[!] COGNITIVE PEAK</text>
                <text x="335" y="75" style={{ fontFamily: "var(--mono)", fontSize: 9, fill: "var(--muted)" }}>DEEP WORK</text>
              </svg>
            </div>
            <div className="chart-legend" style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 8, fontSize: 10 }}>
                <span>08:00</span>
                <span style={{ color: 'var(--vermillion)' }}>IDEAL WINDOW: HARD CONCEPTS</span>
                <span>20:00</span>
            </div>
          </div>
        </section>

        {/* ── THE CRISIS (BLACK) ── */}
        <section style={{ padding: '120px 0', background: 'var(--ink)', color: 'var(--bg)' }}>
            <div className="container">
                <div className="meta-text" style={{ color: 'var(--rule)', marginBottom: 24 }}>The Structural Failure</div>
                <h2 style={{ fontSize: 48, maxWidth: 800, lineHeight: 1.1, fontFamily: 'var(--serif)' }}>
                    The problem isn&apos;t access to content. It&apos;s the mismatch between <em style={{ fontWeight: 400, color: 'var(--rule)' }}>demand and availability.</em>
                </h2>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 40, marginTop: 80 }}>
                    <div style={{ borderLeft: '0.5px solid var(--rule)', paddingLeft: 24 }}>
                        <h3 style={{ fontSize: 20, marginBottom: 16, color: 'var(--rule)', fontFamily: 'var(--serif)' }}>01. Fixed Formats</h3>
                        <p style={{ fontSize: 13, opacity: 0.8 }}>Textbooks and videos treat every learner identically. If the format doesn&apos;t match your brain&apos;s absorption pattern, you don&apos;t fail—the system does.</p>
                    </div>
                    <div style={{ borderLeft: '0.5px solid var(--rule)', paddingLeft: 24 }}>
                        <h3 style={{ fontSize: 20, marginBottom: 16, color: 'var(--rule)', fontFamily: 'var(--serif)' }}>02. Blind Scheduling</h3>
                        <p style={{ fontSize: 13, opacity: 0.8 }}>Calendars ask when you are free, not when you are ready. Learning Dynamic Programming at 11PM after a full day isn&apos;t a discipline problem—it&apos;s a system design error.</p>
                    </div>
                    <div style={{ borderLeft: '0.5px solid var(--rule)', paddingLeft: 24 }}>
                        <h3 style={{ fontSize: 20, marginBottom: 16, color: 'var(--rule)', fontFamily: 'var(--serif)' }}>03. The Disconnect</h3>
                        <p style={{ fontSize: 13, opacity: 0.8 }}>Your learning roadmap and your daily schedule don&apos;t talk. Your scheduler doesn&apos;t know the cognitive cost, and your course doesn&apos;t know you just hit a 4PM trough.</p>
                    </div>
                </div>
            </div>
        </section>

        {/* ── PILLARS (WHITE) ── */}
        <section className="container" id="pillars" style={{ paddingTop: 120, paddingBottom: 120, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 0, background: 'var(--bg)' }}>
            {[
                { 
                    meta: "LAYER 01 — THE ROADMAP", 
                    title: <>It knows <em>what</em> to teach.</>,
                    desc: "An adaptive agent that builds a personalised, prerequisite-aware sequence from your goals. It continuously re-plans based on what you've actually mastered."
                },
                { 
                    meta: "LAYER 02 — THE ENGINE", 
                    title: <>It knows <em>how</em> you learn.</>,
                    desc: "A generation engine that rewrites any concept into the format that works for you: a story, a 60s bit, or a dense reference—delivered by a 3D avatar."
                },
                { 
                    meta: "LAYER 03 — THE SCHEDULER", 
                    title: <>It knows <em>when</em> you&apos;re ready.</>,
                    desc: "A cognitive load engine that models your mental bandwidth. It places hard concepts at your peaks and light review at your troughs—automatically."
                }
            ].map((pillar, i) => (
                <div key={i} style={{ padding: '0 40px', borderRight: i < 2 ? '0.5px solid var(--rule)' : 'none' }} className="pillar-item">
                    <div className="meta-text" style={{ color: 'var(--vermillion)', marginBottom: 16, fontWeight: 700, fontSize: 9 }}>{pillar.meta}</div>
                    <h3 style={{ fontSize: 28, marginBottom: 24, lineHeight: 1.2, fontFamily: 'var(--serif)' }}>{pillar.title}</h3>
                    <p style={{ fontSize: 13, color: 'var(--muted)' }}>{pillar.desc}</p>
                </div>
            ))}
        </section>

        {/* ── TRENDS (BLACK) ── */}
        <section style={{ padding: '120px 0', background: 'var(--ink)', color: 'var(--bg)' }}>
            <div className="container" style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 80, alignItems: 'center' }}>
                <div style={{ padding: 40, border: '0.5px solid var(--rule)', background: 'rgba(255,255,255,0.03)' }}>
                    <div className="meta-text" style={{ color: 'var(--rule)', marginBottom: 16 }}>Live Intelligence</div>
                    <h3 style={{ fontSize: 32, marginBottom: 24, fontFamily: 'var(--serif)' }}>Never learn<br/><em>stale content.</em></h3>
                    <p style={{ fontSize: 14, opacity: 0.8, lineHeight: 1.8 }}>
                        The field moves faster than any fixed curriculum. Axiom's Trend Engine scans Hacker News, Reddit, and NewsAPI in real-time to detect rising signals.
                    </p>
                    <div style={{ marginTop: 32, paddingTop: 32, borderTop: '0.5px solid var(--rule)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                            <span style={{ fontSize: 10, opacity: 0.6 }}>ACTIVE SIGNALS</span>
                            <span style={{ fontSize: 10, color: 'var(--vermillion)' }}>LIVE</span>
                        </div>
                        <div style={{ fontFamily: 'var(--mono)', fontSize: 11 }}>
                            <div style={{ padding: '8px 0', borderBottom: '0.5px solid rgba(255,255,255,0.1)' }}>→ DeepSeek-V3 Architecture <span style={{ float: 'right', color: '#2ECC71' }}>+84%</span></div>
                            <div style={{ padding: '8px 0', borderBottom: '0.5px solid rgba(255,255,255,0.1)' }}>→ WebGPU Inference <span style={{ float: 'right', color: '#2ECC71' }}>+42%</span></div>
                            <div style={{ padding: '8px 0' }}>→ Agentic Workflows <span style={{ float: 'right', color: '#2ECC71' }}>+112%</span></div>
                        </div>
                    </div>
                </div>
                <div>
                    <div className="meta-text" style={{ color: 'var(--rule)', marginBottom: 24 }}>The Dynamic Roadmap</div>
                    <h2 style={{ fontSize: 42, lineHeight: 1.1, fontFamily: 'var(--serif)', marginBottom: 32 }}>Curriculums that <em>evolve with the internet.</em></h2>
                    <p style={{ fontSize: 16, opacity: 0.8, marginBottom: 40 }}>
                        When a new technology reaches a critical trend threshold, Axiom automatically re-calculates your roadmap, inserting prerequisites and identifying the highest ROI learning path.
                    </p>
                    <Link href="/dashboard/trends" className="nav-link" style={{ color: 'var(--bg)', borderBottom: '1px solid var(--bg)' }}>
                        Explore Trend Intelligence &gt;
                    </Link>
                </div>
            </div>
        </section>

        {/* ── TRANSFORMATION (WHITE) ── */}
        <section style={{ padding: '120px 0', background: 'var(--bg)', borderTop: '0.5px solid var(--rule)', borderBottom: '0.5px solid var(--rule)' }}>
            <div className="container" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 80, alignItems: 'center' }}>
                <div>
                    <div className="meta-text" style={{ marginBottom: 24 }}>Content Elasticity</div>
                    <h2 style={{ fontSize: 38, marginBottom: 24, lineHeight: 1.1, fontFamily: 'var(--serif)' }}>One concept.<br />Infinite <em>resolutions.</em></h2>
                    <p style={{ color: 'var(--muted)', marginBottom: 32 }}>Axiom doesn&apos;t just show you content. It re-synthesizes it. Whether you need a 5-minute deep dive or a 30-second mnemonic, the system adjusts the resolution to match your current bandwidth.</p>
                    <div className="meta-text" style={{ padding: 16, borderLeft: '2px solid var(--vermillion)', background: 'rgba(192, 57, 43, 0.05)' }}>
                        &quot;The format mismatch causes disengagement, not lack of effort.&quot;
                    </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                    <div style={{ padding: 32, border: '0.5px solid var(--ink)', background: 'var(--bg-card)' }}>
                        <div style={{ fontSize: 9, background: 'var(--ink)', color: 'var(--bg)', padding: '4px 8px', display: 'inline-block', marginBottom: 16 }}>THE STORY</div>
                        <p style={{ fontFamily: 'var(--serif)', fontSize: 15, fontStyle: 'italic' }}>&quot;Imagine recursion as a set of Russian dolls, where each doll is a function call waiting for the smaller one to finish...&quot;</p>
                    </div>
                    <div style={{ padding: 32, border: '0.5px solid var(--ink)', background: 'var(--bg-card)' }}>
                        <div style={{ fontSize: 9, background: 'var(--ink)', color: 'var(--bg)', padding: '4px 8px', display: 'inline-block', marginBottom: 16 }}>THE BIT</div>
                        <p style={{ fontWeight: 700 }}>RECURSION: 1. Base Case (Stop) 2. Recursive Step (Continue). It&apos;s a loop that uses a stack instead of a counter.</p>
                    </div>
                    <div style={{ padding: 32, border: '0.5px solid var(--ink)', background: 'var(--bg-card)' }}>
                        <div style={{ fontSize: 9, background: 'var(--ink)', color: 'var(--bg)', padding: '4px 8px', display: 'inline-block', marginBottom: 16 }}>THE REFERENCE</div>
                        <p style={{ fontSize: 11, color: 'var(--muted)' }}>Time Complexity: O(2^n) | Space Complexity: O(n) | Pattern: Divide & Conquer. Use when problem has optimal substructure.</p>
                    </div>
                </div>
            </div>
        </section>

        {/* ── COGNITIVE TRACE (BLACK) ── */}
        <section
          style={{
            paddingTop: 120,
            paddingBottom: 120,
            background: 'var(--ink)',
            color: 'var(--bg)'
          }}
        >
          <div className="container" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 80 }}>
            <div style={{ paddingTop: 20 }}>
                <div className="meta-text" style={{ marginBottom: 16, color: 'var(--rule)' }}>
                The Cognitive Load Engine
                </div>
                <h2 style={{ fontSize: 42, marginBottom: 24, fontFamily: 'var(--serif)', lineHeight: 1.1 }}>
                A scheduler with<br />
                <em style={{ fontWeight: 400, color: 'var(--rule)' }}>biological awareness.</em>
                </h2>
                <p style={{ color: "var(--rule)", opacity: 0.8, maxWidth: 400, fontSize: 16 }}>
                Axiom assigns every task a &quot;Cognitive Weight&quot; (CW). It then scans your energy model to ensure high-CW tasks never land in low-bandwidth windows.
                </p>
            </div>

            <div className="trace-log" style={{ background: 'rgba(255,255,255,0.03)', border: '0.5px solid var(--rule)' }}>
                <div
                className="meta-text"
                style={{ marginBottom: 16, display: "block", color: 'var(--rule)' }}
                >
                Real-time Re-planning Log — April 12
                </div>
                <div className="log-line" style={{ color: 'var(--bg)' }}>
                1. Target: &quot;Learn Dynamic Programming&quot; (CW: 8.7/10)
                </div>
                <div className="log-line" style={{ color: 'var(--bg)' }}>
                2. Detected 4:00 PM Energy Dip: 3.2/10
                </div>
                <div className="log-line rule" style={{ color: 'var(--bg)', borderLeftColor: 'var(--bg)' }}>
                3. RULE: Minimum bandwidth for CW &gt; 8 is 7.5.
                </div>
                <div className="log-line conflict" style={{ borderLeftColor: 'var(--vermillion)' }}>
                4. ALERT: Insufficient bandwidth for DP at 4:00 PM.
                </div>
                <div className="log-line" style={{ color: 'var(--bg)' }}>
                5. Scanning for next peak... Found at 6:30 PM (9.1/10)
                </div>
                <div className="log-line action" style={{ color: 'var(--bg)', borderLeftColor: 'var(--bg)' }}>
                6. → REPLAN: Move DP to 6:30 PM.
                </div>
                <div className="log-line action" style={{ color: 'var(--bg)', borderLeftColor: 'var(--bg)' }}>
                7. → ACTION: Insert &quot;Mnemonic Review&quot; at 4:00 PM (CW: 2.1).
                </div>
                <div className="log-line action" style={{ color: 'var(--bg)', borderLeftColor: 'var(--bg)' }}>
                8. → RESULT: Cognitive Alignment Secured.
                </div>
            </div>
          </div>
        </section>

        {/* ── CTA (WHITE) ── */}
        <section
          id="initialize"
          className="container"
          style={{
            padding: "120px 0",
            textAlign: "center",
            background: 'var(--bg)'
          }}
        >
          <div className="meta-text" style={{ color: 'var(--vermillion)', fontWeight: 700 }}>Finality</div>
          <h1 style={{ fontSize: 56, marginTop: 16, fontFamily: 'var(--serif)', lineHeight: 1.1 }}>
            Stop blaming your discipline.<br />Start fixing the <em>system.</em>
          </h1>
          <p
            style={{
              color: "var(--muted)",
              maxWidth: 600,
              margin: "32px auto 0",
              fontSize: 18
            }}
          >
            Join the generation of learners who use their biology instead of fighting it. Initialize your OS and see what you&apos;re actually capable of.
          </p>
          <Link
            href="/onboarding"
            className="btn"
            style={{
              display: "inline-block",
              marginTop: 48,
              padding: "24px 64px",
              fontSize: 14,
              fontWeight: 700,
              background: 'var(--ink)',
              color: 'var(--bg)'
            }}
          >
            Initialize Axiom OS &gt;
          </Link>
          <div className="meta-text" style={{ marginTop: 60, fontSize: 10, opacity: 0.5 }}>
            40M+ Students | India Skills Report 2026 Compatible | Zero Hallucination Guarantee
          </div>
        </section>
      </main>
    </div>
  );
}
