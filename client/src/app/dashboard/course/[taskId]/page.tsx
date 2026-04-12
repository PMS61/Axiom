"use client";

import React, { useEffect, useState } from "react";
import { useApp } from "@/lib/store";
import { generateContentAction, completeNodeAction } from "@/app/actions/roadmap";
import { getCourseAction, saveCourseAction } from "@/app/actions/courses";
import { getTasks } from "@/app/actions/tasks";
import Header from "@/components/Header";
import SlideViewer from "@/components/SlideViewer";
import { useRouter } from "next/navigation";
import { useToast } from "@/components/ui/ToastProvider";

export default function CourseGenerationPage({ params }: { params: Promise<{ taskId: string }> }) {
  const unwrappedParams = React.use(params);
  const router = useRouter();
  const { state, dispatch } = useApp();
  const { addToast } = useToast();
  const [task, setTask] = useState<any>(null);
  
  const [loading, setLoading] = useState(false);
  const [completing, setCompleting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [hasSavedCourse, setHasSavedCourse] = useState(false);
  const [currentSlideIndex, setCurrentSlideIndex] = useState(0);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!result?.slides) return;
      if (e.key === 'ArrowRight') {
        setCurrentSlideIndex((prev) => Math.min(prev + 1, Math.max(0, (result.slides?.length || 1) - 1)));
      } else if (e.key === 'ArrowLeft') {
        setCurrentSlideIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter' && currentSlideIndex === Math.max(0, (result.slides?.length || 1) - 1)) {
        handleComplete();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [result, currentSlideIndex]);

  const handleDownloadPDF = () => {
    window.print();
  };

  useEffect(() => {
    async function initTask() {
      if (state.tasks && state.tasks.length > 0) {
        const foundTask = state.tasks.find((t: any) => t.id === unwrappedParams.taskId);
        if (foundTask) {
          setTask(foundTask);
        } else {
          setError("Task not found.");
        }
      } else {
        // Fallback fetch for direct navigation
        try {
          const res = await getTasks();
          if (res.tasks) {
            const foundTask = res.tasks.find((t: any) => t.id === unwrappedParams.taskId);
            if (foundTask) {
              setTask(foundTask);
            } else {
              setError("Task not found.");
            }
          } else {
             setError("No tasks found in database.");
          }
        } catch (err: any) {
          setError("Error loading tasks: " + err.message);
        }
      }
    }
    initTask();
  }, [state.tasks, unwrappedParams.taskId]);

  useEffect(() => {
    async function loadSavedCourse() {
      if (task) {
        try {
          const res = await getCourseAction(unwrappedParams.taskId);
          if (res.courseData) {
            setResult(res.courseData);
            setHasSavedCourse(true);
          }
        } catch (err) {
          console.error("Failed to load saved course:", err);
        }
      }
    }
    if (!result && !hasSavedCourse) {
      loadSavedCourse();
    }
  }, [task, unwrappedParams.taskId, hasSavedCourse, result]);

  async function handleComplete() {
    if (completing) return;
    setCompleting(true);
    setError(null);

    try {
      const res = await completeNodeAction(unwrappedParams.taskId);
      if (res.success) {
        addToast("Course marked complete. Roadmap unlocked and updated.", "success");
        dispatch({
          type: "UPDATE_TASK_STATE",
          payload: { taskId: unwrappedParams.taskId, state: "completed" },
        });

        if (res.roadmap) {
          dispatch({ type: "SET_ROADMAP", payload: res.roadmap });
        }

        router.push("/dashboard/roadmap");
      } else {
        const message = res.error || "Failed to complete course";
        setError(message);
        addToast(message, "error");
        setCompleting(false);
      }
    } catch (e) {
      setError("System error during completion");
      addToast("System error during completion", "error");
      setCompleting(false);
    }
  }

  async function handleGenerate(e?: React.FormEvent) {
    if (e) e.preventDefault();
    if (!task) return;
    
    setLoading(true);
    setResult(null);
    setError(null);
    addToast(`Generation started for ${task?.name}. You can keep using the app.`, "info");
    
    try {
      const response = await generateContentAction(
        task.name,
        task.subject || task.name,
        "ppt" // Overriding to always output ppt for now
      );
      if (response.data) {
        setResult(response.data);
        setCurrentSlideIndex(0); // Reset on new generation
        setHasSavedCourse(true);
        addToast("Course generation completed.", "success");
        const saveRes = await saveCourseAction(unwrappedParams.taskId, response.data);
        if (saveRes.error) {
          console.warn("Failed to persist course:", saveRes.error);
        }
      } else {
        const message = response.error || "Generation returned empty.";
        setError(message);
        addToast(message, "error");
      }
    } catch (e: any) {
      const message = e.message || "An error occurred";
      setError(message);
      addToast(message, "error");
    } finally {
      setLoading(false);
    }
  }

  if (!task && !error) {
    return (
      <>
        <Header />
        <div style={{ padding: 40, fontFamily: "var(--mono)", color: "var(--muted)" }}>Loading task details...</div>
      </>
    );
  }

  return (
    <>
      <Header />
      
      <div style={{ paddingTop: 60, minHeight: "100vh", backgroundColor: "var(--bg)" }}>
        <section className="container section-rule" style={{ paddingTop: 40, paddingBottom: 40 }}>
          <button 
            onClick={() => router.back()} 
            style={{ background: "none", border: "none", color: "var(--muted)", textDecoration: "underline", cursor: "pointer", fontFamily: "var(--mono)", fontSize: 12, padding: 0, marginBottom: 24 }}
          >
            ← BACK TO ROADMAP
          </button>
          
          <div style={{ marginBottom: 40 }}>
            <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: 16 }}>
              <h1 style={{ fontFamily: "var(--font-playfair)", fontSize: 36, letterSpacing: -0.5, color: "var(--ink)", margin: 0 }}>
                {task?.name}
              </h1>
              <div style={{
                padding: "4px 8px",
                border: "0.5px solid var(--ink)",
                fontSize: 10,
                fontFamily: "var(--mono)",
                color: "var(--ink)"
              }}>
                {task?.contentType?.toUpperCase() || "PPT"}
              </div>
            </div>
            
            <div className="meta-text print-hide" style={{ color: "var(--muted)" }}>
              {task?.subject ? `SUBJECT: ${task.subject}` : "Generative Content Creation"}
            </div>
            <div className="meta-text print-hide" style={{ color: "var(--muted)", marginTop: 4 }}>
              EST. DURATION: {task?.duration} MINS
            </div>
          </div>

          <form onSubmit={handleGenerate} className="print-hide" style={{ display: "flex", flexDirection: "column", gap: 32, maxWidth: 600 }}>
            {loading && (
              <div className="meta-text" style={{ color: "var(--muted)" }}>
                GENERATING IN BACKGROUND... FEEL FREE TO NAVIGATE.
              </div>
            )}
            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
              style={{ opacity: loading ? 0.5 : 1, width: "fit-content" }}
            >
              {loading 
                ? `GENERATING ${task?.contentType?.toUpperCase()}...` 
                : hasSavedCourse 
                  ? `REGENERATE ${task?.contentType?.toUpperCase() || "CONTENT"}`
                  : `GENERATE ${task?.contentType?.toUpperCase() || "CONTENT"}`
              }
            </button>
          </form>
        </section>

        {error && (
          <section className="container section-rule" style={{ paddingTop: 20, paddingBottom: 20 }}>
            <div className="meta-text" style={{ color: "var(--vermillion)" }}>
              ERR: {error}
            </div>
          </section>
        )}

        {result && (
          <section
            className="container"
            style={{
              paddingTop: 40,
              paddingBottom: 80,
              maxWidth: "min(96vw, 1680px)",
            }}
          >


            <div className="print-hide">
              <div className="meta-text" style={{ marginBottom: 16 }}>
                01. PRESENTATION (SLIDE {currentSlideIndex + 1} OF {result.slides?.length || 0})
              </div>
              
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 32 }}>
                <div style={{
                  width: "min(96vw, 1600px)",
                  aspectRatio: "16 / 9",
                  maxHeight: "78vh",
                  border: "0.5px solid var(--rule)",
                  backgroundColor: "var(--card-bg)",
                  padding: "24px",
                  boxSizing: "border-box",
                  display: "flex",
                  flexDirection: "column",
                  overflow: "auto",
                  position: "relative"
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "0.5px solid var(--rule)", paddingBottom: 16, marginBottom: 24, flexShrink: 0 }}>
                    <div className="meta-text" style={{ color: "var(--ink)" }}>ITEM NO. {(currentSlideIndex + 1).toString().padStart(2, "0")}</div>
                    <div className="meta-text" style={{ color: "var(--muted)" }}>TYPE: {result.slides[currentSlideIndex].type || task?.contentType}</div>
                  </div>
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "auto", minHeight: 0 }}>
                    <SlideViewer slide={result.slides[currentSlideIndex]} />
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
                  <button
                    onClick={() => setCurrentSlideIndex(Math.max(0, currentSlideIndex - 1))}
                    disabled={currentSlideIndex === 0 || completing}
                    style={{ background: "none", border: "1px solid var(--rule)", cursor: currentSlideIndex === 0 ? "not-allowed" : "pointer", color: "var(--ink)", opacity: currentSlideIndex === 0 ? 0.2 : 1, fontSize: 13, fontFamily: "var(--mono)", letterSpacing: "0.1em", padding: "12px 24px", borderRadius: "0px" }}
                  >
                    ← PREV
                  </button>

                  <div className="meta-text" style={{ color: "var(--muted)", minWidth: 120, textAlign: "center", fontFamily: "var(--mono)", fontSize: 11 }}>
                    {currentSlideIndex + 1} / {result.slides?.length || 0}
                  </div>

                  {currentSlideIndex >= Math.max(0, (result.slides?.length || 1) - 1) ? (
                    <button
                      onClick={handleComplete}
                      disabled={completing}
                      style={{ 
                        background: "#10b981", // Emerald Green for high visibility
                        border: "none", 
                        cursor: "pointer", 
                        color: "#FFFFFF", 
                        fontSize: 13, 
                        fontFamily: "var(--mono)",
                        letterSpacing: "0.15em",
                        padding: "12px 32px", 
                        borderRadius: "4px",
                        fontWeight: "bold",
                        transition: "all 0.3s ease",
                        opacity: completing ? 0.5 : 1,
                        animation: "pulse 2s infinite",
                        display: "flex",
                        alignItems: "center",
                        gap: 8
                      }}
                    >
                      {completing ? "COMPLETING…" : "COMPLETE & UNLOCK NEXT ✓"}
                    </button>
                  ) : (
                    <button
                      onClick={() => setCurrentSlideIndex(Math.min(Math.max(0, (result.slides?.length || 1) - 1), currentSlideIndex + 1))}
                      style={{ background: "none", border: "1px solid var(--rule)", cursor: "pointer", color: "var(--ink)", fontSize: 13, fontFamily: "var(--mono)", letterSpacing: "0.1em", padding: "12px 24px", borderRadius: "0px" }}
                    >
                      NEXT →
                    </button>
                  )}
                </div>
              </div>

              <div style={{ textAlign: "center", marginTop: 60, padding: 40, borderTop: "0.5px solid var(--rule)" }}>
                <div className="meta-text" style={{ marginBottom: 24, fontSize: 13, color: "var(--ink)" }}>COURSE TERMINATION</div>
                <div style={{ display: "flex", gap: 16, justifyContent: "center", alignItems: "center" }}>
                  <button onClick={handleDownloadPDF} className="btn" style={{ minWidth: 160 }}>EXPORT AS PDF</button>
                  <button 
                    onClick={handleComplete} 
                    disabled={completing}
                    style={{ 
                      minWidth: 160,
                      background: currentSlideIndex >= Math.max(0, (result.slides?.length || 1) - 1) ? "#10b981" : "none",
                      border: "1px solid " + (currentSlideIndex >= Math.max(0, (result.slides?.length || 1) - 1) ? "#10b981" : "var(--rule)"),
                      color: currentSlideIndex >= Math.max(0, (result.slides?.length || 1) - 1) ? "white" : "var(--ink)",
                      padding: "12px 24px",
                      fontFamily: "var(--mono)",
                      fontSize: 12,
                      cursor: "pointer",
                      opacity: completing ? 0.5 : 1
                    }}
                  >
                    {completing ? "FINISHING..." : "FINISH COURSE EARLY"}
                  </button>
                </div>
                <div className="meta-text" style={{ marginTop: 12, color: "var(--muted)" }}>PRO TIP: USE ← AND → ARROW KEYS TO NAVIGATE (OR PRESS ENTER TO FINISH)</div>
              </div>
            </div>

            <div className="print-only">
               <style dangerouslySetInnerHTML={{__html: `
                 .print-only {
                    position: absolute;
                    left: -10000px;
                    top: 0;
                 }
                 @media print {
                   html, body { 
                     margin: 0 !important; 
                     padding: 0 !important; 
                     width: 100% !important;
                     height: 100% !important;
                     background: var(--card-bg) !important; 
                     color: var(--ink) !important; 
                     -webkit-print-color-adjust: exact !important;
                     print-color-adjust: exact !important;
                     overflow: hidden !important;
                   }
                   @page { size: landscape; margin: 0; }
                   .print-only { 
                      position: absolute !important; 
                      top: 0 !important;
                      left: 0 !important;
                      width: 100vw !important;
                      height: 100vh !important;
                      display: block !important;
                   }
                   .print-hide { display: none !important; }
                   .nav, header, button { display: none !important; }
                   
                   /* Perfect center 16:9 block for printing */
                   /* Robust full-page blocks for printing */
                   .print-page-wrapper {
                      width: 100vw;
                      height: 100vh;
                      display: flex;
                      flex-direction: column;
                      page-break-after: always;
                      page-break-inside: avoid;
                      overflow: hidden;
                      background: var(--card-bg) !important;
                      margin: 0 !important;
                      padding: 0 !important;
                   }
                   .print-slide-container {
                      width: 100%;
                      height: 100%;
                      background: var(--card-bg) !important;
                      padding: 50px; /* Generous padding for content safety */
                      box-sizing: border-box;
                      display: flex;
                      flex-direction: column;
                      overflow: hidden;
                      border: none !important;
                   }
                 }
               `}} />
               {(result.slides || []).map((slide: any, index: number) => (
                  <div key={`print-${index}`} className="print-page-wrapper">
                    <div className="print-slide-container">
                       <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "0.5px solid var(--rule)", paddingBottom: 16, marginBottom: 16, flexShrink: 0 }}>
                          <div className="meta-text" style={{ color: "var(--ink)", fontSize: 12 }}>ITEM NO. {(index + 1).toString().padStart(2, "0")}</div>
                          <div className="meta-text" style={{ color: "var(--muted)", fontSize: 12 }}>TYPE: {slide.type || task?.contentType}</div>
                       </div>
                       <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
                         <SlideViewer slide={slide} />
                       </div>
                    </div>
                  </div>
               ))}
            </div>
          </section>
        )}
      </div>
      {/* ── Inline CSS for animations ── */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateX(-8px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </>
  );
}
