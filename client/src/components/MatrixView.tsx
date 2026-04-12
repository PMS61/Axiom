/* ═══════════════════════════════════════════════════════════
   THE AXIOM — 96x7 Matrix UI Component
   Represents the 96 15-minute slots across 7 days.
   Supports drag-and-drop scheduling.

   Static row sizing: all 15-minute slots render at fixed
   30px height to keep scheduled tasks visually stable.
   ═══════════════════════════════════════════════════════════ */

"use client";

import { slotToTime, isSlotBlocked } from "@/lib/engine";
import { useApp } from "@/lib/store";

import { Task } from "@/lib/types";
import { useDroppable, useDraggable } from "@dnd-kit/core";

function DraggableMatrixTask({ task, durationSlots, slotHeight, onTaskClick }: { task: any, durationSlots: number, slotHeight: number, onTaskClick?: (task: any) => void }) {
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: task.id,
    data: { task }
  });

  const isCompleted = task.state === "completed";

  const style = {
    position: "absolute" as const,
    top: 2,
    left: 6,
    right: 6,
    height: slotHeight * durationSlots - 4,
    background: isCompleted ? "var(--rule)" : "var(--bg)",
    color: isCompleted ? "var(--muted)" : "var(--fg)",
    border: isCompleted ? "0.5px solid var(--muted)" : "0.5px solid var(--ink)",
    borderTop: isCompleted ? "3px solid var(--muted)" : `3px solid ${Math.abs(task.etask) > 7 ? 'var(--vermillion)' : 'var(--ink)'}`,
    zIndex: transform ? 999 : 10,
    padding: "10px",
    overflow: "hidden",
    cursor: "grab",
    opacity: isCompleted ? 0.6 : 1,
    transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...listeners}
      {...attributes}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 6 }}>
        <div style={{ fontWeight: 600, fontSize: 12, lineHeight: 1.35, wordBreak: "break-word", textDecoration: isCompleted ? "line-through" : "none" }}>{task.name}</div>
        <button
          onPointerDown={(e) => {
            e.stopPropagation();
            if (onTaskClick) onTaskClick(task);
          }}
          style={{
            background: isCompleted ? "var(--muted)" : "var(--ink)",
            color: "var(--bg)",
            border: "none",
            cursor: "pointer",
            padding: "3px 6px",
            fontSize: 10,
            fontWeight: 700,
            fontFamily: "var(--mono)",
            flexShrink: 0,
            opacity: isCompleted ? 0.5 : 1
          }}
          title="View Task Details"
        >
          [i]
        </button>
      </div>
      <div className="meta-text" style={{ marginTop: 6, fontSize: 10 }}>Etask {(task.etask ?? 0).toFixed(1)}</div>
    </div>
  );
}

function DroppableSlot({ dayIdx, actualDayOfWeek, slot, tasksStartingHere, isHour, onTaskClick }: { dayIdx: number, actualDayOfWeek: number, slot: number, tasksStartingHere: any[], isHour: boolean, onTaskClick?: (task: any) => void }) {
  const { state, dispatch } = useApp();
  const id = `slot-${dayIdx}-${slot}`;
  const { isOver, setNodeRef } = useDroppable({ id });

  const isConfirmed = state.confirmedSlots.includes(id);

  const profile = state.userProfile;
  const slotMinutes = slot * 15;

  const blocked = isSlotBlocked(slot, actualDayOfWeek, profile);
  let isSleep = false;
  let blockLabel = "";

  if (profile) {
    const { wakeTime, sleepTime } = profile;
    if (wakeTime !== null && sleepTime !== null) {
      if (sleepTime > wakeTime) {
        if (slotMinutes >= sleepTime || slotMinutes < wakeTime) isSleep = true;
      } else {
        if (slotMinutes >= sleepTime && slotMinutes < wakeTime) isSleep = true;
      }
    }

    if (blocked && !isSleep) {
       const commitment = profile.fixedCommitments?.find((c: any) => c.days.includes(actualDayOfWeek) && slotMinutes >= c.start_min && slotMinutes < c.end_min);
       if (commitment) blockLabel = commitment.title;
    }
  }

  let isPeak = false;
  let isLow = false;
  if (!blocked) {
    for (const w of (profile?.peakFocusWindows || [])) {
      if (slotMinutes >= w.start_min && slotMinutes < w.end_min) { isPeak = true; break; }
    }
    if (!isPeak) {
      for (const w of (profile?.lowEnergyWindows || [])) {
        if (slotMinutes >= w.start_min && slotMinutes < w.end_min) { isLow = true; break; }
      }
    }
  }

  // Visual state
  let bgColor = "transparent";
  let pattern = "none";

  if (isSleep) {
    pattern = "repeating-linear-gradient(45deg, transparent, transparent 6px, var(--rule) 6px, var(--rule) 7px)";
  } else if (blocked) {
    bgColor = "var(--blocked-bg)";
    pattern = `repeating-linear-gradient(-45deg, transparent, transparent 6px, var(--rule) 6px, var(--rule) 6.5px)`;
  } else if (isPeak) {
    bgColor = "var(--peak-bg)";
    // Dots pattern for peak focus windows
    pattern = "radial-gradient(var(--rule) 0.5px, transparent 0.5px)";
  } else if (isLow) {
    bgColor = "var(--low-bg)";
  }

  if (isConfirmed) {
    // Distinct cross-hatch pattern for confirmed slots
    pattern = "repeating-linear-gradient(0deg, transparent, transparent 1px, var(--rule) 1px, var(--rule) 2px), repeating-linear-gradient(90deg, transparent, transparent 1px, var(--rule) 1px, var(--rule) 2px)";
    bgColor = "var(--card-bg)";
  }

  if (isOver) {
    bgColor = blocked ? "var(--vermillion)" : "var(--bg-invert)";
    pattern = "none";
  }

  const subtleZoneLabel = isSleep
    ? ""
    : blocked
      ? (blockLabel || "Reserved")
      : isPeak
        ? "Peak Focus"
        : isLow
          ? "Low Energy"
          : "";

  return (
    <div
      ref={setNodeRef}
      className={`rule-left ${isHour ? "hour-line" : ""}`}
      style={{
        height: 30,
        backgroundColor: bgColor,
        backgroundImage: pattern !== "none" ? pattern : "none",
        backgroundSize: "12px 12px",
        position: "relative",
        transition: "background-color 0.2s, opacity 0.2s",
        cursor: blocked ? "not-allowed" : "crosshair",
        opacity: isSleep ? 0.6 : 1,
        borderBottom: isHour ? "0.5px solid var(--rule)" : "none",
      }}
      title={isConfirmed ? "Confirmed Productive Window" : isSleep ? "Sleep Wakeup/Windown Window" : blocked ? `Blocked: ${blockLabel}` : isPeak ? "Peak Focus Window" : isLow ? "Low Energy Window" : undefined}
      onClick={() => {
        if (!blocked && tasksStartingHere.length === 0) {
          dispatch({ type: "TOGGLE_CONFIRM_SLOT", payload: id });
        }
      }}
    >
      {subtleZoneLabel && tasksStartingHere.length === 0 && (
        <span
          className="meta-text"
          style={{
            position: "absolute",
            left: 6,
            right: 6,
            top: "50%",
            transform: "translateY(-50%)",
            fontSize: 8,
            letterSpacing: 0.4,
            opacity: 0.35,
            textTransform: "uppercase",
            textAlign: "center",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            zIndex: 1,
          }}
        >
          {subtleZoneLabel}
        </span>
      )}
      {tasksStartingHere.map(task => {
        const durationSlots = task.scheduledSlot!.endSlot - task.scheduledSlot!.startSlot;
        return <DraggableMatrixTask key={task.id} task={task} durationSlots={durationSlots} slotHeight={30} onTaskClick={onTaskClick} />;
      })}
    </div>
  );
}

export default function MatrixView({ onTaskClick }: { onTaskClick?: (task: any) => void }) {
  const { state } = useApp();

  const today = new Date();

  const days = Array.from({ length: 7 }).map((_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() + i);
    const dayName = d.toLocaleDateString("en-US", { weekday: 'short' }).toUpperCase();
    const dateStr = `${d.getDate()}/${d.getMonth() + 1}`;
    return {
      label: `${dayName} ${dateStr}`,
      dayOfWeek: d.getDay()
    };
  });

  const totalSlots = 24 * 4;
  const allSlots = Array.from({ length: totalSlots }, (_, slot) => slot);
  const { wakeTime, sleepTime } = state.userProfile ?? {};

  const visibleSlots = (() => {
    if (wakeTime == null || sleepTime == null || wakeTime === sleepTime) {
      return allSlots;
    }

    const wakeSlot = Math.max(0, Math.min(totalSlots - 1, Math.floor(wakeTime / 15)));
    const sleepSlot = Math.max(0, Math.min(totalSlots, Math.ceil(sleepTime / 15)));

    if (sleepSlot > wakeSlot) {
      return Array.from({ length: sleepSlot - wakeSlot }, (_, i) => wakeSlot + i);
    }

    return [
      ...Array.from({ length: totalSlots - wakeSlot }, (_, i) => wakeSlot + i),
      ...Array.from({ length: sleepSlot }, (_, i) => i),
    ];
  })();

  return (
    <div style={{ border: "0.5px solid var(--rule)", background: "var(--card-bg)", overflowX: "auto" }}>
      <div style={{ minWidth: 920 }}>
        {/* Header */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "56px repeat(7, 1fr)",
          borderBottom: "0.5px solid var(--rule)",
          background: "var(--bg)"
        }}>
          <div /> {/* Time gutter corner */}
          {days.map(day => (
            <div key={day.label} className="meta-text rule-left" style={{ padding: "8px 12px", textAlign: "center" }}>
              {day.label}
            </div>
          ))}
        </div>

      {/* Grid container */}
      <div style={{
        height: "auto",
        overflowY: "visible",
        position: "relative"
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "56px repeat(7, 1fr)",
          gridAutoRows: "minmax(30px, auto)"
        }}>
          {visibleSlots.map((slot) => {
            const isHour = slot % 4 === 0;

            return (
              <div key={slot} style={{ display: "contents" }}>
                {/* Time label */}
                <div
                  className={isHour ? "hour-line" : ""}
                  style={{
                    padding: "4px 8px 0 0",
                    textAlign: "right",
                    borderRight: "0.5px solid var(--rule)",
                  }}
                >
                  {isHour && (
                    <span className="time-label">{slotToTime(slot)}</span>
                  )}
                </div>

                {/* Day columns */}
                {days.map((dayInfo, dIdx) => {
                  const dayData = state.scheduledDays.find(d => d.dayOffset === dIdx);
                  const tasksStartingHere = dayData?.sections.flatMap(sec =>
                    sec.tasks.filter(st => st.startSlot === slot)
                      .map(st => {
                        const originalTask = state.tasks.find(t => t.id === st.taskId);
                        if (!originalTask || originalTask.state === "completed") return null;
                        const durSlots = Math.max(1, Math.ceil(st.completionTime / 15));
                        return {
                          ...originalTask,
                          id: st.chunkId || st.taskId,
                          originalTaskId: st.taskId,
                          name: st.taskName,
                          completionTime: st.completionTime,
                          scheduledSlot: {
                            startSlot: st.startSlot!,
                            endSlot: st.startSlot! + durSlots,
                            day: dIdx,
                            fitnessScore: 0,
                            reasoningSteps: []
                          }
                        };
                      })
                      .filter(Boolean) as Task[]
                  ) || [];

                  return (
                    <DroppableSlot
                      key={`${dayInfo.label}-${slot}`}
                      dayIdx={dIdx}
                      actualDayOfWeek={dayInfo.dayOfWeek}
                      slot={slot}
                      tasksStartingHere={tasksStartingHere}
                      isHour={isHour}
                      onTaskClick={onTaskClick}
                    />
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      </div>
      <div className="meta-text" style={{ padding: "8px 12px", borderTop: "0.5px solid var(--rule)", textAlign: "center" }}>
        {visibleSlots.length}×7 Matrix View Active
      </div>
    </div>
  );
}
