"use client";

import { useState } from "react";
import { saveUserInterests } from "@/app/actions/trends";
import { DOMAINS, PROFILE_TYPES, type Domain, type ProfileType } from "@/lib/trend-engine/types";

const PROFILE_LABELS: Record<ProfileType, string> = {
  student: "Student",
  developer: "Developer",
  data_scientist: "Data Scientist",
  designer: "Designer",
  entrepreneur: "Entrepreneur",
  finance_professional: "Finance Professional",
  content_creator: "Content Creator",
  researcher: "Researcher",
};

interface InterestsPickerProps {
  onComplete?: () => void;
  onSkip?: () => void;
}

export default function InterestsPicker({ onComplete, onSkip }: InterestsPickerProps) {
  const [profileType, setProfileType] = useState<ProfileType>("developer");
  const [selectedDomains, setSelectedDomains] = useState<Domain[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleDomain(d: Domain) {
    setSelectedDomains((prev) =>
      prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d],
    );
  }

  async function handleSave() {
    if (selectedDomains.length === 0) {
      setError("Pick at least one domain.");
      return;
    }

    setSaving(true);
    const res = await saveUserInterests({
      domains: selectedDomains,
      profile_type: profileType,
    });

    setSaving(false);

    if (res.error) {
      setError(res.error);
    } else {
      onComplete?.();
    }
  }

  return (
    <div className="space-y-6">
      {/* Profile type */}
      <div>
        <p className="text-sm font-medium text-neutral-300 mb-3">I am a…</p>
        <div className="flex flex-wrap gap-2">
          {PROFILE_TYPES.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setProfileType(p)}
              className={`text-xs px-3 py-1.5 rounded-full border transition ${
                profileType === p
                  ? "border-white bg-white text-black"
                  : "border-neutral-700 text-neutral-400 hover:border-neutral-500"
              }`}
            >
              {PROFILE_LABELS[p]}
            </button>
          ))}
        </div>
      </div>

      {/* Domain selection */}
      <div>
        <p className="text-sm font-medium text-neutral-300 mb-1">
          I want to track these domains
        </p>
        <p className="text-xs text-neutral-600 mb-3">Select up to 5. Used to personalize trend results.</p>
        <div className="flex flex-wrap gap-2">
          {DOMAINS.map((d) => {
            const selected = selectedDomains.includes(d);
            const maxed = selectedDomains.length >= 5 && !selected;
            return (
              <button
                key={d}
                type="button"
                onClick={() => !maxed && toggleDomain(d)}
                disabled={maxed}
                className={`text-xs px-3 py-1.5 rounded-full border transition ${
                  selected
                    ? "border-purple-500 bg-purple-900/40 text-purple-300"
                    : maxed
                      ? "border-neutral-800 text-neutral-700 cursor-not-allowed"
                      : "border-neutral-700 text-neutral-400 hover:border-neutral-500"
                }`}
              >
                {d}
              </button>
            );
          })}
        </div>
        <p className="text-[10px] text-neutral-700 mt-2">
          {selectedDomains.length}/5 selected
        </p>
      </div>

      {error && <p className="text-xs text-red-400">{error}</p>}

      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleSave}
          disabled={saving}
          className="flex-1 py-2.5 bg-white text-black text-sm font-medium rounded-lg hover:bg-neutral-200 disabled:opacity-50 transition"
        >
          {saving ? "Saving…" : "Save & Continue →"}
        </button>
        {onSkip && (
          <button
            type="button"
            onClick={onSkip}
            className="px-4 py-2.5 text-sm text-neutral-500 hover:text-neutral-300 transition"
          >
            Skip
          </button>
        )}
      </div>
    </div>
  );
}
