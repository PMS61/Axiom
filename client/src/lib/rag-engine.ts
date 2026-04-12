import { TaskType, TaskPriority } from "./types";
let pdfjsInitialized = false;

async function getPDFJS() {
  const pdfjs = await import('pdfjs-dist/build/pdf.min.mjs');
  if (typeof window !== "undefined" && !pdfjsInitialized) {
    try {
      const lib = (pdfjs as any).default || pdfjs;
      if (lib.GlobalWorkerOptions) {
        // Pointing to local public/pdf.worker.min.mjs (copied in turn)
        lib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';
        pdfjsInitialized = true;
      }
    } catch (e) {
      console.warn("Failed to set PDF.js workerSrc:", e);
    }
  }
  return (pdfjs as any).default || pdfjs;
}

async function getTesseract() {
  return await import('tesseract.js');
}

interface TypeRule {
  pattern: RegExp;
  type: TaskType;
  mul: number;
  durMins: number;
  diffDelta: number;
}

const TYPE_RULES: TypeRule[] = [
  { pattern: /theorem|proof|law|principle|concept|definition|what is|introduction|overview|fundamentals|theory|basis|foundation|understand|module|unit|chapter|lesson|part/i, type: "learning", mul: 1.4, durMins: 60, diffDelta: 0 },
  { pattern: /implement|code|build|write|program|algorithm|solve|exercise|problem|apply|construct|develop|create|design and|compute|derivation|deriving/i, type: "problem_solving", mul: 1.3, durMins: 90, diffDelta: 1 },
  { pattern: /report|essay|summary|document|draft|write up|explain|describe|analysis|write a|prepare a|case study|presentation/i, type: "writing", mul: 1.1, durMins: 75, diffDelta: 0 },
  { pattern: /revise|review|recap|practice problems|test yourself|quiz|re-read|consolidate|reinforce|go over|past papers|mock/i, type: "revision", mul: 0.9, durMins: 45, diffDelta: -1 },
  { pattern: /read|chapter|section|study|textbook|notes|material|reference|literature|go through|watching|video/i, type: "reading", mul: 0.8, durMins: 45, diffDelta: -1 },
];

const DIFF_SIGNALS: [RegExp, number][] = [
  [/advanced|complex|optimis|dynamic programming|distributed|concurrent|recursive|asymptotic|proof|NP-hard|eigenvalue|gradient|backprop|derive/i, 2],
  [/implement|algorithm|analysis|design|architecture|protocol|schema|system/i, 1],
  [/basic|intro|simple|overview|introduction|what is|getting started|fundamentals|survey/i, -1],
];

export function classifyTopic(t: string): TypeRule {
  for (const r of TYPE_RULES) {
    if (r.pattern.test(t)) return r;
  }
  return TYPE_RULES[0];
}

export function scoreDifficulty(t: string, p: number, total: number): number {
  let s = 2;
  for (const [pat, d] of DIFF_SIGNALS) {
    if (pat.test(t)) {
      s += d;
      break;
    }
  }
  s += Math.floor((p / Math.max(1, total)) * 2);
  return Math.min(5, Math.max(1, s));
}

export function computeCL(d: number, m: number, dur: number): number {
  const w = dur <= 30 ? 0.8 : dur <= 60 ? 1 : dur <= 90 ? 1.2 : 1.4;
  return parseFloat((d * m * w).toFixed(2));
}

export function parseDeadline(s: string): Date {
  const td = new Date();
  td.setHours(0, 0, 0, 0);
  const m = s.match(/in\s+(\d+)\s+(day|week|month)/i);
  if (m) {
    const n = +m[1];
    const u = m[2].toLowerCase();
    const d = new Date(td);
    if (u === "day") d.setDate(d.getDate() + n);
    if (u === "week") d.setDate(d.getDate() + n * 7);
    if (u === "month") d.setMonth(d.getMonth() + n);
    return d;
  }
  const p = new Date(s);
  if (!isNaN(p.getTime())) return p;
  const fb = new Date(td);
  fb.setDate(fb.getDate() + 14);
  return fb;
}

export function fmtDate(d: Date): string {
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

// ── Syllabus-First Topic Extraction ──────────────────────
// Priority cascade designed for academic syllabi:
//   1. "Unit/Module/Chapter N: Topic" structured headers
//   2. Numbered items "1.1 Topic" / "1) Topic"
//   3. Bullet / dash items
//   4. "Label: definition" colon splits (take only the label)
//   5. Short Title Case phrases (fallback for bare headers)

function cleanTitle(raw: string): string | null {
  let t = raw.trim();

  // Strip trailing parenthetical fragments like "(e" or "[s"
  t = t.replace(/[\s(\[]+[a-z]?$/i, "").trim();

  // If the text contains "topic definition" (period + lowercase = sentence),
  // cut at the sentence boundary to keep only the heading part
  for (let i = 1; i < t.length - 1; i++) {
    if (t[i] === "." && /[a-z]/.test(t[i + 1])) {
      t = t.slice(0, i).trim();
      break;
    }
  }

  // Also split at question marks — keep the question as the title
  const qIdx = t.indexOf("?");
  if (qIdx > 3 && qIdx < t.length - 2) {
    t = t.slice(0, qIdx + 1).trim();
  }

  // Cap at 8 words max
  const words = t.split(/\s+/);
  if (words.length > 8) t = words.slice(0, 8).join(" ");

  // Length gate
  if (t.length < 3 || t.length > 80) return null;

  // Reject full sentences (common sentence-starting words + long)
  const SENTENCE_STARTERS = /^(the|this|it|a |an |in |on |at |for |to |of |is |are |was|were|has|have|had|we |you |they )/i;
  if (SENTENCE_STARTERS.test(t) && words.length > 4) return null;

  return t;
}

export function extractTopics(text: string): string[] {
  const cleanBuf = text
    .replace(/\[Page\s+\d+\]/gi, "")
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n");

  const lines = cleanBuf.split("\n").map(l => l.trim()).filter(Boolean);
  const topics: string[] = [];

  for (const line of lines) {
    // ── P1: Structured unit/module/chapter headers ──
    // "Unit 1: Introduction to Research"
    // "Module 2 - Research Design"
    // "Chapter 3. Types of Research"
    const unitRe = /^(?:unit|module|chapter|topic|section|lecture|part|lesson)\s*[.\-\s]*[\dIVX]+[.\-:\s]+(.+)/i;
    const unitMatch = line.match(unitRe);
    if (unitMatch) {
      const title = cleanTitle(unitMatch[1]);
      if (title) { topics.push(title); continue; }
    }

    // ── P2: Numbered list items ──
    // "1.1 Introduction" / "1.1.2 Sub-topic" / "1) Topic" / "I. Topic"
    const numRe = /^(?:\d+[.)]\d*[.)]?\d*[.)]?\s+|[IVX]+[.)]\s+)(.+)/;
    const numMatch = line.match(numRe);
    if (numMatch) {
      const title = cleanTitle(numMatch[1]);
      if (title) { topics.push(title); continue; }
    }

    // ── P3: Bullet / dash items ──
    const bulRe = /^[-\u2022*\u2726\u2192\u25B8\u25BA>]\s+(.+)/;
    const bulMatch = line.match(bulRe);
    if (bulMatch) {
      const title = cleanTitle(bulMatch[1]);
      if (title) { topics.push(title); continue; }
    }

    // ── P4: "Label: definition" — extract only the label ──
    // e.g. "Research Design: Research design is the blueprint..."
    const colonIdx = line.indexOf(":");
    if (colonIdx > 3 && colonIdx < 55 && /^[A-Z]/.test(line)) {
      const label = line.slice(0, colonIdx).trim();
      const SKIP_LABELS = /^(note|example|hint|see|ref|source|figure|table|page|date|time|marks|credits|hours|total|max|min)\s*$/i;
      if (!SKIP_LABELS.test(label)) {
        const title = cleanTitle(label);
        if (title) { topics.push(title); continue; }
      }
    }

    // ── P5: Short Title Case phrases (bare headers without markers) ──
    const words = line.split(/\s+/);
    if (
      /^[A-Z]/.test(line) &&
      words.length >= 2 &&
      words.length <= 8 &&
      line.length > 5 &&
      line.length < 60 &&
      !line.endsWith(".")
    ) {
      // Extra: must not look like a running sentence
      const SENTENCE_STARTERS = /^(the|this|it|a |an |in |on |at |for |to |of |is |are )/i;
      if (!SENTENCE_STARTERS.test(line)) {
        const title = cleanTitle(line);
        if (title) topics.push(title);
      }
    }
  }

  // Deduplicate (case-insensitive) while preserving document sequence
  const seen = new Set<string>();
  return topics.filter(t => {
    const key = t.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function chunkText(text: string, size = 280): string[] {
  const paras = text.split(/\n{2,}/).filter(p => p.trim().length > 20);
  const chunks: string[] = [];
  for (const p of paras) {
    if (p.length <= size) {
      chunks.push(p.trim());
      continue;
    }
    const sents = p.split(/(?<=[.!?])\s+/);
    let cur = "";
    for (const s of sents) {
      if (cur.length + s.length > size && cur) {
        chunks.push(cur.trim());
        cur = s;
      } else {
        cur += (cur ? " " : "") + s;
      }
    }
    if (cur.trim()) chunks.push(cur.trim());
  }
  return chunks.filter(c => c.length > 10);
}

export function cosineSim(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

export function topK(q: number[], cE: number[][], chunks: string[], k = 10): { chunk: string; score: number; i: number }[] {
  return cE.map((e, i) => ({ chunk: chunks[i], score: cosineSim(q, e), i }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.min(k, chunks.length));
}

export function buildTasksFromContext(fullText: string, topic: string, deadline: string, subject?: string) {
  let topics = extractTopics(fullText);

  // Fallback: if extraction found fewer than 3 topics, chunk the text
  if (topics.length < 3) {
    const paragraphs = fullText.split(/\n{2,}/).slice(0, 10);
    topics = paragraphs
      .map(p => {
        const clean = p.replace(/\[Page\s+\d+\]/gi, "").trim();
        const firstLine = clean.split("\n")[0].split(/[.:]/)[0].trim();
        return cleanTitle(firstLine);
      })
      .filter((t): t is string => t !== null);
  }

  // Ensure at least one revision and one practice task exist
  if (!topics.some(t => /revise|review|revision/i.test(t))) {
    topics.push(`Review ${topic}`);
  }
  if (!topics.some(t => /exercise|problem|practice|implement/i.test(t))) {
    topics.push(`Practice ${topic}`);
  }

  topics = Array.from(new Set(topics)).slice(0, 14);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const end = parseDeadline(deadline);
  const totalDays = Math.max(1, Math.floor((end.getTime() - today.getTime()) / 86400000));

  const raw = topics.map((t, i) => {
    const cls = classifyTopic(t);
    const diff = Math.min(5, Math.max(1, scoreDifficulty(t, i, topics.length) + cls.diffDelta));
    const etask = computeCL(diff, cls.mul, cls.durMins);
    const posRatio = i / topics.length;

    let priority: 'high' | 'normal' | 'low' = 'low';
    if (diff >= 4) priority = 'high';
    else if (posRatio < 0.3 && (cls.type === 'learning' || cls.type === 'problem_solving')) priority = 'high';
    else if (totalDays <= 7 && posRatio < 0.5) priority = 'high';
    else if (cls.type === 'learning' || cls.type === 'problem_solving') priority = 'normal';
    else if (posRatio > 0.7) priority = 'normal';

    return {
      name: t,
      subject: subject || topic,
      type: cls.type,
      completionTime: cls.durMins,
      difficulty: diff,
      etask,
      multiplier: cls.mul,
      priority,
      sequence: i,
      deadline: ""
    };
  });

  const totalEtask = raw.reduce((s, t) => s + t.etask, 0);
  let cum = 0;

  return raw.map(t => {
    cum += t.etask;
    const offset = Math.max(1, Math.floor((cum / Math.max(totalEtask, 0.1)) * totalDays));
    const d = new Date(today);
    d.setDate(today.getDate() + Math.min(offset, totalDays));
    return { ...t, deadline: d.toISOString() };
  });
}

// ── File Parsing ──────────────────────────────────────────

export async function extractTextFromPDF(file: File, onProgress?: (msg: string) => void): Promise<string> {
  const pdfjs = await getPDFJS();
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;

  if (pdf.numPages > 5) {
    throw new Error(`PDF too large (${pdf.numPages} pages). Axiom Extraction supports up to 5 pages for precise task mapping.`);
  }

  let text = '';

  for (let i = 1; i <= pdf.numPages; i++) {
    if (onProgress) onProgress(`Extracting page ${i}/${pdf.numPages}`);
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((it: any) => it.str)
      .join(' ')
      .replace(/\s+/g, ' ');
    if (pageText.trim()) text += `[Page ${i}] ${pageText.trim()}\n\n`;
  }
  return text.trim() || '(No extractable text found in this PDF)';
}

export async function extractTextFromImage(file: File, onProgress?: (msg: string) => void): Promise<string> {
  const tesseract = await getTesseract();
  const worker = await tesseract.createWorker('eng', 1, {
    logger: (m: { status?: string; progress?: number }) => {
      if (m.status === 'recognizing text' && onProgress) {
        onProgress(`OCR: ${Math.round((m.progress ?? 0) * 100)}%`);
      }
    }
  });

  try {
    const { data: { text } } = await worker.recognize(file);
    return text.trim() || '(No text detected in this image)';
  } finally {
    await worker.terminate();
  }
}

export async function extractTextFromFile(file: File, onProgress?: (msg: string) => void): Promise<string> {
  if (file.type === 'application/pdf') {
    return extractTextFromPDF(file, onProgress);
  } else if (file.type.startsWith('image/')) {
    return extractTextFromImage(file, onProgress);
  } else {
    // Treat as text
    const text = await file.text();
    if (text.length > 50000) {
      throw new Error(`Text too large (${text.length} chars). Axiom supports up to 50,000 characters for optimal generation.`);
    }
    return text;
  }
}
