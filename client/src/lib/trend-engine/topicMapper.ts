/* ═══════════════════════════════════════════════════════════
   Trend Engine — Topic Mapper
   Maps raw article titles → normalized topic labels.
   Uses keyword lookup first (fast), Gemini for unknowns.
   ═══════════════════════════════════════════════════════════ */

// keyword → canonical topic label
// Order matters: more specific patterns should come first
const KEYWORD_MAP: Array<[RegExp, string]> = [
  // AI & ML
  [/\b(llm|large language model|gpt|gemini|claude|mistral|llama|chatgpt)\b/i, "Large Language Models"],
  [/\b(agentic ai|ai agent|autonomous agent|multi.?agent)\b/i, "AI Agents"],
  [/\b(deep learning|neural net|transformer model|attention mechanism)\b/i, "Deep Learning"],
  [/\b(machine learning|ml model|scikit|supervised learning|unsupervised)\b/i, "Machine Learning"],
  [/\b(computer vision|image recognition|object detection|diffusion model|stable diffusion|midjourney|dall.e)\b/i, "Computer Vision & GenAI Images"],
  [/\b(artificial intelligence|ai|openai)\b/i, "Artificial Intelligence"],

  // Web Dev
  [/\b(next\.?js|nextjs)\b/i, "Next.js"],
  [/\b(react\.?js|react hooks|react server)\b/i, "React"],
  [/\b(typescript|ts type|type safety)\b/i, "TypeScript"],
  [/\b(javascript|js framework|node\.?js|bun|deno)\b/i, "JavaScript & Runtime"],
  [/\b(tailwind|css framework|shadcn)\b/i, "CSS & UI Frameworks"],
  [/\b(web assembly|wasm)\b/i, "WebAssembly"],
  [/\b(full.?stack|full stack developer)\b/i, "Full-Stack Development"],
  [/\b(web dev|frontend|backend|api design|rest|graphql)\b/i, "Web Development"],

  // Cloud & DevOps
  [/\b(kubernetes|k8s|container orchestration)\b/i, "Kubernetes"],
  [/\b(docker|containerization|dockerfile)\b/i, "Docker & Containers"],
  [/\b(aws|amazon web services|ec2|s3|lambda)\b/i, "AWS"],
  [/\b(google cloud|gcp|gke)\b/i, "Google Cloud"],
  [/\b(azure|microsoft cloud)\b/i, "Azure"],
  [/\b(devops|ci.?cd|github actions|terraform|infrastructure as code)\b/i, "DevOps & CI/CD"],
  [/\b(serverless|edge computing|cloudflare workers|vercel|netlify)\b/i, "Serverless & Edge"],

  // Data
  [/\b(data engineering|data pipeline|apache spark|kafka|airflow|dbt)\b/i, "Data Engineering"],
  [/\b(data science|pandas|numpy|jupyter|data analysis)\b/i, "Data Science"],
  [/\b(sql|postgresql|mysql|database|nosql|mongodb|redis)\b/i, "Databases"],

  // Security
  [/\b(zero.?day|vulnerability|exploit|cve|ransomware|malware)\b/i, "Cybersecurity Threats"],
  [/\b(cybersecurity|infosec|penetration test|devsecops)\b/i, "Cybersecurity"],

  // Languages
  [/\b(rust programming|rust lang|ownership model)\b/i, "Rust"],
  [/\b(golang|go lang|go routine)\b/i, "Go"],
  [/\b(python programming|pypi|pip install)\b/i, "Python"],

  // Mobile
  [/\b(react native|expo framework)\b/i, "React Native"],
  [/\b(flutter|dart lang)\b/i, "Flutter"],
  [/\b(ios development|swift lang|swiftui)\b/i, "iOS Development"],
  [/\b(android development|jetpack compose|kotlin)\b/i, "Android Development"],

  // Blockchain
  [/\b(solidity|smart contract|ethereum|defi|nft|web3|blockchain)\b/i, "Blockchain & Web3"],
  [/\b(bitcoin|cryptocurrency|crypto market)\b/i, "Cryptocurrency"],

  // Career & Meta
  [/\b(system design|distributed system|scalability|microservice)\b/i, "System Design"],
  [/\b(open source|github|git workflow)\b/i, "Open Source"],
  [/\b(software engineering|software development|coding interview)\b/i, "Software Engineering"],
  [/\b(startup|vc funding|product management|saas|founder)\b/i, "Startups & Product"],
];

export function mapTitleToTopic(title: string): string | null {
  for (const [pattern, topic] of KEYWORD_MAP) {
    if (pattern.test(title)) return topic;
  }
  return null;
}

// Batch map multiple titles — returns topic or null for each
export function mapTitlesToTopics(titles: string[]): (string | null)[] {
  return titles.map(mapTitleToTopic);
}

// Get a canonical topic for a user-typed query
export function normalizeQueryToTopic(query: string): string {
  const mapped = mapTitleToTopic(query);
  if (mapped) return mapped;
  // Fall back: title-case the query as-is
  return query
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(" ");
}
