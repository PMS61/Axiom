/**
 * RAG Configuration
 * Add RAG_BACKEND_URL to your .env.local file
 * Example: RAG_BACKEND_URL=http://localhost:8000
 */

export const RAG_CONFIG = {
  BACKEND_URL: process.env.RAG_BACKEND_URL || "http://localhost:8000",
  DEFAULT_K: 5,
  MAX_RETRIES: 3,
  TIMEOUT_MS: 10000,
};

export const RAG_ENDPOINTS = {
  UPLOAD_AND_CLUSTER: "/admin/upload_and_cluster",
  LIST_CLUSTERS: "/rag/list_clusters", 
  VIEW_PDF: "/rag/view_pdf",
  GET_CHUNKS: "/rag/chunks",
} as const;
