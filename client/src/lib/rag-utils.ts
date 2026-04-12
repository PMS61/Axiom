"use server";

import { RAG_CONFIG, RAG_ENDPOINTS } from "./rag-config";

interface RAGRequest {
  cluster_id: string;
  prompt: string;
}

interface RAGResponse {
  prompt: string;
  response: string;
  relevant_chunks: string[];
}

interface RAGClusterRequest {
  id: string;
  prompt: string;
  k?: number;
}

/**
 * Fetches RAG chunks from multiple clusters based on the provided cluster IDs and prompt
 * @param clusters - Array of cluster requests with id, prompt, and optional k value
 * @param k - Default number of top chunks to retrieve per cluster (default: 5)
 * @returns Promise<string[]> - Array of relevant chunks from all clusters
 */
export async function getRagChunks(
  clusters: RAGClusterRequest[],
  k: number = RAG_CONFIG.DEFAULT_K
): Promise<string[]> {
  const RAG_BACKEND_URL = RAG_CONFIG.BACKEND_URL;
  
  if (!clusters || clusters.length === 0) {
    console.warn("No clusters provided to getRagChunks");
    return [];
  }

  try {
    // Create requests for each cluster
    const requests = clusters.map(async (cluster) => {
      const request: RAGRequest = {
        cluster_id: cluster.id,
        prompt: cluster.prompt
      };

      const response = await fetch(`${RAG_BACKEND_URL}${RAG_ENDPOINTS.GET_CHUNKS}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: AbortSignal.timeout(RAG_CONFIG.TIMEOUT_MS),
      });

      if (!response.ok) {
        console.error(`Failed to fetch chunks for cluster ${cluster.id}: ${response.status} ${response.statusText}`);
        return [];
      }

      const data: RAGResponse = await response.json();
      const chunksToReturn = data.relevant_chunks.slice(0, cluster.k || k);
      
      return chunksToReturn;
    });

    // Execute all requests in parallel
    const results = await Promise.all(requests);
    
    // Flatten all chunks into a single array
    const allChunks = results.flat();
    
    console.log(`Retrieved ${allChunks.length} chunks from ${clusters.length} clusters`);
    
    return allChunks;
    
  } catch (error) {
    console.error("Error fetching RAG chunks:", error);
    return [];
  }
}

/**
 * Fetches RAG chunks from a single cluster
 * @param clusterId - The cluster ID to fetch chunks from
 * @param prompt - The prompt/query to search for
 * @param k - Number of top chunks to retrieve (default: 5)
 * @returns Promise<string[]> - Array of relevant chunks
 */
export async function getRagChunksFromCluster(
  clusterId: string,
  prompt: string,
  k: number = 5
): Promise<string[]> {
  return getRagChunks([{ id: clusterId, prompt, k }], k);
}

/**
 * Helper function to format RAG chunks into a context string for prompts
 * @param chunks - Array of relevant chunks
 * @param title - Optional title for the context section
 * @returns Formatted context string
 */
export async function formatRagContext(chunks: string[], title: string = "Relevant Context"): Promise<string> {
  if (!chunks || chunks.length === 0) {
    return "";
  }

  const formattedChunks = chunks
    .map((chunk, index) => `${index + 1}. ${chunk.trim()}`)
    .join('\n\n');

  return `
${title}:
${formattedChunks}

Please use the above context to enhance your response with accurate and relevant information.
`;
}

/**
 * List all available clusters from the RAG backend
 * @returns Promise<Array<{cluster_id: string, path: string}>> - Array of available clusters
 */
export async function listRagClusters(): Promise<Array<{cluster_id: string, path: string}>> {
  const RAG_BACKEND_URL = RAG_CONFIG.BACKEND_URL;
  
  try {
    const response = await fetch(`${RAG_BACKEND_URL}${RAG_ENDPOINTS.LIST_CLUSTERS}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(RAG_CONFIG.TIMEOUT_MS),
    });

    if (!response.ok) {
      console.error(`Failed to list clusters: ${response.status} ${response.statusText}`);
      return [];
    }

    const clusters = await response.json();
    return clusters || [];
    
  } catch (error) {
    console.error("Error listing RAG clusters:", error);
    return [];
  }
}

/**
 * Upload a textbook file and create a cluster
 * @param file - The textbook file to upload
 * @returns Promise<{cluster_id: string, file_path: string} | null> - Cluster info or null on failure
 */
export async function uploadTextbookAndCreateCluster(file: File): Promise<{cluster_id: string, file_path: string} | null> {
  const RAG_BACKEND_URL = RAG_CONFIG.BACKEND_URL;
  
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${RAG_BACKEND_URL}${RAG_ENDPOINTS.UPLOAD_AND_CLUSTER}`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(30000), // 30 seconds for file upload
    });

    if (!response.ok) {
      console.error(`Failed to upload textbook: ${response.status} ${response.statusText}`);
      return null;
    }

    const result = await response.json();
    return {
      cluster_id: result.cluster_id,
      file_path: result.file_path
    };
    
  } catch (error) {
    console.error("Error uploading textbook:", error);
    return null;
  }
}
