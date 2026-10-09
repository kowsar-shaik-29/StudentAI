import {
  HealthResponse,
  AgentRunResponse,
  LearningAskResponse,
  LearningProfileResponse,
  LearningRecommendationsResponse,
  StudyPlanResponse,
  CareerAskResponse,
  JobFitResponse,
  ProjectListResponse,
  ProjectAskResponse,
  DocumentUploadResponse,
  UserResponse,
} from "../types/api";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      ...options,
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(options?.headers || {}),
      },
    });

    if (!res.ok) {
      let detail = res.statusText;
      try {
        const errorJson = await res.json();
        detail = errorJson.detail || JSON.stringify(errorJson);
      } catch {
        // Fall back to status text
      }
      throw new ApiError(res.status, detail);
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      503,
      `Network connection failure: Cannot reach StudentAI backend at ${BASE_URL}. Ensure FastAPI is running on port 8000.`
    );
  }
}

export const api = {
  getHealth: () => request<HealthResponse>("/health"),

  // Authentication
  register: (username: string, password: string) =>
    request<UserResponse>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  login: (username: string, password: string) =>
    request<UserResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  logout: () =>
    request<{ message: string }>("/api/auth/logout", {
      method: "POST",
    }),

  getMe: () => request<UserResponse>("/api/auth/me"),

  runAgent: (query: string) =>
    request<AgentRunResponse>("/api/agent/run", {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  askLearning: (query: string, top_k = 5, intent?: string, user_id = "default_user") =>
    request<LearningAskResponse>("/api/learning/ask", {
      method: "POST",
      body: JSON.stringify({ query, top_k, intent, user_id }),
    }),

  getLearningProfile: (user_id = "default_user") =>
    request<LearningProfileResponse>(`/api/learning/profile?user_id=${encodeURIComponent(user_id)}`),

  getRecommendations: (user_id = "default_user", limit = 5, topic?: string) => {
    let q = `/api/learning/recommendations?user_id=${encodeURIComponent(user_id)}&limit=${limit}`;
    if (topic) q += `&topic=${encodeURIComponent(topic)}`;
    return request<LearningRecommendationsResponse>(q);
  },

  createStudyPlan: (subject: string, days = 7, daily_minutes = 60, user_id = "default_user") =>
    request<StudyPlanResponse>("/api/study-plans", {
      method: "POST",
      body: JSON.stringify({ subject, days, daily_minutes, user_id }),
    }),

  askCareer: (query: string, top_k = 5, document_type?: string, document_id?: string) =>
    request<CareerAskResponse>("/api/career/ask", {
      method: "POST",
      body: JSON.stringify({ query, top_k, document_type, document_id }),
    }),

  analyzeJobFit: (resume_id: string, job_id: string) =>
    request<JobFitResponse>("/api/career/job-fit", {
      method: "POST",
      body: JSON.stringify({ resume_id, job_id }),
    }),

  listProjects: () => request<ProjectListResponse>("/api/projects"),

  askProject: (query: string, project_id?: string, top_k = 5) =>
    request<ProjectAskResponse>("/api/projects/ask", {
      method: "POST",
      body: JSON.stringify({ query, project_id, top_k }),
    }),

  uploadDocument: async (
    file: File,
    category = "academic",
    subject = "general",
    document_type = "notes"
  ): Promise<DocumentUploadResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", category);
    formData.append("subject", subject);
    formData.append("document_type", document_type);

    const url = `${BASE_URL}/api/documents/upload`;
    try {
      const res = await fetch(url, {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      if (!res.ok) {
        let detail = res.statusText;
        try {
          const errData = await res.json();
          detail = errData.detail || JSON.stringify(errData);
        } catch {
          // ignore
        }
        throw new ApiError(res.status, detail);
      }

      return await res.json();
    } catch (err) {
      if (err instanceof ApiError) {
        throw err;
      }
      throw new ApiError(503, `Upload failed: Cannot connect to StudentAI backend at ${BASE_URL}.`);
    }
  },
};
