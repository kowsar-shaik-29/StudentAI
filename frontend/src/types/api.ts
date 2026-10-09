/**
 * TypeScript definitions mapping strictly to StudentAI FastAPI schemas.
 */

export interface HealthResponse {
  status: string;
  version: string;
}

export interface ExecutionTraceItem {
  step: number;
  tool_name: string;
  args: Record<string, any>;
  status: "success" | "failure" | "skipped" | "blocked";
  duration_ms: number;
  error?: string | null;
}

export interface CitationReport {
  is_valid: boolean;
  cited_pages: number[];
  valid_pages?: number[];
  invalid_pages?: number[];
  malformed_citations?: string[];
  errors?: string[];
  warnings?: string[];
}

export interface SourceChunk {
  file_name: string;
  page_number: number;
  score?: number;
  chunk_id?: string;
  text?: string;
  category?: string;
  subject?: string;
  document_type?: string;
}

export interface AgentRunResponse {
  query: string;
  answer: string;
  classification: string;
  workflow: string;
  execution_trace: ExecutionTraceItem[];
  sources: SourceChunk[];
  citations_verified: boolean;
  citation_report?: CitationReport | null;
  limitations: string[];
  total_duration_ms: number;
}

export interface LearningAskResponse {
  intent: string;
  topic: string;
  subtopic: string;
  answer: string;
  questions?: Array<{
    question: string;
    difficulty?: string;
    answer?: string;
  }>;
  sources: SourceChunk[];
}

export interface SubtopicStat {
  topic: string;
  subtopic: string;
  accuracy: number | null;
  weakness_score: number;
  weakness_level: "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_DATA";
  questions_attempted: number;
  interaction_count: number;
}

export interface LearningRecommendation {
  rank: number;
  topic: string;
  subtopic: string;
  priority: "HIGH" | "MEDIUM" | "LOW" | "INSUFFICIENT_DATA";
  weakness_score: number;
  reason: string;
  recommended_action: string;
  source_documents: string[];
}

export interface LearningProfileResponse {
  user_id: string;
  topics: Array<{
    topic: string;
    interactions: number;
    quiz_accuracy: number | null;
    weakness_level: string;
  }>;
  weak_topics: SubtopicStat[];
  strong_topics: SubtopicStat[];
  insufficient_data_topics: SubtopicStat[];
  recommendations: LearningRecommendation[];
  message?: string | null;
}

export interface LearningRecommendationsResponse {
  user_id: string;
  recommendations: LearningRecommendation[];
  message?: string | null;
}

export interface StudyPlanActivity {
  type: string;
  description: string;
  minutes: number;
}

export interface StudyPlanDay {
  day: number;
  focus: Array<{
    topic: string;
    subtopic: string;
    priority: string;
  }>;
  activities: StudyPlanActivity[];
  estimated_minutes: number;
}

export interface StudyPlanResponse {
  plan_id?: string | null;
  subject: string;
  days: number;
  daily_minutes: number;
  plan: StudyPlanDay[];
  message?: string | null;
}

export interface CareerAskResponse {
  query: string;
  answer: string;
  sources: SourceChunk[];
}

export interface MatchedSkill {
  requirement: string;
  matched_resume_skill: string;
  skill_type: string;
  resume_evidence?: string;
}

export interface PartialMatch {
  requirement: string;
  candidate_mention: string;
  resume_evidence?: string;
}

export interface MissingSkill {
  requirement: string;
  skill_type: string;
  job_evidence?: string;
}

export interface JobFitResponse {
  resume: {
    document_id: string;
    file_name: string;
    skill_count: number;
  };
  job: {
    document_id: string;
    file_name: string;
    requirement_count: number;
  };
  match: {
    coverage_percentage: number;
    matched_count: number;
    missing_count: number;
    partial_count: number;
    matched_skills: MatchedSkill[];
    partial_matches: PartialMatch[];
    missing_skills: MissingSkill[];
    limitations: string[];
  };
}

export interface ProjectItem {
  project_id: string;
  project_name: string;
  document_count: number;
  filenames: string[];
}

export interface ProjectListResponse {
  projects: ProjectItem[];
  total: number;
}

export interface ProjectAskResponse {
  query: string;
  project_id?: string | null;
  answer: string;
  ambiguous: boolean;
  available_projects: string[];
  sources: SourceChunk[];
}

export interface DocumentUploadResponse {
  file_name: string;
  document_id: string;
  category: string;
  subject: string;
  document_type: string;
  chunks: number;
  pages: number;
  embeddings: number;
}
