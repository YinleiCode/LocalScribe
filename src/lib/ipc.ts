// Typed wrappers around the LocalScribe backend.
// Tauri builds use native `invoke`; browser builds use the local FastAPI bridge.

import { invokeCommand, listenEvent, type UnlistenFn } from "./runtime";
export { assertDiarizationPreservesTranscript } from "./diarization-partition";

export type SpeakerOverlapCandidate = {
  start: number;
  end: number;
  primary_speaker: string;
  secondary_speaker: string;
  confidence: number;
  window_ratio: number;
  context_score: number;
  candidate_score: number;
  source: "osd_campp_context_v1" | string;
};

export type Segment = {
  start: number;
  end: number;
  text: string;
  original_text?: string;
  sync_cues?: Array<{ start: number; end: number; text: string; reliable?: boolean }>;
  speaker?: string; // 由 diarization 设置
  speaker_confidence?: number;
  speaker_votes?: Record<string, number>;
  speaker_subsegments?: Array<{ start: number; end: number; speaker: string; duration: number }>;
  speaker_change_points?: number[];
  speaker_cues?: Array<{
    cue_index: number;
    start: number;
    end: number;
    text?: string;
    speaker: string;
    confidence: number;
    source: "short_window_projection" | string;
    votes?: Record<string, number>;
    review?: boolean;
  }>;
  speaker_cue_embeddings?: Array<{
    cue_index: number;
    start: number;
    end: number;
    speaker: string;
    score: number;
    margin: number;
    voice_coverage_seconds: number;
    voice_coverage_ratio: number;
    overlap_ratio: number;
    decision: string;
    source: string;
    embedding_scope: string;
    second_score?: number;
    second_speaker?: string;
  }>;
  speaker_cue_review?: boolean;
  speaker_cue_mode?: string;
  speaker_cue_split?: boolean;
  speaker_overlap_risk?: boolean;
  overlap_ratio?: number;
  speaker_overlap_confidence?: number;
  speaker_overlap_candidates?: SpeakerOverlapCandidate[];
  speaker_overlap_ratio?: number;
  speaker_resegmented?: boolean;
  speaker_resegmentation_review?: boolean;
  speaker_handoff_review?: boolean;
  speaker_assignment_review?: boolean;
  speaker_review_reason?: string;
  speaker_handoff_voice_guard_repaired?: boolean;
  voice_pitch_hz?: number;
  voice_pitch_confidence?: number;
  voice_band?: "low" | "mid" | "high" | "unknown" | string;
  speaker_handoff_split?: boolean;
  speaker_handoff_bridge?: boolean;
  speaker_handoff_relabel?: boolean;
  speaker_handoff_text_review?: boolean;
  speaker_split_from_index?: number;
  voice_band_repaired?: boolean;
  continuity_repaired?: boolean;
  original_speaker?: string;
  speaker_voiceprint_reidentified?: boolean;
  speaker_voiceprint_review?: boolean;
  speaker_voiceprint_score?: number;
  speaker_voiceprint_anchor?: string;
  speaker_calibrated?: boolean;
  speaker_calibration_source?: string;
  voice_line_refined?: boolean;
  voice_line_review?: boolean;
};

export type SpeakerVoiceProfile = {
  pitch_hz?: number;
  pitch_confidence?: number;
  voice_band?: string;
  samples?: number;
};

export type SpeakerVoiceMixSummary = {
  dominant_band?: string;
  minority_band?: string;
  duration_by_band?: Record<string, number>;
  segments_by_band?: Record<string, number>;
  minority_ratio?: number;
  low_pitch_hz?: number | null;
  high_pitch_hz?: number | null;
  mixed?: boolean;
  severe_mixed?: boolean;
};

export type SpeakerVoiceLineGroups = {
  profiles?: Record<string, SpeakerVoiceProfile>;
  groups?: Record<string, string[]>;
  line_labels?: Record<string, string>;
};

/** 用户登记的"声纹样本"。embedding 为 256 维归一化向量 */
export type SpeakerProfile = {
  name: string;
  embedding: number[];
  embeddings?: number[][];
  anchor_count?: number;
  sample_seconds?: number;
  quality?: Record<string, unknown>;
  enrollment_source?: string;
  enrollment_ready?: boolean;
  enrollment_reasons?: string[];
  /** 创建时间(ISO),便于 UI 展示 */
  created_at?: string;
};

export type DiarizeResponse = {
  segments: Segment[];
  speakers: string[];
  matched_profiles: Record<string, string>;
  stats: DiarizationStats;
};

export type DiarizationSpeakerSummary = {
  speaker: string;
  segments: number;
  segment_ratio: number;
  duration_s: number;
  duration_ratio: number;
  turns: number;
  stable_turns: number;
  short_segments?: number;
  short_ratio?: number;
  filler_segments?: number;
  filler_ratio?: number;
  sandwiched_segments?: number;
  sandwiched_ratio?: number;
};

export type DiarizationReviewSegment = {
  index: number;
  start: number;
  end: number;
  duration_s: number;
  text: string;
  from_speaker: string;
  to_speaker: string;
  reason: string;
};

export type DiarizationCandidate = {
  n_speakers: number;
  speakers: string[];
  segments: Array<{ start: number; end: number; text: string; speaker: string }>;
  matched_profiles: Record<string, string>;
  stats: DiarizationStats;
  summary: {
    speakers: DiarizationSpeakerSummary[];
    turns: number;
    stable_turns: number;
  };
  score: number;
  actual_n_speakers?: number;
  stable_speakers: number;
  weak_speakers: number;
  tiny_speakers: number;
  fragmented_speakers?: number;
  marginal_speakers?: number;
  dominant_ratio: number;
  issues: string[];
  reason: string;
  fragile_speakers?: string[];
  mergeable_speakers?: string[];
  merge_map?: Record<string, string>;
  merge_distribution?: Record<string, Record<string, number>>;
  merge_reason?: string;
  reassignment_distribution?: Record<string, Record<string, number>>;
  reassignment_reason?: string;
  local_leakage_distribution?: Record<string, Record<string, number>>;
  local_leakage_reason?: string;
  short_sandwich_distribution?: Record<string, Record<string, number>>;
  short_sandwich_reason?: string;
  voice_profiles?: Record<string, SpeakerVoiceProfile>;
  voice_mix_summary?: Record<string, SpeakerVoiceMixSummary>;
  mixed_voice_speakers?: string[];
  severe_mixed_voice_speakers?: string[];
  voice_mix_penalty?: number;
  voice_line_groups?: SpeakerVoiceLineGroups;
  voice_guard_count?: number;
  voice_guard_reason?: string;
  handoff_split_count?: number;
  handoff_split_reason?: string;
  postprocess_skipped_reason?: string;
  review_segments?: DiarizationReviewSegment[];
};

export type RecommendDiarizationResponse = {
  recommended_n_speakers: number;
  recommended_candidate_n_speakers?: number;
  reason: string;
  confidence?: "high" | "medium" | "low";
  confidence_reason?: string;
  score_gap_to_next?: number;
  merge_map?: Record<string, string>;
  merge_distribution?: Record<string, Record<string, number>>;
  merge_reason?: string;
  reassignment_distribution?: Record<string, Record<string, number>>;
  reassignment_reason?: string;
  short_sandwich_distribution?: Record<string, Record<string, number>>;
  short_sandwich_reason?: string;
  voice_profiles?: Record<string, SpeakerVoiceProfile>;
  voice_mix_summary?: Record<string, SpeakerVoiceMixSummary>;
  mixed_voice_speakers?: string[];
  severe_mixed_voice_speakers?: string[];
  voice_mix_penalty?: number;
  voice_line_groups?: SpeakerVoiceLineGroups;
  voice_guard_count?: number;
  voice_guard_reason?: string;
  review_segments?: DiarizationReviewSegment[];
  errors?: Array<{ n_speakers: number; error: string }>;
  candidates: DiarizationCandidate[];
};

export type DiarizationStats = {
  embeddings: number;
  duration_s: number;
  clusters: number;
  matched_profile_count: number;
  segment_count: number;
  status?: "ok" | "partial" | "error";
  applied?: boolean;
  errors?: Array<{ n_speakers: number; error: string }>;
  segmentation_preserved?: boolean;
  failure_reason?: string;
  auto?: boolean;
  requested_n_speakers?: number;
  recommended_n_speakers?: number;
  recommended_run_n_speakers?: number;
  recommended_score?: number | null;
  selected_score?: number | null;
  silhouette_sweep?: Record<string, number>;
  over_split_risk?: boolean;
  over_split_score_gap?: number;
  risk_level?: "low" | "medium" | "high";
  risk_reason?: string;
  recommendation_confidence?: "high" | "medium" | "low" | null;
  recommendation_confidence_reason?: string | null;
  score_gap_to_next?: number | null;
  merge_map?: Record<string, string>;
  merge_distribution?: Record<string, Record<string, number>>;
  merge_reason?: string;
  reassignment_distribution?: Record<string, Record<string, number>>;
  reassignment_reason?: string;
  local_leakage_distribution?: Record<string, Record<string, number>>;
  local_leakage_reason?: string;
  short_sandwich_distribution?: Record<string, Record<string, number>>;
  short_sandwich_reason?: string;
  voice_profiles?: Record<string, SpeakerVoiceProfile>;
  voice_mix_summary?: Record<string, SpeakerVoiceMixSummary>;
  mixed_voice_speakers?: string[];
  severe_mixed_voice_speakers?: string[];
  voice_mix_penalty?: number;
  voice_line_groups?: SpeakerVoiceLineGroups;
  voice_guard_count?: number;
  voice_guard_reason?: string;
  postprocess_skipped_reason?: string;
  review_segments?: DiarizationReviewSegment[];
  engine?: string;
  runtime_backend?: string;
  fallback_reason?: string;
  vad_fallback_reason?: string;
  embedding_fallback_reason?: string;
  overlap_detection?: {
    available: boolean;
    interval_count: number;
    filtered_subsegments: number;
    cluster_filter_enabled?: boolean;
    overlap_seconds?: number;
    overlap_ratio?: number;
    backend?: string;
  };
  voiceprint_reidentify?: VoiceprintReidentifyStats;
};

export type VoiceprintAnchor = {
  speaker: string;
  start: number;
  end: number;
  index?: number;
  text?: string;
};

export type VoiceprintAnchorPreflightCandidate = {
  index: number;
  speaker: string;
  start: number;
  end: number;
  duration: number;
  text: string;
  covered_seconds: number;
  quality: {
    vector_count: number;
    pair_count: number;
    median_similarity: number | null;
    p10_similarity: number | null;
    min_similarity: number | null;
    stable: boolean;
  };
  reason: string;
};

export type VoiceprintAnchorPreflightResponse = {
  candidates: VoiceprintAnchorPreflightCandidate[];
  stats: {
    mode: string;
    checked_segments: number;
    eligible_segments: number;
    rejected_segments: number;
    min_anchor_seconds: number;
    min_quality_vectors: number;
    min_anchor_consistency: number;
    reason: string;
  };
};

export type VoiceprintProfile = SpeakerProfile & {
  dims?: number;
  anchor_count?: number;
  sample_seconds?: number;
};

export type VoiceprintReidentifyStats = {
  mode: string;
  engine: string;
  embedding_dim: number;
  anchor_count: number;
  profile_count: number;
  rejected_anchor_count: number;
  rejected_profile_count?: number;
  rejected_anchors?: Array<Record<string, unknown>>;
  rejected_profiles?: Array<Record<string, unknown>>;
  threshold: number;
  review_threshold: number;
  margin: number;
  min_anchor_seconds?: number;
  min_profile_seconds?: number;
  min_quality_vectors?: number;
  min_anchor_consistency?: number;
  min_profile_consistency?: number;
  require_enrollment_quality?: boolean;
  segment_count: number;
  matched_segments: number;
  changed_segments: number;
  review_segments: number;
  skipped_no_voice_segments: number;
  reason?: string;
};

export type VoiceprintReidentifyResponse = {
  segments: Segment[];
  speakers: string[];
  profiles: VoiceprintProfile[];
  stats: VoiceprintReidentifyStats;
};

export type ASRInterval = Readonly<{
  start: number;
  end: number;
}>;

export type ASRDurationInterval = Readonly<ASRInterval & {
  duration: number;
}>;

export type ASRRecoveryFailure = Readonly<ASRInterval & {
  reason: string;
  speech_duration_s?: number;
  normalized_chars?: number;
  chars_per_s?: number;
}>;

export type ASRLocalRecoveryAttempt = Readonly<{
  framing: string;
  pad_s: number;
  raw: string;
  normalized: string;
  residual: string;
  residual_text: string;
  status: "error" | "rejected" | "matched_existing" | "valid" | string;
  min_required_chars: number;
  provider_id: string;
  provider_kind: string;
  model_id: string;
  model_family: string;
  hallucination_risk: boolean;
  body?: string;
  left_overlap_chars?: number;
  right_overlap_chars?: number;
  rejection_reason?: string;
  error?: string;
  slice_start?: number;
  slice_end?: number;
  slice_sha256?: string;
  model_revision?: string | null;
  config_sha256?: string | null;
  weights_manifest_sha256?: string | null;
  evidence_sha256?: string;
}>;

export type ASRLocalRecoveryDecision =
  | "matched_existing"
  | "insert_accepted"
  | "rejected"
  | "error";

export type ASRLocalRecoverySnapshot = Readonly<{
  segment_count: number;
  text_sha256: string;
  covered_count: number;
  failed_count: number;
  attempted_ranges: ReadonlyArray<readonly [number, number]>;
  recognized_ranges: ReadonlyArray<readonly [number, number]>;
  failed_ranges: ReadonlyArray<readonly [number, number]>;
  attempted_partition_sha256: string;
  recognized_partition_sha256: string;
  failed_partition_sha256: string;
  partition_valid: boolean;
}>;

export type ASRLocalRecoveryDetail = Readonly<{
  start: number;
  end: number;
  window_count: number;
  original_failures: ReadonlyArray<ASRRecoveryFailure>;
  left_context: string;
  right_context: string;
  overlapping_context: string;
  local_reference: string;
  min_required_chars: number;
  attempts: ReadonlyArray<ASRLocalRecoveryAttempt>;
  evidence_decision: ASRLocalRecoveryDecision;
  decision: ASRLocalRecoveryDecision;
  normalization_rejection_reason: string | null;
  consensus: string;
  evidence_framings: readonly string[];
  evidence_providers: readonly string[];
  evidence_models: readonly string[];
  evidence_ids: readonly string[];
  primary_status: string;
  primary_consensus: string;
  primary_evidence_framings: readonly string[];
  inserted_raw_text: string;
  inserted_text: string;
  insertion_normalization: Readonly<Record<string, unknown>> | null;
}>;

export type ASRLocalRecovery = Readonly<{
  mode: "off" | "audit" | "merge";
  requested_mode: string;
  diagnostic: string | null;
  pending_windows: number;
  pending_groups: number;
  attempts: number;
  matched_existing: number;
  inserted: number;
  rejected: number;
  error: number;
  before: ASRLocalRecoverySnapshot;
  after: ASRLocalRecoverySnapshot;
  details: ReadonlyArray<ASRLocalRecoveryDetail>;
  details_truncated: boolean;
  provider: Readonly<{
    requested: string;
    available: boolean;
    error: string | null;
    provider_id?: string;
    provider_kind?: string;
    model_id?: string;
    model_family?: string;
    model_revision?: string;
    config_sha256?: string;
    weights_manifest_sha256?: string;
    weight_files?: number;
  }>;
  text_normalization: Readonly<{
    language: string;
    profile: string | null;
    error: string | null;
  }>;
}>;

export type SpeechCoverageReport = Readonly<{
  status: string;
  reason: string;
  speech_ranges: number;
  speech_duration_s: number;
  covered_speech_s: number;
  uncovered_speech_s: number;
  speech_coverage_ratio: number | null;
  max_uncovered_speech_s: number;
  leading_uncovered_speech_s: number;
  trailing_uncovered_speech_s: number;
  speech_intervals: ReadonlyArray<ASRInterval>;
  covered_intervals: ReadonlyArray<ASRInterval>;
  uncovered_speech_ranges: ReadonlyArray<ASRDurationInterval>;
  uncovered_speech_ranges_truncated: boolean;
  settings: Readonly<{
    segment_collar_s: number;
    min_speech_coverage_ratio: number;
    max_uncovered_speech_s: number;
    max_edge_uncovered_speech_s: number;
  }>;
  basis?: string;
  wallclock_attempted_chunks?: number;
  wallclock_recognized_chunks?: number;
  wallclock_failed_chunks?: number;
  wallclock_failed_ranges?: ReadonlyArray<ASRInterval>;
  wallclock_failure_reasons?: ReadonlyArray<ASRRecoveryFailure>;
  wallclock_failure_details_truncated?: boolean;
  wallclock_min_chars_per_s?: number;
  wallclock_strict_coverage?: boolean;
  wallclock_max_chunk_s?: number | null;
  local_recovery?: ASRLocalRecovery;
}>;

export type ASRSpeechCoverage = SpeechCoverageReport;

export type ASRHumanReviewStatus = "pending" | "approved" | "needs_changes";

export type ASRHumanReviewItemStatus =
  | "pending"
  | "confirmed_present"
  | "confirmed_missing"
  | "substitution"
  | "noise"
  | "resolved";

export type ASRHumanReviewItem = Readonly<{
  id: string;
  start: number;
  end: number;
  status: ASRHumanReviewItemStatus;
  review_status?: ASRHumanReviewItemStatus;
  source_decision?: ASRLocalRecoveryDecision;
  source_evidence_ids?: readonly string[];
  corrected_text?: string;
  heard_text?: string;
  replacement_text?: string;
  note?: string;
  reviewed_at?: string;
}>;

export type ASRHumanReview = Readonly<{
  schema_version: 1;
  status?: unknown;
  items: ReadonlyArray<ASRHumanReviewItem>;
  source?: unknown;
  coverage_partition_sha256?: string;
  reviewer?: string;
  created_at?: string;
  updated_at?: unknown;
}>;

export type ASRStrongReviewCandidate = Readonly<{
  window?: number;
  start?: number;
  end?: number;
  reason?: string;
  primary_para_similarity?: number;
  primary?: string;
  paraformer?: string;
  qwen?: string;
  qwen_hallucination_risk?: boolean;
}>;

export type ASRStrongReviewStats = Readonly<{
  mode?: string;
  enabled?: boolean;
  applied?: boolean;
  reason?: string;
  trigger?: string;
  review_recommended?: boolean;
  candidate_window_count?: number;
  reviewed_windows?: number;
  changed_windows?: number;
  replacement_count?: number;
  audit_only_window_count?: number;
  audit_only_candidates?: ReadonlyArray<ASRStrongReviewCandidate>;
  timeline_preserved?: boolean;
  cost_seconds?: number;
  error?: string;
  auto_review_decision?: Readonly<{
    eligible?: boolean;
    recommended?: boolean;
    auto_run_enabled?: boolean;
    auto_run_reason?: string;
    reason?: string;
    audio_risk_level?: string;
    estimated_snr_db?: number | null;
    timing_reliable?: boolean | null;
    alignment_similarity?: number | null;
    alignment_reason?: string;
    noise_evidence?: readonly string[];
    decode_disagreement_evidence?: readonly string[];
  }>;
}>;

export type FilterStats = {
  input?: number;
  output?: number;
  removed_total?: number;
  vad?: number;
  logprob?: number;
  phrases?: number;
  repetition?: number;
  density?: number;
  similarity?: number;
  audio_standardization?: Record<string, unknown>;
  audio_quality?: Record<string, unknown>;
  text_normalization?: Record<string, unknown>;
  speech_coverage?: SpeechCoverageReport;
  timing_mode?: string;
  timing_reliable?: boolean;
  timing_reason?: string;
  timing_alignment_reason?: string;
  equal_char_ratio?: number;
  min_equal_ratio?: number;
  asr_quality_mode?: "standard" | "strong" | string;
  strong_asr?: ASRStrongReviewStats;
};

export type ASRQualityReport = {
  mode: string;
  backend: string;
  model_id: string;
  duration_s: number;
  transcribe_seconds: number;
  rtf: number;
  segments: number;
  chars: number;
  punctuation_ratio: number;
  traditional_char_hits: string[];
  hotwords: {
    count: number;
    exact_hit_count: number;
    coverage: number | null;
    exact_hits: string[];
    missing_terms: string[];
    near_misses: Array<{
      term: string;
      candidates: Array<{ text: string; similarity: number }>;
    }>;
  };
  term_consistency?: {
    mode: string;
    candidate_count: number;
    candidates: TermConsistencyCandidate[];
  };
  audio_quality?: Record<string, unknown>;
  audio_preprocessing?: Record<string, unknown>;
  industry_pipeline?: {
    source: string;
    principle: string;
    steps: Array<{ step: string; status: string; detail: string }>;
  };
  review: {
    segment_count: number;
    segments: Array<{
      index: number;
      start: number;
      end: number;
      text: string;
      original_text?: string;
      reasons: string[];
    }>;
  };
  risk_level: "low" | "medium" | "high";
  risk_reasons: string[];
  spot_check_reasons?: string[];
  recommendation: string;
};

export type TermConsistencyCandidate = {
  id: string;
  kind?: "phonetic_entity" | "entity_drift" | "orthographic_term" | string;
  action?: "review" | "maybe_unify" | string;
  confidence: number;
  terms: string[];
  suggested_canonical?: string | null;
  phonetic_key?: string;
  total_count: number;
  reason?: string;
  variants?: Array<{
    text: string;
    count: number;
    contexts?: Array<{
      index: number;
      start: number;
      end: number;
      text: string;
    }>;
  }>;
  contexts?: Array<{
    index: number;
    start: number;
    end: number;
    text: string;
  }>;
};

export type TranscribeResult = {
  audio: string;
  language: string | null;
  duration: number;
  transcribe_seconds: number;
  rtf: number;
  backend: string;
  model_id: string;
  segments: Segment[];
  filter_stats?: FilterStats;
  asr_quality?: ASRQualityReport;
  diarization_stats?: DiarizationStats;
};

export type EnvironmentInfo = {
  apple_silicon: boolean;
  default_backend: string;
  ffmpeg: string | null;
  ffprobe: string | null;
  default_model_id: string;
  diarization_engine?: string;
  diarization_engines_available?: string[];
  diarization_fallback_reason?: string | null;
};

export type ModelStatus = {
  backend?: string;
  model_id: string;
  exists: boolean;
  path: string | null;
  source?: "env" | "project" | "hf_cache" | "modelscope_cache" | "bundle" | null;
  /** 推荐用户放置文件的目标路径(LocalScribe/models/<basename>/) */
  expected_local_path?: string | null;
};

export type ProbeAudioInfo = {
  audio: string;
  duration: number;
  size: number;
  format_name: string;
  has_audio_stream: boolean;
  ffmpeg: string | null;
  ffprobe: string | null;
};

export type ASRPreprocessMode = "off" | "standard" | "adaptive" | "ai_denoise" | "enhance";
export type ASRPreprocessSetting = "auto_preflight" | ASRPreprocessMode;

export type ASRPreflightModeSummary = {
  mode: ASRPreprocessMode | string;
  status: string;
  risk_level?: "low" | "medium" | "high" | "unknown" | string;
  strong_review_count?: number;
  review_count?: number;
  term_candidate_count?: number;
  chars?: number;
  avg_punctuation_ratio?: number;
  avg_rtf?: number;
  preprocess_fallback?: boolean;
  preprocess_fallback_count?: number;
  preprocess_filters?: string[];
  ok_count?: number;
  error_count?: number;
  score_key?: Array<number | string>;
  reason?: string;
};

export type ASRPreflightSampleRow = {
  clip: string;
  clip_index?: number;
  clip_start?: number;
  clip_duration?: number;
  mode: ASRPreprocessMode | string;
  status: string;
  risk_level: "low" | "medium" | "high" | "unknown" | string;
  review_count: number;
  strong_review_count: number;
  term_candidate_count: number;
  chars: number;
  segments: number;
  punctuation_ratio: number;
  rtf: number;
  cost_seconds: number;
  preprocess_mode?: string;
  preprocess_fallback?: boolean;
  preprocess_filters?: string[];
  error?: string;
};

export type ASRPreflightSelectResponse = {
  mode: "asr_preflight_select";
  audio: string;
  backend: string;
  model: string;
  duration_s: number;
  probe: ProbeAudioInfo;
  audio_quality: Record<string, unknown>;
  skipped: boolean;
  skip_reason?: string;
  windows: Array<{ index: number; start: number; end: number; duration?: number; path?: string }>;
  sample_rows: ASRPreflightSampleRow[];
  mode_summary: ASRPreflightModeSummary[];
  recommended_mode: ASRPreprocessMode | string;
  recommendation_reason?: string;
};

export type CorrectionMode = "light" | "medium" | "heavy";

export type GlossaryEntry = {
  term: string;
  may_appear_as?: string[];
  category?: string;
  freq?: number;
};

export type CorrectResponse = {
  segments: Segment[];
  changed: number;
  total: number;
  model: string;
  mode: CorrectionMode;
  glossary: GlossaryEntry[];
  cancelled?: boolean;
  concurrency?: number;
};

export type PolishResponse = {
  text: string;
  model: string;
  char_count: number;
  finish_reason?: string;
  truncated?: boolean;
  input_chars?: number;
};

export type TranslateResponse = {
  text: string;
  source_language: string | null;
  target_language: string;
  model: string;
  char_count: number;
  finish_reason?: string;
  truncated?: boolean;
  input_chars?: number;
};

export type LLMAdvanced = {
  temperature: number;
  max_tokens: number;
  top_p: number;
  frequency_penalty: number;
  presence_penalty: number;
};

export type CorrectionSettings = {
  enabled: boolean;
  auto_pipeline: boolean;
  provider: string;
  base_url: string;
  model: string;
  mode: CorrectionMode;
  batch_size: number;
  context_hint: string;
  use_glossary: boolean;
  concurrency: number;
  advanced: LLMAdvanced;
};

export type PolishSettings = {
  enabled: boolean;
  model: string;
  advanced: LLMAdvanced;
};

export type TranslationSettings = {
  model: string;
  advanced: LLMAdvanced;
};

export type DiarizationSettings = {
  enabled: boolean;
  /** 分离引擎: auto 优先本地中文会议引擎,不可用时明确 fallback */
  engine: "auto" | "senko" | "resemblyzer" | "pyannote" | string;
  /** 期望的说话人数(KMeans 簇数)。1 = 单人(跳过聚类) */
  n_speakers: number;
  /** 注册的声纹样本库 — 自动给真实姓名 */
  speakers: SpeakerProfile[];
};

export type AppSettings = {
  model_id: string;
  backend: string;
  language: string;
  asr_hotwords: string;
  asr_quality_mode: "standard" | "strong";
  audio_preprocess: ASRPreprocessSetting;
  transcript_sync: "precise" | "fast";
  output_formats: string[];
  output_dir: string | null;
  correction: CorrectionSettings;
  polish: PolishSettings;
  translation: TranslationSettings;
  diarization: DiarizationSettings;
};

// ---- backend bridge ----

export function sanitizeLibraryStem(value: string): string {
  const sanitized = Array.from(value.trim(), (char) => {
    const code = char.codePointAt(0) ?? 0;
    return char === "/" || char === "\\" || code < 32 || (code >= 127 && code <= 159) ? "_" : char;
  }).join("").replace(/^[.\s]+|[.\s]+$/g, "");
  return sanitized || "meeting";
}

export function libraryStemFromFilename(filename: string): string {
  return sanitizeLibraryStem(filename.replace(/\.[^.]+$/, ""));
}

export function libraryStemKey(value: string): string {
  return sanitizeLibraryStem(value).normalize("NFC").toLocaleLowerCase();
}

export const ipc = {
  environment: () => invokeCommand<EnvironmentInfo>("environment"),
  checkModel: (params?: { backend?: string; model_id?: string }) =>
    invokeCommand<ModelStatus>("check_model", params ?? {}),
  probeAudio: (audio: string) => invokeCommand<ProbeAudioInfo>("probe_audio", { audio }),
  asrPreflightSelect: (params: {
    audio: string;
    backend?: string;
    model_id?: string;
    language?: string;
    hotwords?: string;
    preferred_mode?: ASRPreprocessMode;
    modes?: ASRPreprocessMode[];
    clip_seconds?: number;
    max_clips?: number;
    force?: boolean;
  }) => invokeCommand<ASRPreflightSelectResponse>("asr_preflight_select", params),
  diarize: (params: {
    audio: string;
    segments: Segment[];
    n_speakers?: number;
    engine?: string;
    profiles?: SpeakerProfile[];
  }) => invokeCommand<DiarizeResponse>("diarize", {
    audio: params.audio,
    segments: params.segments,
    nSpeakers: params.n_speakers,
    engine: params.engine,
    profiles: params.profiles,
    preserveSegmentation: true,
  }),
  recommendDiarization: (params: {
    audio: string;
    segments: Segment[];
    min_speakers?: number;
    max_speakers?: number;
    engine?: string;
    profiles?: SpeakerProfile[];
  }) => invokeCommand<RecommendDiarizationResponse>("recommend_diarization", {
    audio: params.audio,
    segments: params.segments,
    minSpeakers: params.min_speakers,
    maxSpeakers: params.max_speakers,
    engine: params.engine,
    profiles: params.profiles,
    preserveSegmentation: true,
  }),
  extractVoiceEmbedding: (audio: string) =>
    invokeCommand<{ embedding: number[]; dims: number; audio: string }>(
      "extract_voice_embedding",
      { audio },
    ),
  preflightVoiceprintAnchors: (params: {
    audio: string;
    segments: Segment[];
    engine?: string;
  }) => invokeCommand<VoiceprintAnchorPreflightResponse>("preflight_voiceprint_anchors", {
    audio: params.audio,
    segments: params.segments,
    engine: params.engine,
  }),
  reidentifySpeakers: (params: {
    audio: string;
    segments: Segment[];
    anchors: VoiceprintAnchor[];
    engine?: string;
    threshold?: number;
    review_threshold?: number;
    margin?: number;
    require_enrollment_quality?: boolean;
  }) => invokeCommand<VoiceprintReidentifyResponse>("reidentify_speakers", {
    audio: params.audio,
    segments: params.segments,
    anchors: params.anchors,
    engine: params.engine,
    threshold: params.threshold,
    reviewThreshold: params.review_threshold,
    margin: params.margin,
    requireEnrollmentQuality: params.require_enrollment_quality,
  }),
  transcribe: (params: {
    audio: string;
    backend?: string;
    model_id?: string;
    language?: string;
    initial_prompt?: string;
    hotwords?: string;
    asr_quality_mode?: "standard" | "strong";
    normalizer_profile?: string;
    audio_preprocess?: ASRPreprocessMode;
    timing_align?: boolean;
  }) => invokeCommand<TranscribeResult>("transcribe", params),
  correctSegments: (params: {
    segments: Segment[];
    provider?: string;
    base_url?: string;
    model?: string;
    mode?: CorrectionMode;
    batch_size?: number;
    context_hint?: string;
    use_glossary?: boolean;
    concurrency?: number;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
    language?: string;
  }) => invokeCommand<CorrectResponse>("correct_segments", params),
  polishArticle: (params: {
    segments: Segment[];
    provider?: string;
    base_url?: string;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
  }) => invokeCommand<PolishResponse>("polish_article", params),

  translateArticle: (params: {
    text: string;
    source_language?: string;
    target_language: string;
    glossary?: GlossaryEntry[];
    provider?: string;
    base_url?: string;
    model?: string;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    frequency_penalty?: number;
    presence_penalty?: number;
  }) => invokeCommand<TranslateResponse>("translate_article", params),

  // ASR / correction control
  asrCancel: () => invokeCommand<{ status: string }>("asr_cancel"),
  correctPause: () => invokeCommand<{ status: string }>("correct_pause"),
  correctResume: () => invokeCommand<{ status: string }>("correct_resume"),
  correctCancel: () => invokeCommand<{ status: string }>("correct_cancel"),
  correctStatus: () => invokeCommand<{ paused: boolean; cancelled: boolean }>("correct_status"),

  // secrets
  setApiKey: (provider: string, apiKey: string) =>
    invokeCommand<void>("set_api_key", { provider, apiKey }),
  hasApiKey: (provider: string) => invokeCommand<boolean>("has_api_key", { provider }),
  getApiKey: (provider: string) => invokeCommand<string>("get_api_key", { provider }),
  deleteApiKey: (provider: string) => invokeCommand<void>("delete_api_key", { provider }),

  // settings
  loadSettings: () => invokeCommand<AppSettings>("load_settings"),
  saveSettings: (settings: AppSettings) => invokeCommand<void>("save_settings", { settings }),

  // model cache
  checkModelCache: (model_id: string) => invokeCommand<ModelStatus>("check_model_cache", { model_id }),
  revealModelsDir: (model_id?: string) =>
    invokeCommand<string>("reveal_models_dir", model_id ? { model_id } : {}),
  openUrl: (url: string) => invokeCommand<void>("open_url", { url }),

  // library
  librarySaveRaw: (args: {
    stem: string;
    audio_filename: string;
    source_audio?: string;
    txt: string;
    srt: string;
    json: string;
    result: TranscribeResult;
  }) => invokeCommand<LibraryMeta>("library_save_raw", {
    args: { ...args, stem: sanitizeLibraryStem(args.stem) },
  }),
  librarySaveAsrReview: (args: { stem: string; review: ASRHumanReview }) =>
    invokeCommand<void>("library_save_asr_review", {
      args: { ...args, stem: sanitizeLibraryStem(args.stem) },
    }),
  librarySaveCorrected: (args: {
    stem: string;
    txt: string;
    srt: string;
    json: string;
    diff: string;
    model: string;
    changed: number;
    total: number;
    glossary?: GlossaryEntry[];
  }) => invokeCommand<LibraryMeta>("library_save_corrected", {
    args: { ...args, stem: sanitizeLibraryStem(args.stem) },
  }),
  librarySaveRawAndCorrected: (args: {
    raw: {
      stem: string;
      audio_filename: string;
      source_audio?: string;
      txt: string;
      srt: string;
      json: string;
      result: TranscribeResult;
    };
    corrected?: {
      stem: string;
      txt: string;
      srt: string;
      json: string;
      diff: string;
      model: string;
      changed: number;
      total: number;
      glossary?: GlossaryEntry[];
    };
    clear_corrected?: boolean;
  }) => invokeCommand<LibraryMeta>("library_save_raw_and_corrected", {
    args: {
      raw: { ...args.raw, stem: sanitizeLibraryStem(args.raw.stem) },
      corrected: args.corrected
        ? { ...args.corrected, stem: sanitizeLibraryStem(args.corrected.stem) }
        : undefined,
      clear_corrected: args.clear_corrected ?? false,
    },
  }),
  librarySavePolished: (args: {
    stem: string;
    text: string;
    model: string;
    source?: "corrected" | "raw";
  }) => invokeCommand<LibraryMeta>("library_save_polished", {
    args: { ...args, stem: sanitizeLibraryStem(args.stem) },
  }),
  libraryList: () => invokeCommand<LibraryMeta[]>("library_list"),
  libraryLoad: (stem: string) => invokeCommand<LoadedTask>("library_load", { stem: sanitizeLibraryStem(stem) }),
  libraryDelete: (stem: string) => invokeCommand<void>("library_delete", { stem: sanitizeLibraryStem(stem) }),
  libraryArchive: (stem: string) => invokeCommand<string | null>("library_archive", { stem: sanitizeLibraryStem(stem) }),
  libraryRootPath: () => invokeCommand<string>("library_root_path"),

  // articles 知识库
  articleSave: (args: SaveArticleArgs) => invokeCommand<ArticleMeta>("article_save", { args }),
  articleList: () => invokeCommand<ArticleMeta[]>("article_list"),
  articleDelete: (filename: string) => invokeCommand<void>("article_delete", { filename }),
  articleRename: (oldFilename: string, newTitle: string) =>
    invokeCommand<ArticleMeta>("article_rename", { oldFilename, newTitle }),
  articleRead: (filename: string) => invokeCommand<string>("article_read", { filename }),
  articlesRootPath: () => invokeCommand<string>("articles_root_path"),
};

export type SaveArticleArgs = {
  title: string;
  content: string;
  source_audio?: string;
  source_stem?: string;
  duration_seconds?: number;
  model?: string;
  based_on?: "corrected" | "raw";
  tags?: string[];
  note?: string;
  overwrite?: boolean;
};

export type ArticleMeta = {
  title: string;
  filename: string;
  path: string;
  source_audio: string | null;
  source_stem: string | null;
  duration_seconds: number | null;
  char_count: number;
  model: string | null;
  based_on: string | null;
  tags: string[];
  note: string | null;
  created_at: string;
  modified_at: string;
};

export type LibraryMeta = {
  stem: string;
  audio_filename: string;
  audio_path: string | null;
  duration: number;
  segments: number;
  backend: string;
  model_id: string;
  created_at: number;
  updated_at: number;
  has_corrected: boolean;
  has_polished: boolean;
  correction_model: string | null;
  correction_changed: number | null;
  correction_glossary: GlossaryEntry[] | null;
  polish_model: string | null;
  polish_source: string | null;
};

export type LoadedTask = {
  meta: LibraryMeta;
  raw_json: TranscribeResult;
  asr_human_review?: ASRHumanReview | null;
  corrected_json: {
    segments: Segment[];
    corrected_by?: string;
    changed?: number;
    total?: number;
    glossary?: GlossaryEntry[];
  } | null;
  polished_text: string | null;
};

// ---- progress events ----

export type ProgressData = {
  current?: number;
  total?: number;
  preview?: string;
  stage?: string;
  error?: string;
};

export type ProgressMethod = "transcribe" | "correct" | "asr_preflight_select";

export function onProgress(
  method: ProgressMethod,
  handler: (data: ProgressData) => void,
): Promise<UnlistenFn> {
  return listenEvent<ProgressData>(`scribe://progress/${method}`, handler);
}
