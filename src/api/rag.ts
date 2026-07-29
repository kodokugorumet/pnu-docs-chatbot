export const GENERATION_PROVIDERS = ['auto', 'local', 'frontier'] as const
export const PARSER_PROFILES = ['baseline', 'challenger', 'cascade'] as const

export type GenerationProvider = (typeof GENERATION_PROVIDERS)[number]
export type ConcreteProvider = Exclude<GenerationProvider, 'auto'>
export type ParserProfile = (typeof PARSER_PROFILES)[number]

export type CitationLocation = {
  block_id?: string | null
  page?: number | null
  page_end?: number | null
  section_path?: string[] | null
  table_id?: string | null
  row?: number | null
  column?: number | null
}

export type ClaimCitation = CitationLocation & {
  source_number?: number
  chunk_id?: string
}

export type Claim = {
  text: string
  supported: boolean
  confidence?: number
  source_ids?: string[]
  source_numbers?: number[]
  citations?: ClaimCitation[]
  validation_reason?:
    | 'supported'
    | 'model_abstention'
    | 'critical_value_mismatch'
    | 'low_lexical_overlap'
    | string
  missing_critical_values?: string[]
  best_score?: number
}

export type SearchResult = {
  source_number?: number
  chunk_id: string
  doc_id?: string
  document_id?: string
  chunk_index?: number
  institution?: string
  file_name?: string
  source_path?: string
  relative_path?: string
  source_title?: string | null
  source_url?: string | null
  download_url?: string | null
  source_host?: string | null
  fetched_at?: string | null
  published_at?: string | null
  category?: string | null
  char_count?: number
  score?: number
  preview?: string
  location?: CitationLocation | null
  locations?: CitationLocation[]
  location_count?: number
  locations_truncated?: boolean
  page?: number | null
  page_start?: number | null
  page_end?: number | null
  section_path?: string[] | string | null
  table_id?: string | null
  table_ids?: string[]
  row?: number | null
  column?: number | null
  block_ids?: string[]
  metadata?: Record<string, unknown>
  scores?: Record<string, number | null | undefined>
}

export type GenerationAttempt = {
  provider?: string
  model?: string | null
  status?: string
  error?: string | null
  duration_ms?: number | null
}

export type GenerationInfo = {
  requested: GenerationProvider
  used: string
  model?: string | null
  fallback_reason?: string | null
  attempts?: GenerationAttempt[]
}

export type PipelineTraceStage = {
  id?: string
  label?: string
  state?: string
  status?: string
  duration_ms?: number | null
  detail?: string | null
}

export type ChatTrace = {
  request_id?: string
  stages?: PipelineTraceStage[]
}

export type ChatRequest = {
  question: string
  institution?: string
  top_k?: number
  provider: GenerationProvider
  model?: string
  parser_profile: ParserProfile
}

export type ChatResponse = {
  answer: string
  cited_answer?: string
  claims?: Claim[]
  results: SearchResult[]
  generation?: Partial<GenerationInfo>
  generator?: string
  trace?: ChatTrace
  retrieval?: Record<string, unknown>
  request_id?: string
  parser_profile?: ParserProfile
}

export type PipelineStage = {
  id: string
  label: string
  state: string
  detail?: string
}

export type ProviderCapability = {
  id: ConcreteProvider
  label: string
  available: boolean
  state: string
  model?: string
  defaultModel?: string
  models?: ProviderModelCapability[]
  reason?: string
}

export type ProviderModelCapability = {
  id: string
  label: string
  available: boolean
  reason?: string
}

export type ParserProfileCapability = {
  id: ParserProfile
  label: string
  ready: boolean
  chunkCount?: number
  documentCount?: number
  runId?: string
  corpusRevision?: string
  denseReady?: boolean
  reason?: string
}

export type HealthResponse = {
  status?: string
  ready: boolean
  pipeline?: unknown
  providers?: unknown
  default_provider?: GenerationProvider
  chunk_count?: number
  institution_count?: number
  generation_mode?: string
  gemini_configured?: boolean
  gemini_model?: string
  default_parser_profile?: ParserProfile
  parser_profiles?: unknown
  [key: string]: unknown
}

type ApiErrorPayload = {
  error?: string
  message?: string
  retryable?: boolean
  request_id?: string
}

const API_BASE_URL = String(import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(/\/+$/, '')
const API_TOKEN = String(import.meta.env.VITE_RAG_API_TOKEN ?? '').trim()
const HEALTH_TIMEOUT_MS = 8000

function envNumber(value: unknown, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export const CHAT_TIMEOUT_MS = envNumber(import.meta.env.VITE_CHAT_TIMEOUT_MS, 45000)

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function optionalString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function optionalBoolean(value: unknown) {
  return typeof value === 'boolean' ? value : undefined
}

function optionalInteger(value: unknown) {
  return typeof value === 'number' && Number.isInteger(value) ? value : undefined
}

function normalizeSectionPath(value: unknown): string[] | null | undefined {
  if (Array.isArray(value)) {
    const path = value
      .map((item) => optionalString(item))
      .filter((item): item is string => Boolean(item))
    return path.length ? path : null
  }
  const path = optionalString(value)
  return path ? [path] : value === null ? null : undefined
}

export function normalizeCitationLocation(value: unknown): CitationLocation | null {
  if (!isRecord(value)) {
    return null
  }

  const location: CitationLocation = {
    block_id: optionalString(value.block_id) ?? null,
    page: optionalInteger(value.page) ?? optionalInteger(value.page_start) ?? null,
    page_end: optionalInteger(value.page_end) ?? null,
    section_path: normalizeSectionPath(value.section_path) ?? null,
    table_id: optionalString(value.table_id) ?? null,
    row: optionalInteger(value.row) ?? null,
    column: optionalInteger(value.column) ?? null,
  }

  const hasDisplayLocation =
    location.page !== null ||
    Boolean(location.section_path?.length) ||
    location.table_id !== null ||
    location.row !== null ||
    location.column !== null

  return hasDisplayLocation ? location : null
}

function locationsFromValue(value: unknown) {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => normalizeCitationLocation(item))
    .filter((item): item is CitationLocation => item !== null)
}

export function getResultLocations(result: SearchResult): CitationLocation[] {
  const metadata = isRecord(result.metadata) ? result.metadata : {}
  const candidates = [
    ...locationsFromValue(result.locations),
    ...locationsFromValue(metadata.locations),
  ]

  const direct =
    normalizeCitationLocation(result.location) ??
    normalizeCitationLocation(metadata.location)
  if (direct) {
    candidates.push(direct)
  }

  if (!candidates.length) {
    const fallback = normalizeCitationLocation({
      block_id: result.block_ids?.[0] ?? metadata.block_id,
      page: result.page ?? result.page_start ?? metadata.page ?? metadata.page_start,
      page_end: result.page_end ?? metadata.page_end,
      section_path: result.section_path ?? metadata.section_path,
      table_id: result.table_id ?? result.table_ids?.[0] ?? metadata.table_id,
      row: result.row ?? metadata.row,
      column: result.column ?? metadata.column,
    })
    if (fallback) {
      candidates.push(fallback)
    }
  }

  const tableIds = result.table_ids ?? (Array.isArray(metadata.table_ids) ? metadata.table_ids : [])
  if (candidates.length === 1 && !candidates[0].table_id && tableIds.length) {
    const base = candidates.shift()
    if (base) {
      for (const tableId of tableIds) {
        const normalized = optionalString(tableId)
        if (normalized) {
          candidates.push({ ...base, table_id: normalized })
        }
      }
    }
  }

  const seen = new Set<string>()
  return candidates.filter((location) => {
    const key = JSON.stringify(location)
    if (seen.has(key)) {
      return false
    }
    seen.add(key)
    return true
  })
}

const providerLabels: Record<ConcreteProvider, string> = {
  local: '로컬 LLM',
  frontier: '프론티어 AI (Gemini)',
}

const parserProfileLabels: Record<ParserProfile, string> = {
  baseline: 'Baseline · 기본 파서',
  challenger: 'Challenger · 대체 파서',
  cascade: 'Cascade · 품질 기반 선택',
}

const parserProfileCompactLabels: Record<ParserProfile, string> = {
  baseline: 'Baseline',
  challenger: 'Challenger',
  cascade: 'Cascade',
}

function parserProfileId(value: unknown): ParserProfile | null {
  const profile = optionalString(value)?.toLowerCase()
  return PARSER_PROFILES.includes(profile as ParserProfile)
    ? (profile as ParserProfile)
    : null
}

export function parserProfileDisplayName(
  profile: ParserProfile,
  compact = false,
) {
  return compact
    ? parserProfileCompactLabels[profile]
    : parserProfileLabels[profile]
}

export function getParserProfileCapabilities(
  health: HealthResponse | null,
): ParserProfileCapability[] {
  const found = new Map<ParserProfile, ParserProfileCapability>()
  const values = health?.parser_profiles
  if (Array.isArray(values)) {
    for (const value of values) {
      if (!isRecord(value)) {
        continue
      }
      const id = parserProfileId(value.id) ?? parserProfileId(value.profile)
      if (!id) {
        continue
      }
      found.set(id, {
        id,
        label: optionalString(value.label) ?? parserProfileLabels[id],
        ready:
          optionalBoolean(value.ready) ??
          optionalString(value.status)?.toLowerCase() === 'ready',
        chunkCount: optionalInteger(value.chunk_count),
        documentCount: optionalInteger(value.document_count),
        runId: optionalString(value.run_id),
        corpusRevision: optionalString(value.corpus_revision),
        denseReady: optionalBoolean(value.dense_ready),
        reason:
          optionalString(value.reason) ??
          optionalString(value.error) ??
          optionalString(value.message),
      })
    }
  }

  if (!found.size && health) {
    const fallbackId =
      parserProfileId(health.default_parser_profile) ??
      parserProfileId(health.profile) ??
      'cascade'
    found.set(fallbackId, {
      id: fallbackId,
      label: parserProfileLabels[fallbackId],
      ready: health.ready,
      chunkCount: optionalInteger(health.chunk_count),
      documentCount: optionalInteger(health.document_count),
      runId: optionalString(health.run_id),
      corpusRevision: optionalString(health.corpus_revision),
      denseReady: optionalBoolean(health.dense_ready),
    })
  }

  return PARSER_PROFILES.map(
    (id) =>
      found.get(id) ?? {
        id,
        label: parserProfileLabels[id],
        ready: false,
        reason: health
          ? `${parserProfileLabels[id]} 인덱스가 서버에 등록되지 않았습니다.`
          : 'API 상태를 확인한 뒤 선택할 수 있습니다.',
      },
  )
}

export function providerDisplayName(provider: string) {
  if (provider === 'auto') return '자동 선택'
  if (provider === 'local') return providerLabels.local
  if (provider === 'frontier' || provider === 'gemini') return providerLabels.frontier
  if (provider === 'extractive') return '검색 근거 요약'
  return provider
}

function providerAvailability(record: Record<string, unknown>) {
  const explicit =
    optionalBoolean(record.available) ??
    optionalBoolean(record.ready) ??
    optionalBoolean(record.enabled) ??
    optionalBoolean(record.configured)
  if (explicit !== undefined) {
    return explicit
  }

  const state = (optionalString(record.state) ?? optionalString(record.status) ?? '').toLowerCase()
  if (['ready', 'available', 'configured', 'healthy', 'degraded'].includes(state)) {
    return true
  }
  return false
}

function providerModels(value: unknown): ProviderModelCapability[] {
  if (!Array.isArray(value)) {
    return []
  }

  const normalized = new Map<string, ProviderModelCapability>()
  for (const item of value) {
    const record = isRecord(item) ? item : {}
    const id =
      optionalString(item) ??
      optionalString(record.id) ??
      optionalString(record.model)
    if (!id) {
      continue
    }
    normalized.set(id, {
      id,
      label: optionalString(record.label) ?? id,
      available: optionalBoolean(record.available) ?? true,
      reason:
        optionalString(record.reason) ??
        optionalString(record.error) ??
        optionalString(record.message),
    })
  }
  return [...normalized.values()]
}

function providerRecord(
  id: ConcreteProvider,
  value: unknown,
): ProviderCapability {
  const record = isRecord(value)
    ? value
    : typeof value === 'string'
      ? { state: value }
      : {}
  const available = typeof value === 'boolean' ? value : providerAvailability(record)
  const model = optionalString(record.model) ?? optionalString(record.model_name)
  const defaultModel =
    optionalString(record.default_model) ??
    optionalString(record.defaultModel) ??
    model
  const models = id === 'local' ? providerModels(record.models) : []
  for (const configuredModel of [defaultModel, model]) {
    if (
      configuredModel &&
      !models.some((candidate) => candidate.id === configuredModel)
    ) {
      models.push({
        id: configuredModel,
        label: configuredModel,
        available: true,
      })
    }
  }
  const state =
    optionalString(record.state) ??
    optionalString(record.status) ??
    (available ? 'ready' : 'unavailable')
  return {
    id,
    label: optionalString(record.label) ?? providerLabels[id],
    available,
    state,
    model,
    defaultModel,
    models: id === 'local' ? models : undefined,
    reason:
      optionalString(record.reason) ??
      optionalString(record.error) ??
      optionalString(record.message),
  }
}

function providerId(value: unknown): ConcreteProvider | null {
  const id = optionalString(value)?.toLowerCase()
  return id === 'local' || id === 'frontier' ? id : null
}

export function getProviderCapabilities(health: HealthResponse | null): ProviderCapability[] {
  const found = new Map<ConcreteProvider, ProviderCapability>()
  const providers = health?.providers

  if (Array.isArray(providers)) {
    for (const value of providers) {
      if (typeof value === 'string') {
        const id = providerId(value)
        if (id) {
          found.set(id, providerRecord(id, true))
        }
        continue
      }
      if (!isRecord(value)) {
        continue
      }
      const id = providerId(value.id) ?? providerId(value.provider)
      if (id) {
        found.set(id, providerRecord(id, value))
      }
    }
  } else if (isRecord(providers)) {
    for (const id of ['local', 'frontier'] as const) {
      if (id in providers) {
        found.set(id, providerRecord(id, providers[id]))
      }
    }
  }

  if (!found.size && health) {
    found.set(
      'frontier',
      providerRecord('frontier', {
        available: health.gemini_configured === true,
        model: health.gemini_model,
        reason:
          health.gemini_configured === true
            ? undefined
            : '프론티어 제공자 상태가 보고되지 않았습니다.',
      }),
    )
  }

  for (const id of ['local', 'frontier'] as const) {
    if (!found.has(id)) {
      found.set(
        id,
        providerRecord(id, {
          available: false,
          reason: health
            ? `${providerLabels[id]}가 서버에 구성되지 않았습니다.`
            : 'API 상태를 확인한 뒤 선택할 수 있습니다.',
        }),
      )
    }
  }

  return (['local', 'frontier'] as const).map(
    (id) => found.get(id) as ProviderCapability,
  )
}

const pipelineLabels: Record<string, string> = {
  parser: 'Parser',
  parsed_block: 'ParsedBlock',
  corpus_gate: 'Corpus Gate',
  chunking: '구조 기반 Chunk',
  bm25: 'BM25',
  dense: 'Dense',
  rrf: 'RRF',
  reranker: 'Reranker',
  local_llm: 'Local LLM',
  generation: '답변 생성',
  citation: 'Citation',
}

function pipelineStage(id: string, value: unknown): PipelineStage {
  const record = isRecord(value)
    ? value
    : typeof value === 'string'
      ? { state: value }
      : {}
  const ready =
    typeof value === 'boolean'
      ? value
      : optionalBoolean(record.ready) ??
        optionalBoolean(record.available) ??
        optionalBoolean(record.enabled)
  const state =
    optionalString(record.state) ??
    optionalString(record.status) ??
    (ready === true ? 'ready' : ready === false ? 'unavailable' : 'unknown')
  return {
    id,
    label: optionalString(record.label) ?? pipelineLabels[id] ?? id.replaceAll('_', ' '),
    state,
    detail:
      optionalString(record.detail) ??
      optionalString(record.reason) ??
      optionalString(record.message),
  }
}

export function getPipelineStages(health: HealthResponse | null): PipelineStage[] {
  if (!health) {
    return []
  }

  const pipeline = health.pipeline
  if (Array.isArray(pipeline)) {
    return pipeline
      .map((value, index) => {
        if (!isRecord(value)) {
          return null
        }
        const id = optionalString(value.id) ?? optionalString(value.name) ?? `stage_${index + 1}`
        return pipelineStage(id, value)
      })
      .filter((stage): stage is PipelineStage => stage !== null)
  }

  if (isRecord(pipeline)) {
    const stages = pipeline.stages
    if (Array.isArray(stages)) {
      return stages
        .map((value, index) => {
          if (!isRecord(value)) {
            return null
          }
          const id = optionalString(value.id) ?? optionalString(value.name) ?? `stage_${index + 1}`
          return pipelineStage(id, value)
        })
        .filter((stage): stage is PipelineStage => stage !== null)
    }
    if (isRecord(stages)) {
      return Object.entries(stages).map(([id, value]) => pipelineStage(id, value))
    }

    const ignored = new Set(['status', 'state', 'ready', 'detail', 'updated_at', 'version'])
    const mapped = Object.entries(pipeline)
      .filter(
        ([id, value]) =>
          !ignored.has(id) &&
          (typeof value === 'boolean' ||
            typeof value === 'string' ||
            isRecord(value)),
      )
      .map(([id, value]) => pipelineStage(id, value))
    if (mapped.length) {
      return mapped
    }
  }

  return [
    pipelineStage('bm25', {
      ready: health.ready,
      detail:
        typeof health.chunk_count === 'number'
          ? `${health.chunk_count.toLocaleString()}개 chunk`
          : undefined,
    }),
  ]
}

export function normalizeGeneration(
  response: ChatResponse,
  requested: GenerationProvider,
): GenerationInfo {
  if (response.generation) {
    const generation = response.generation
    const normalizedRequested = GENERATION_PROVIDERS.includes(
      generation.requested as GenerationProvider,
    )
      ? (generation.requested as GenerationProvider)
      : requested
    return {
      requested: normalizedRequested,
      used: optionalString(generation.used) ?? 'unknown',
      model: optionalString(generation.model) ?? null,
      fallback_reason: optionalString(generation.fallback_reason) ?? null,
      attempts: Array.isArray(generation.attempts) ? generation.attempts : undefined,
    }
  }

  const generator = optionalString(response.generator) ?? 'extractive'
  const [kind, ...modelParts] = generator.split(':')
  const used =
    kind === 'local'
      ? 'local'
      : kind === 'frontier' || kind === 'gemini'
        ? 'frontier'
        : 'extractive'
  return {
    requested,
    used,
    model: modelParts.join(':') || null,
    fallback_reason: generator.includes('_after_') ? generator : null,
  }
}

function defaultErrorMessage(status: number) {
  if (status === 400) return '요청 내용을 확인해 주세요.'
  if (status === 401 || status === 403) return 'API 인증 또는 접근 권한을 확인해 주세요.'
  if (status === 413) return '질문이 서버의 허용 길이를 초과했습니다.'
  if (status === 429) return '요청이 많습니다. 잠시 후 다시 시도해 주세요.'
  if (status === 503) return '검색 또는 생성 서비스가 아직 준비되지 않았습니다.'
  if (status >= 500) return '서버가 요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.'
  return `API 요청에 실패했습니다. (HTTP ${status})`
}

export class RagApiError extends Error {
  readonly code: string
  readonly status: number
  readonly retryable: boolean
  readonly requestId?: string

  constructor(
    message: string,
    options: {
      code: string
      status?: number
      retryable?: boolean
      requestId?: string
    },
  ) {
    super(message)
    this.name = 'RagApiError'
    this.code = options.code
    this.status = options.status ?? 0
    this.retryable = options.retryable ?? false
    this.requestId = options.requestId
  }
}

async function readPayload(response: Response): Promise<unknown> {
  const text = await response.text()
  if (!text) {
    return {}
  }
  try {
    return JSON.parse(text) as unknown
  } catch {
    throw new RagApiError('서버 응답이 올바른 JSON 형식이 아닙니다.', {
      code: 'invalid_response',
      status: response.status,
      retryable: response.status >= 500,
    })
  }
}

async function requestJson(
  path: string,
  init: RequestInit = {},
  timeoutMs = HEALTH_TIMEOUT_MS,
): Promise<unknown> {
  const controller = new AbortController()
  let timedOut = false
  const forwardAbort = () => controller.abort(init.signal?.reason)
  if (init.signal?.aborted) {
    forwardAbort()
  } else {
    init.signal?.addEventListener('abort', forwardAbort, { once: true })
  }
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const headers = new Headers(init.headers)
  headers.set('Accept', 'application/json')
  if (init.body) {
    headers.set('Content-Type', 'application/json')
  }
  if (API_TOKEN) {
    headers.set('X-RAG-API-Key', API_TOKEN)
  }

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    })
    const payload = await readPayload(response)
    if (!response.ok) {
      const error = isRecord(payload) ? (payload as ApiErrorPayload) : {}
      throw new RagApiError(
        optionalString(error.message) ?? defaultErrorMessage(response.status),
        {
          code: optionalString(error.error) ?? `http_${response.status}`,
          status: response.status,
          retryable:
            optionalBoolean(error.retryable) ??
            (response.status === 408 ||
              response.status === 429 ||
              response.status >= 500),
          requestId: optionalString(error.request_id),
        },
      )
    }
    return payload
  } catch (error) {
    if (error instanceof RagApiError) {
      throw error
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new RagApiError(
        timedOut
          ? '답변 생성 시간이 초과되었습니다. 질문 범위를 좁혀 다시 시도해 주세요.'
          : '요청을 취소했습니다.',
        {
          code: timedOut ? 'timeout' : 'cancelled',
          retryable: timedOut,
        },
      )
    }
    throw new RagApiError('API 서버에 연결할 수 없습니다. 서버 상태와 주소를 확인해 주세요.', {
      code: 'network_error',
      retryable: true,
    })
  } finally {
    window.clearTimeout(timer)
    init.signal?.removeEventListener('abort', forwardAbort)
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const payload = await requestJson('/health', { signal })
  if (!isRecord(payload)) {
    throw new RagApiError('상태 응답 형식을 확인할 수 없습니다.', {
      code: 'invalid_health_response',
      retryable: true,
    })
  }
  return {
    ...payload,
    ready: payload.ready === true,
    status: optionalString(payload.status),
    default_provider: GENERATION_PROVIDERS.includes(
      payload.default_provider as GenerationProvider,
    )
      ? (payload.default_provider as GenerationProvider)
      : undefined,
    default_parser_profile:
      parserProfileId(payload.default_parser_profile) ?? undefined,
  }
}

export async function getInstitutions(
  parserProfile?: ParserProfile,
  signal?: AbortSignal,
): Promise<string[]> {
  const query = parserProfile
    ? `?parser_profile=${encodeURIComponent(parserProfile)}`
    : ''
  const payload = await requestJson(`/institutions${query}`, { signal })
  if (!isRecord(payload) || !Array.isArray(payload.institutions)) {
    return []
  }
  return payload.institutions
    .map((item) => optionalString(item))
    .filter((item): item is string => Boolean(item))
}

export async function chat(
  request: ChatRequest,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const payload = await requestJson(
    '/chat',
    {
      method: 'POST',
      body: JSON.stringify(request),
      signal,
    },
    CHAT_TIMEOUT_MS,
  )
  if (!isRecord(payload) || typeof payload.answer !== 'string') {
    throw new RagApiError('답변 응답 형식을 확인할 수 없습니다.', {
      code: 'invalid_chat_response',
      retryable: true,
    })
  }
  return {
    ...(payload as Omit<ChatResponse, 'answer' | 'results'>),
    answer: payload.answer,
    results: Array.isArray(payload.results) ? (payload.results as SearchResult[]) : [],
    claims: Array.isArray(payload.claims) ? (payload.claims as Claim[]) : [],
  }
}
