import {
  AlertTriangle,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Clock3,
  Copy,
  FileSearch,
  FileText,
  History,
  Library,
  ListChecks,
  LoaderCircle,
  Menu,
  PanelRightOpen,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import {
  chat as requestChat,
  getHealth,
  getInstitutions,
  getParserProfileCapabilities,
  getPipelineStages,
  getProviderCapabilities,
  getResultLocations,
  normalizeGeneration,
  parserProfileDisplayName,
  providerDisplayName,
  RagApiError,
  unloadLocalModel,
} from './api/rag'
import type {
  ChatRequest,
  ChatTrace,
  Claim,
  GenerationInfo,
  GenerationProvider,
  HealthResponse,
  ParserProfile,
  SearchResult,
} from './api/rag'
import CitationLocation from './components/CitationLocation'
import PipelineStatus from './components/PipelineStatus'
import ProviderSelect from './components/ProviderSelect'
import './App.css'

type Role = 'user' | 'assistant'
type MobilePanel = 'nav' | 'sources' | null
type SourceTab = 'claims' | 'sources' | 'locations'

type Message = {
  id: string
  role: Role
  content: string
  results?: SearchResult[]
  claims?: Claim[]
  status?: 'search' | 'error'
  generation?: GenerationInfo
  trace?: ChatTrace
  retrieval?: Record<string, unknown>
  parserProfile?: ParserProfile
  error?: {
    code: string
    retryable: boolean
    requestId?: string
  }
  request?: ChatRequest
}

type SubmitQuestionOptions = {
  institution?: string
  provider?: GenerationProvider
  topK?: number
  model?: string | null
  parserProfile?: ParserProfile
  appendUser?: boolean
}

function envNumber(value: unknown, fallback: number) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

const MAX_QUESTION_CHARS = envNumber(import.meta.env.VITE_MAX_QUESTION_CHARS, 1000)
const MAX_SOURCE_LOCATIONS = 3
const MAX_DETAIL_LOCATIONS = 8
const sanjiniSrc = '/sanjini.webp'
const allInstitutions = '전체 기관'
const evidenceScopePresets = [
  {
    value: 4,
    label: '정밀',
    detail: '관련성이 높은 근거에 집중합니다.',
  },
  {
    value: 8,
    label: '균형',
    detail: '정확도와 검색 범위의 균형을 맞춥니다.',
  },
  {
    value: 12,
    label: '확장',
    detail: '여러 규정과 문서를 폭넓게 살핍니다.',
  },
] as const
type EvidenceTopK = (typeof evidenceScopePresets)[number]['value']
const defaultEvidenceTopK: EvidenceTopK = 8
const defaultInstitutions = [
  allInstitutions,
  '금융감독원',
  '한국은행',
  '한국거래소',
  '한국예탁결제원',
  '부산대학교',
  '한국인터넷진흥원(KISA)',
  '한국해양과학기술원',
]

const suggestedQuestions = [
  {
    label: '상장폐지 제도',
    institution: '한국거래소',
    question: '상장폐지 제도 개선 내용을 핵심만 알려줘',
  },
  {
    label: '신탁 수탁고',
    institution: '금융감독원',
    question: '신탁 수탁고 현황을 찾아서 요약해줘',
  },
  {
    label: '지급결제 리스크',
    institution: '한국은행',
    question: '한국은행 지급결제 리스크 문서를 찾아줘',
  },
  {
    label: '휴학 규정',
    institution: '부산대학교',
    question: '부산대학교 휴학 관련 규정을 검색해줘',
  },
]

const recentQueries = [
  {
    label: '상장폐지 공시',
    institution: '한국거래소',
  },
  {
    label: '신탁 수탁고 현황',
    institution: '금융감독원',
  },
  {
    label: '지급결제 리스크',
    institution: '한국은행',
  },
  {
    label: '휴학 신청',
    institution: '부산대학교',
  },
]

function makeId() {
  return crypto.randomUUID()
}

function cleanPreview(value: string) {
  return value.replace(/\s+/g, ' ').trim()
}

function resultScoreLabel(result: SearchResult) {
  const score =
    typeof result.score === 'number'
      ? result.score
      : Object.values(result.scores ?? {}).find(
          (value): value is number => typeof value === 'number',
        )
  return typeof score === 'number' ? Math.abs(score).toFixed(2) : '—'
}

function resultLocationSummary(result: SearchResult, limit: number) {
  const locations = getResultLocations(result)
  const reportedCount =
    typeof result.location_count === 'number'
      ? result.location_count
      : locations.length
  return {
    locations: locations.slice(0, limit),
    hiddenCount: Math.max(reportedCount - Math.min(locations.length, limit), 0),
  }
}

function uniqueDocumentCount(results?: SearchResult[]) {
  if (!results?.length) {
    return 0
  }
  return new Set(
    results.map((result) => {
      const stableId = [
        result.document_id,
        result.doc_id,
        result.relative_path,
        result.source_path,
      ].find((value) => typeof value === 'string' && value.trim())
      if (stableId) {
        return stableId
      }
      if (result.file_name) {
        return `${result.institution ?? ''}:${result.file_name}`
      }
      return result.chunk_id
    }),
  ).size
}

function retrievalSummary(retrieval?: Record<string, unknown>) {
  if (!retrieval) {
    return null
  }
  const strategy = [retrieval.strategy, retrieval.mode, retrieval.fusion]
    .find((value) => typeof value === 'string')
  if (typeof strategy === 'string' && strategy.trim()) {
    return strategy
  }
  const resultCount = [retrieval.result_count, retrieval.results, retrieval.top_k]
    .find((value) => typeof value === 'number')
  return typeof resultCount === 'number' ? `검색 결과 ${resultCount}개` : null
}

function contextDeduplicationSummary(retrieval?: Record<string, unknown>) {
  const value = retrieval?.context_deduplication
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const diagnostics = value as Record<string, unknown>
  const removedCount = diagnostics.removed_count
  const keptCount = diagnostics.kept_count
  if (typeof removedCount !== 'number' || typeof keptCount !== 'number') {
    return null
  }
  return `중복 근거 ${removedCount}개 제거 · 최종 ${keptCount}개`
}

function criticalValueLabel(value: string) {
  const [kind, ...parts] = value.split(':')
  const normalized = parts.join(':')
  if (kind === 'academic_year') return `${normalized}학년도`
  if (kind === 'semester') return `${normalized}학기`
  if (kind === 'round') return `${normalized}차`
  if (kind === 'date') return normalized
  if (kind === 'month_day') {
    const [month, day] = normalized.split('-').map(Number)
    return Number.isFinite(month) && Number.isFinite(day)
      ? `${month}월 ${day}일`
      : normalized
  }
  if (kind === 'time_minutes') {
    const minutes = Number(normalized)
    if (Number.isFinite(minutes)) {
      const hour = Math.floor(minutes / 60)
      const minute = minutes % 60
      return `${hour.toString().padStart(2, '0')}:${minute
        .toString()
        .padStart(2, '0')}`
    }
  }
  if (kind === 'amount_krw') {
    const amount = Number(normalized)
    return Number.isFinite(amount)
      ? `${amount.toLocaleString('ko-KR')}원`
      : `${normalized}원`
  }
  if (kind === 'percent') return `${normalized}%`
  if (kind === 'quantity') return normalized.replace(':', '')
  if (kind === 'phone') return `전화번호 ${normalized}`
  return value
}

function claimValidationLabel(claim: Claim) {
  if (claim.supported) {
    return '근거 확인'
  }
  if (claim.validation_reason === 'model_abstention') {
    return '모델이 근거 부족으로 답변 보류'
  }
  if (claim.validation_reason === 'critical_value_mismatch') {
    const missing = (claim.missing_critical_values ?? [])
      .map(criticalValueLabel)
      .join(', ')
    return missing
      ? `핵심 값을 근거에서 확인하지 못함 · ${missing}`
      : '핵심 값을 근거에서 확인하지 못함'
  }
  if (claim.validation_reason === 'low_lexical_overlap') {
    return '검색 근거와 연결 부족'
  }
  return '근거 부족'
}

function traceSummary(trace?: ChatTrace) {
  const stages = trace?.stages ?? []
  if (!stages.length) {
    return null
  }
  const totalDuration = stages.reduce(
    (sum, stage) =>
      sum + (typeof stage.duration_ms === 'number' ? stage.duration_ms : 0),
    0,
  )
  return totalDuration > 0
    ? `${stages.length}단계 · ${(totalDuration / 1000).toFixed(1)}초`
    : `처리 단계 ${stages.length}개`
}

function requestStageClass(state: string) {
  const normalized = state.toLowerCase()
  if (
    [
      'ready',
      'healthy',
      'available',
      'configured',
      'complete',
      'external',
      'disabled',
      'single_lane',
    ].includes(normalized)
  ) {
    return 'is-done'
  }
  if (['building', 'loading', 'running', 'pending', 'checking'].includes(normalized)) {
    return 'is-active'
  }
  if (['degraded', 'fallback', 'partial'].includes(normalized)) {
    return 'is-degraded'
  }
  return 'is-unavailable'
}

function normalizeAnswerLines(content: string) {
  return content
    .trim()
    .replace(/\s+(?=-\s+)/g, '\n')
    .replace(/\s+(?=\d+[.)]\s+)/g, '\n')
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
}

function getBulletText(line: string) {
  return line.replace(/^[-*•]\s+/, '').replace(/^\d+[.)]\s+/, '')
}

function isBulletLine(line: string) {
  return /^[-*•]\s+/.test(line) || /^\d+[.)]\s+/.test(line)
}

function renderWithCitations(
  text: string,
  onCitationSelect?: (sourceNumber: number) => void,
) {
  return text.split(/(\[\d+\])/g).map((part, index) =>
    /^\[\d+\]$/.test(part) ? (
      <button
        aria-label={`근거 ${part.slice(1, -1)}번 보기`}
        className="citation-token"
        key={`${part}-${index}`}
        onClick={(event) => {
          event.stopPropagation()
          onCitationSelect?.(Number(part.slice(1, -1)))
        }}
        type="button"
      >
        {part}
      </button>
    ) : (
      part
    ),
  )
}

function AnswerContent({
  content,
  onCitationSelect,
}: {
  content: string
  onCitationSelect?: (sourceNumber: number) => void
}) {
  const lines = normalizeAnswerLines(content)

  return (
    <div className="answer-content">
      {lines.map((line, index) =>
        isBulletLine(line) ? (
          <div className="answer-bullet" key={`${line}-${index}`}>
            <span aria-hidden="true" />
            <p>{renderWithCitations(getBulletText(line), onCitationSelect)}</p>
          </div>
        ) : (
          <p key={`${line}-${index}`}>
            {renderWithCitations(line, onCitationSelect)}
          </p>
        ),
      )}
    </div>
  )
}

function getSupportedClaimCount(message: Message) {
  return message.claims?.filter((claim) => claim.supported).length ?? 0
}

function weakAnswerHint(message: Message) {
  if (message.status === 'error') {
    return null
  }

  const claimCount = message.claims?.length ?? 0
  const supportedCount = getSupportedClaimCount(message)
  const sourceCount = message.results?.length ?? 0

  if (sourceCount === 0) {
    return '검색된 근거가 없습니다. 기관 범위나 질문 표현을 바꿔 다시 검색해 보세요.'
  }
  if (
    claimCount === 0 ||
    supportedCount >= Math.ceil(claimCount * 0.6)
  ) {
    return null
  }
  if (
    message.claims?.every(
      (claim) => claim.validation_reason === 'model_abstention',
    )
  ) {
    return '모델이 제공된 검색 근거만으로 답하기 어렵다고 판단했습니다. 검증 탭에서 판정 내용을 확인할 수 있습니다.'
  }
  return '모델 초안 중 근거가 부족한 문장은 최종 답변에서 제외했습니다. 검증 탭에서 판정 이유를 확인할 수 있습니다.'
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [institution, setInstitution] = useState(allInstitutions)
  const [institutions, setInstitutions] = useState(defaultInstitutions)
  const [provider, setProvider] = useState<GenerationProvider>('auto')
  const [parserProfile, setParserProfile] = useState<ParserProfile>('cascade')
  const [localModelPreference, setLocalModelPreference] = useState<string | null>(null)
  const [topK, setTopK] = useState<EvidenceTopK>(defaultEvidenceTopK)
  const [pendingProvider, setPendingProvider] = useState<GenerationProvider | null>(null)
  const [pendingParserProfile, setPendingParserProfile] =
    useState<ParserProfile | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [isHealthRefreshing, setIsHealthRefreshing] = useState(true)
  const [isLocalModelUnloading, setIsLocalModelUnloading] = useState(false)
  const [localModelUnloadError, setLocalModelUnloadError] =
    useState<string | null>(null)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null)
  const [sourceTab, setSourceTab] = useState<SourceTab>('claims')
  const [isLoading, setIsLoading] = useState(false)
  const messageStreamRef = useRef<HTMLDivElement | null>(null)
  const activeRequestRef = useRef<AbortController | null>(null)
  const localModelUnloadRequestRef = useRef(false)
  const healthRefreshSequenceRef = useRef(0)
  const defaultProviderAppliedRef = useRef(false)
  const defaultParserProfileAppliedRef = useRef(false)

  const assistantMessages = messages.filter((message) => message.role === 'assistant')
  const providerCapabilities = useMemo(
    () => getProviderCapabilities(health),
    [health],
  )
  const parserProfileCapabilities = useMemo(
    () => getParserProfileCapabilities(health),
    [health],
  )
  const selectedParserProfileCapability = useMemo(
    () =>
      parserProfileCapabilities.find(
        (capability) => capability.id === parserProfile,
      ),
    [parserProfile, parserProfileCapabilities],
  )
  const selectedParserProfileReady =
    health?.ready === true &&
    selectedParserProfileCapability?.ready === true
  const localProviderCapability = useMemo(
    () => providerCapabilities.find((capability) => capability.id === 'local'),
    [providerCapabilities],
  )
  const selectedLocalModel = useMemo(() => {
    const models = localProviderCapability?.models ?? []
    const preferred = localModelPreference
      ? models.find(
          (model) => model.id === localModelPreference && model.available,
        )
      : undefined
    if (preferred) {
      return preferred.id
    }

    const configured =
      localProviderCapability?.defaultModel ?? localProviderCapability?.model
    const configuredModel = configured
      ? models.find((model) => model.id === configured)
      : undefined
    if (configured && (!configuredModel || configuredModel.available)) {
      return configured
    }
    return models.find((model) => model.available)?.id ?? ''
  }, [localModelPreference, localProviderCapability])
  const pipelineStages = useMemo(() => getPipelineStages(health), [health])
  const availableSuggestedQuestions = useMemo(
    () =>
      suggestedQuestions.filter((item) =>
        institutions.includes(item.institution),
      ),
    [institutions],
  )
  const availableRecentQueries = useMemo(
    () =>
      recentQueries.filter((item) =>
        institutions.includes(item.institution),
      ),
    [institutions],
  )
  const selectedAnswer = useMemo(
    () =>
      assistantMessages.find((message) => message.id === selectedMessageId) ??
      assistantMessages.at(-1) ??
      null,
    [assistantMessages, selectedMessageId],
  )

  const refreshStatus = useCallback(async (signal?: AbortSignal) => {
    const sequence = healthRefreshSequenceRef.current + 1
    healthRefreshSequenceRef.current = sequence
    setIsHealthRefreshing(true)
    const [healthResult, institutionResult] = await Promise.allSettled([
      getHealth(signal),
      getInstitutions(parserProfile, signal),
    ])
    if (
      signal?.aborted ||
      sequence !== healthRefreshSequenceRef.current
    ) {
      return
    }

    if (healthResult.status === 'fulfilled') {
      const nextHealth = healthResult.value
      setHealth(nextHealth)
      setHealthError(
        nextHealth.ready
          ? null
          : '검색 인덱스 또는 필수 파이프라인이 준비되지 않았습니다.',
      )
      const nextCapabilities = getProviderCapabilities(nextHealth)
      setLocalModelUnloadError(null)
      if (!defaultProviderAppliedRef.current) {
        const preferred = nextHealth.default_provider ?? 'auto'
        const preferredCapability = nextCapabilities.find(
          (capability) => capability.id === preferred,
        )
        setProvider(
          preferred === 'auto' || preferredCapability?.available
            ? preferred
            : 'auto',
        )
        defaultProviderAppliedRef.current = true
      } else {
        setProvider((current) =>
          current === 'auto' ||
          nextCapabilities.some(
            (capability) =>
              capability.id === current && capability.available,
          )
            ? current
            : 'auto',
        )
      }
      const nextParserProfiles = getParserProfileCapabilities(nextHealth)
      const preferredParserProfile =
        nextHealth.default_parser_profile ?? 'cascade'
      if (!defaultParserProfileAppliedRef.current) {
        const preferredCapability = nextParserProfiles.find(
          (capability) =>
            capability.id === preferredParserProfile && capability.ready,
        )
        setParserProfile(
          preferredCapability?.id ??
            nextParserProfiles.find((capability) => capability.ready)?.id ??
            preferredParserProfile,
        )
        defaultParserProfileAppliedRef.current = true
      } else {
        setParserProfile((current) =>
          nextParserProfiles.some(
            (capability) => capability.id === current && capability.ready,
          )
            ? current
            : nextParserProfiles.find((capability) => capability.ready)?.id ??
              preferredParserProfile,
        )
      }
    } else {
      setHealth(null)
      setHealthError(
        healthResult.reason instanceof Error
          ? healthResult.reason.message
          : 'API 상태를 확인할 수 없습니다.',
      )
    }

    if (
      institutionResult.status === 'fulfilled' &&
      institutionResult.value.length > 0
    ) {
      setInstitutions([
        allInstitutions,
        ...institutionResult.value.filter((value) => value !== allInstitutions),
      ])
    }
    setIsHealthRefreshing(false)
  }, [parserProfile])

  useEffect(() => {
    const controller = new AbortController()
    const initialTimer = window.setTimeout(
      () => void refreshStatus(controller.signal),
      0,
    )
    const timer = window.setInterval(() => void refreshStatus(), 30000)
    return () => {
      controller.abort()
      window.clearTimeout(initialTimer)
      window.clearInterval(timer)
    }
  }, [refreshStatus])

  useEffect(
    () => () => {
      activeRequestRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    const stream = messageStreamRef.current
    if (!stream) {
      return
    }

    requestAnimationFrame(() => {
      stream.scrollTop = stream.scrollHeight
    })
  }, [isLoading, messages])

  const activeInstitution = institution === allInstitutions ? undefined : institution

  async function submitQuestion(
    question: string,
    options: SubmitQuestionOptions = {},
  ) {
    const institutionOverride = Object.prototype.hasOwnProperty.call(
      options,
      'institution',
    )
      ? options.institution
      : activeInstitution
    const providerOverride = options.provider ?? provider
    const topKOverride = options.topK ?? topK
    const parserProfileOverride =
      options.parserProfile ?? parserProfile
    const modelOverride = Object.prototype.hasOwnProperty.call(options, 'model')
      ? options.model
      : selectedLocalModel
    const appendUser = options.appendUser ?? true
    const trimmed = question.trim()
    if (
      !trimmed ||
      activeRequestRef.current ||
      health?.ready !== true ||
      !parserProfileCapabilities.some(
        (capability) =>
          capability.id === parserProfileOverride && capability.ready,
      )
    ) {
      return
    }
    if (trimmed.length > MAX_QUESTION_CHARS) {
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'error',
        content: `질문은 ${MAX_QUESTION_CHARS.toLocaleString()}자 이내로 입력해 주세요.`,
        results: [],
        error: {
          code: 'question_too_long',
          retryable: false,
        },
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      return
    }

    const normalizedModel = modelOverride?.trim()
    const request: ChatRequest = {
      question: trimmed,
      institution: institutionOverride,
      provider: providerOverride,
      top_k: topKOverride,
      parser_profile: parserProfileOverride,
      ...(providerOverride === 'local' && normalizedModel
        ? { model: normalizedModel }
        : {}),
    }

    if (appendUser) {
      const userMessage: Message = {
        id: makeId(),
        role: 'user',
        content: trimmed,
      }
      setMessages((current) => [...current, userMessage])
    }
    setInput('')
    setIsLoading(true)
    if (providerOverride === 'local' || providerOverride === 'auto') {
      setLocalModelUnloadError(null)
    }
    setPendingProvider(providerOverride)
    setPendingParserProfile(parserProfileOverride)

    const controller = new AbortController()
    activeRequestRef.current = controller

    try {
      const data = await requestChat(request, controller.signal)
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'search',
        content: data.cited_answer ?? data.answer,
        claims: data.claims ?? [],
        results: data.results,
        generation: normalizeGeneration(data, providerOverride),
        trace: data.trace,
        retrieval: data.retrieval,
        parserProfile: data.parser_profile ?? parserProfileOverride,
        request,
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab(answer.claims?.length ? 'claims' : 'sources')
    } catch (error) {
      const apiError =
        error instanceof RagApiError
          ? error
          : new RagApiError('요청을 처리하지 못했습니다. 다시 시도해 주세요.', {
              code: 'unknown_error',
              retryable: true,
            })
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'error',
        content: apiError.message,
        results: [],
        error: {
          code: apiError.code,
          retryable: apiError.retryable,
          requestId: apiError.requestId,
        },
        request,
        parserProfile: parserProfileOverride,
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab('sources')
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null
        setIsLoading(false)
        setPendingProvider(null)
        setPendingParserProfile(null)
      }
      if (providerOverride === 'local' || providerOverride === 'auto') {
        void refreshStatus()
      }
    }
  }

  async function handleLocalModelUnload() {
    if (
      isLoading ||
      localModelUnloadRequestRef.current ||
      isLocalModelUnloading
    ) {
      return
    }
    localModelUnloadRequestRef.current = true
    setIsLocalModelUnloading(true)
    setLocalModelUnloadError(null)
    try {
      await unloadLocalModel()
      await refreshStatus()
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : '로컬 모델을 메모리에서 내리지 못했습니다.'
      await refreshStatus()
      setLocalModelUnloadError(message)
    } finally {
      localModelUnloadRequestRef.current = false
      setIsLocalModelUnloading(false)
    }
  }

  const cancelRequest = () => {
    activeRequestRef.current?.abort()
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    void submitQuestion(input)
  }

  const selectedSupportedCount = selectedAnswer ? getSupportedClaimCount(selectedAnswer) : 0
  const selectedClaimCount = selectedAnswer?.claims?.length ?? 0
  const selectedSourceCount = selectedAnswer?.results?.length ?? 0
  const selectedDocumentCount = uniqueDocumentCount(selectedAnswer?.results)
  const selectedUnsupportedCount = Math.max(selectedClaimCount - selectedSupportedCount, 0)

  return (
    <main className="app-shell">
      <aside className={`left-rail ${mobilePanel === 'nav' ? 'is-open' : ''}`}>
        <div className="brand">
          <img src={sanjiniSrc} alt="부산대학교 마스코트 산지니" />
          <div>
            <strong>공문서 RAG</strong>
            <span>기관 문서 검색 챗봇</span>
          </div>
        </div>

        <button
          className="new-chat-button"
          disabled={isLoading}
          onClick={() => {
            setMessages([])
            setSelectedMessageId(null)
          }}
          type="button"
        >
          <Sparkles size={18} />새 질문
        </button>

        <section className="rail-section">
          <div className="section-title">
            <Search size={15} />
            검색 기관
          </div>
          <div className="institution-control">
            <select
              aria-label="검색 기관 선택"
              disabled={isLoading}
              onChange={(event) => setInstitution(event.target.value)}
              value={institution}
            >
              {institutions.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
            <span>{institution === allInstitutions ? '전체 문서에서 검색합니다.' : `${institution} 문서만 검색합니다.`}</span>
          </div>
          <div className="institution-shortcuts">
            {institutions.slice(0, 5).map((option) => (
              <button
                className={institution === option ? 'is-active' : ''}
                disabled={isLoading}
                key={option}
                onClick={() => setInstitution(option)}
                type="button"
              >
                {option}
              </button>
            ))}
          </div>
        </section>

        <section className="rail-section">
          <div className="section-title">
            <Library size={15} />
            파이프라인 상태
          </div>
          <PipelineStatus
            chunkCount={selectedParserProfileCapability?.chunkCount}
            error={healthError}
            health={health}
            onRefresh={() => void refreshStatus()}
            profileLabel={parserProfileDisplayName(parserProfile, true)}
            profileReady={selectedParserProfileCapability?.ready}
            refreshing={isHealthRefreshing}
          />
        </section>

        <section className="rail-section history-section">
          <div className="section-title">
            <History size={15} />
            빠른 검색
          </div>
          <div className="conversation-list">
            {availableRecentQueries.map((item) => (
              <button
                disabled={isLoading || !selectedParserProfileReady}
                key={item.label}
                onClick={() => {
                  setInstitution(item.institution)
                  void submitQuestion(item.label, {
                    institution: item.institution,
                  })
                }}
                type="button"
              >
                {item.label}
              </button>
            ))}
          </div>
        </section>
      </aside>

      {mobilePanel && (
        <button
          aria-label="패널 닫기"
          className="mobile-scrim"
          onClick={() => setMobilePanel(null)}
          type="button"
        />
      )}

      <section className="chat-column">
        <header className="chat-header">
          <button
            aria-label="메뉴 열기"
            className="icon-button mobile-only"
            onClick={() => setMobilePanel('nav')}
            type="button"
          >
            <Menu size={19} />
          </button>

          <div>
            <span className="eyebrow">PNU Docs AI</span>
            <h1>기관 공문서에서 근거를 검색합니다</h1>
          </div>

          <div className="header-actions">
            <span className="scope-pill">
              {institution} · {parserProfileDisplayName(parserProfile, true)}
            </span>
            <button
              aria-label="근거 패널 열기"
              className="icon-button mobile-only"
              onClick={() => setMobilePanel('sources')}
              type="button"
            >
              <PanelRightOpen size={19} />
            </button>
          </div>
        </header>

        <div aria-busy={isLoading} className="message-stream" ref={messageStreamRef}>
          {messages.length === 0 ? (
            <section className="empty-state">
              <img src={sanjiniSrc} alt="" />
              <span>
                {isHealthRefreshing && !health
                  ? '검색 파이프라인 확인 중'
                  : selectedParserProfileReady
                    ? '검색 파이프라인 준비 완료'
                    : '검색 파이프라인 준비 필요'}
              </span>
              <h2>질문을 입력하면 파싱된 공문서 chunk에서 근거 후보를 찾아옵니다</h2>
              <div className="suggestion-grid">
                {availableSuggestedQuestions.map((item) => (
                  <button
                    disabled={isLoading || !selectedParserProfileReady}
                    key={item.question}
                    onClick={() => {
                      setInstitution(item.institution)
                      void submitQuestion(item.question, {
                        institution: item.institution,
                      })
                    }}
                    type="button"
                  >
                    <BookOpen size={17} />
                    <span>
                      <strong>{item.label}</strong>
                      <small>{item.institution}</small>
                    </span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <>
              {messages.map((message) => {
                const supportedClaimCount = getSupportedClaimCount(message)
                const claimCount = message.claims?.length ?? 0
                const sourceCount = message.results?.length ?? 0
                const documentCount = uniqueDocumentCount(message.results)
                const requestTrace = traceSummary(message.trace)
                const retrieval = retrievalSummary(message.retrieval)
                const deduplication = contextDeduplicationSummary(message.retrieval)
                const hasFallback = Boolean(message.generation?.fallback_reason)
                const weakAnswer = weakAnswerHint(message)

                return (
                  <article
                    className={`message-row ${message.role}`}
                    key={message.id}
                    onClick={() => {
                      if (message.role === 'assistant') {
                        setSelectedMessageId(message.id)
                        setSourceTab(message.claims?.length ? 'claims' : 'sources')
                      }
                    }}
                  >
                    {message.role === 'assistant' && (
                      <img className="avatar" src={sanjiniSrc} alt="산지니" />
                    )}
                    <div className="message-bubble">
                      {message.role === 'assistant' && (
                        <div className="answer-meta">
                          {message.status === 'error' ? <ShieldCheck size={16} /> : <CheckCircle2 size={16} />}
                          <span>
                            {message.status === 'error'
                              ? message.error?.code === 'cancelled'
                                ? '요청 취소됨'
                                : '요청 처리 실패'
                              : `검증된 문장 ${supportedClaimCount}개`}
                          </span>
                        </div>
                      )}

                      {message.role === 'assistant' ? (
                        <AnswerContent
                          content={message.content}
                          onCitationSelect={() => {
                            setSelectedMessageId(message.id)
                            setSourceTab('sources')
                            setMobilePanel('sources')
                          }}
                        />
                      ) : (
                        <p>{message.content}</p>
                      )}

                      {message.role === 'assistant' && message.status !== 'error' && (
                        <div className="answer-stats">
                          {message.parserProfile && (
                            <span className="parser-profile-badge">
                              파서{' '}
                              {parserProfileDisplayName(
                                message.parserProfile,
                                true,
                              )}
                            </span>
                          )}
                          <span>근거 chunk {sourceCount}개</span>
                          <span>고유 문서 {documentCount}개</span>
                          <span>검증 문장 {claimCount ? `${supportedClaimCount}/${claimCount}` : '0개'}</span>
                          {message.generation && (
                            <span>
                              {providerDisplayName(message.generation.used)}
                              {message.generation.model ? ` · ${message.generation.model}` : ''}
                            </span>
                          )}
                          {retrieval && <span>{retrieval}</span>}
                          {deduplication && <span>{deduplication}</span>}
                          {requestTrace && <span>{requestTrace}</span>}
                        </div>
                      )}

                      {message.role === 'assistant' && hasFallback && (
                        <div className="answer-hint generation-fallback">
                          <AlertTriangle size={15} />
                          <span>
                            요청한 생성 경로를 사용할 수 없어 다른 경로로 답변했습니다.
                            {message.generation?.fallback_reason
                              ? ` (${message.generation.fallback_reason})`
                              : ''}
                          </span>
                        </div>
                      )}

                      {message.role === 'assistant' && weakAnswer && (
                        <div className="answer-hint">
                          <AlertTriangle size={15} />
                          <span>{weakAnswer}</span>
                        </div>
                      )}

                      {message.role === 'assistant' && (
                        <div className="message-actions">
                          <button
                            onClick={(event) => {
                              event.stopPropagation()
                              void navigator.clipboard.writeText(message.content)
                            }}
                            type="button"
                          >
                            <Copy size={15} />
                            복사
                          </button>
                          {message.status !== 'error' && (
                            <button
                              onClick={(event) => {
                                event.stopPropagation()
                                setSelectedMessageId(message.id)
                                setSourceTab('sources')
                                setMobilePanel('sources')
                              }}
                              type="button"
                            >
                              <FileSearch size={15} />
                              근거 {sourceCount}개
                            </button>
                          )}
                          {message.status === 'error' &&
                            message.error?.retryable &&
                            message.request && (
                              <button
                                disabled={isLoading || !selectedParserProfileReady}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  void submitQuestion(
                                    message.request?.question ?? '',
                                    {
                                      institution: message.request?.institution,
                                      provider: message.request?.provider ?? 'auto',
                                      topK:
                                        message.request?.top_k ??
                                        defaultEvidenceTopK,
                                      model: message.request?.model ?? null,
                                      parserProfile:
                                        message.request?.parser_profile ??
                                        'cascade',
                                      appendUser: false,
                                    },
                                  )
                                }}
                                type="button"
                              >
                                <RotateCcw size={15} />
                                다시 시도
                              </button>
                            )}
                        </div>
                      )}
                      {message.error?.requestId && (
                        <small className="request-id">
                          요청 ID {message.error.requestId}
                        </small>
                      )}
                    </div>
                  </article>
                )
              })}

              {isLoading && (
                <article aria-live="polite" className="message-row assistant is-loading">
                  <img className="avatar" src={sanjiniSrc} alt="산지니" />
                  <div className="message-bubble loading-bubble">
                    <div className="answer-meta">
                      <LoaderCircle className="loading-icon" size={16} />
                      <span>질문 처리 중</span>
                    </div>
                    <p>
                      {parserProfileDisplayName(
                        pendingParserProfile ?? parserProfile,
                        true,
                      )}{' '}
                      파서와 {providerDisplayName(pendingProvider ?? provider)} 경로로
                      검색, 답변 생성, 인용 검증을 요청했습니다.
                    </p>
                    {pipelineStages.length > 0 && (
                      <ol
                        aria-label="서버가 보고한 파이프라인 상태"
                        className="loading-steps"
                      >
                        {pipelineStages.map((stage) => (
                        <li
                            className={requestStageClass(stage.state)}
                            key={stage.id}
                            title={stage.detail}
                        >
                            {stage.label}
                        </li>
                      ))}
                      </ol>
                    )}
                    <div className="loading-footer">
                      <small>표시는 서버의 현재 준비 상태이며 요청별 완료 상태는 응답 후 반영됩니다.</small>
                      <button onClick={cancelRequest} type="button">
                        <X size={14} />
                        취소
                      </button>
                    </div>
                  </div>
                </article>
              )}
            </>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-options">
            <ProviderSelect
              activeRequestProvider={isLoading ? pendingProvider : null}
              disabled={isLoading}
              model={selectedLocalModel}
              onChange={setProvider}
              onModelChange={setLocalModelPreference}
              onUnloadLocalModel={() => void handleLocalModelUnload()}
              providers={providerCapabilities}
              unloadError={localModelUnloadError}
              unloadingLocalModel={isLocalModelUnloading}
              value={provider}
            />
            <label className="parser-profile-select">
              <span>검색 파싱 버전</span>
              <select
                aria-label="검색 파싱 버전"
                disabled={isLoading}
                onChange={(event) =>
                  setParserProfile(event.target.value as ParserProfile)
                }
                value={parserProfile}
              >
                {parserProfileCapabilities.map((capability) => (
                  <option
                    disabled={!capability.ready}
                    key={capability.id}
                    value={capability.id}
                  >
                    {capability.label}
                  </option>
                ))}
              </select>
              <small>
                {selectedParserProfileCapability?.ready
                  ? `${selectedParserProfileCapability.documentCount?.toLocaleString() ?? '-'}개 문서 · ${selectedParserProfileCapability.chunkCount?.toLocaleString() ?? '-'}개 chunk`
                  : selectedParserProfileCapability?.reason ??
                    '선택한 파서 인덱스를 확인할 수 없습니다.'}
              </small>
            </label>
            <label className="composer-institution">
              <span>검색 기관</span>
              <select
                aria-label="검색 기관"
                disabled={isLoading}
                onChange={(event) => setInstitution(event.target.value)}
                value={institution}
              >
                {institutions.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
          </div>
          <fieldset className="evidence-scope">
            <legend>검색 근거 범위</legend>
            <div className="evidence-scope-options">
              {evidenceScopePresets.map((preset) => (
                <button
                  aria-pressed={topK === preset.value}
                  className={topK === preset.value ? 'is-active' : ''}
                  disabled={isLoading}
                  key={preset.value}
                  onClick={() => setTopK(preset.value)}
                  title={preset.detail}
                  type="button"
                >
                  <strong>{preset.label}</strong>
                  <span>
                    {preset.value}개
                    {preset.value === defaultEvidenceTopK ? ' · 기본' : ''}
                  </span>
                </button>
              ))}
            </div>
          </fieldset>
          {!selectedParserProfileReady && (
            <div className="composer-notice" role="status">
              <AlertTriangle size={14} />
              <span>
                {isHealthRefreshing
                  ? '검색 파이프라인 상태를 확인하고 있습니다.'
                  : selectedParserProfileCapability?.reason ??
                    healthError ??
                    '검색 파이프라인이 준비되지 않았습니다.'}
              </span>
            </div>
          )}
          <div className="input-row">
            <textarea
              aria-label="질문 입력"
              disabled={isLoading || !selectedParserProfileReady}
              maxLength={MAX_QUESTION_CHARS}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  event.currentTarget.form?.requestSubmit()
                }
              }}
              placeholder="예: 상장폐지 제도 개선 내용을 찾아줘"
              rows={1}
              value={input}
            />
            <button
              aria-label="질문 보내기"
              className="send-button"
              disabled={
                isLoading ||
                !selectedParserProfileReady ||
                input.trim().length === 0
              }
              type="submit"
            >
              <ArrowUp size={19} />
            </button>
          </div>
        </form>
      </section>

      <aside className={`source-panel ${mobilePanel === 'sources' ? 'is-open' : ''}`}>
        <div className="source-panel-header">
          <div>
            <span className="eyebrow">Evidence</span>
            <h2>근거 문서</h2>
          </div>
          <button
            aria-label="근거 패널 닫기"
            className="icon-button mobile-only"
            onClick={() => setMobilePanel(null)}
            type="button"
          >
            <X size={19} />
          </button>
        </div>

        {selectedAnswer ? (
          <>
            <div className="confidence-box">
              <ShieldCheck size={21} />
              <div>
                <strong>근거 chunk {selectedSourceCount}개</strong>
                <span>
                  고유 문서 {selectedDocumentCount}개 · 검증{' '}
                  {selectedSupportedCount}/{selectedClaimCount || 0}
                  {selectedUnsupportedCount > 0 ? `, 근거 부족 ${selectedUnsupportedCount}` : ''}
                </span>
              </div>
            </div>

            <div className="source-tabs" role="tablist" aria-label="근거 보기 방식">
              <button
                aria-selected={sourceTab === 'claims'}
                className={sourceTab === 'claims' ? 'is-active' : ''}
                onClick={() => setSourceTab('claims')}
                role="tab"
                type="button"
              >
                <ListChecks size={15} />
                검증
              </button>
              <button
                aria-selected={sourceTab === 'sources'}
                className={sourceTab === 'sources' ? 'is-active' : ''}
                onClick={() => setSourceTab('sources')}
                role="tab"
                type="button"
              >
                <FileSearch size={15} />
                문서
              </button>
              <button
                aria-selected={sourceTab === 'locations'}
                className={sourceTab === 'locations' ? 'is-active' : ''}
                onClick={() => setSourceTab('locations')}
                role="tab"
                type="button"
              >
                <FileText size={15} />
                위치
              </button>
            </div>

            {sourceTab === 'claims' && (
              <>
                {selectedAnswer.claims && selectedAnswer.claims.length > 0 ? (
                  <div className="claim-list">
                    {selectedAnswer.claims.map((claim) => (
                      <article className="claim-card" key={claim.text}>
                        <p>{claim.text}</p>
                        <div className="claim-meta">
                          <span className={claim.supported ? 'is-supported' : 'is-unsupported'}>
                            {claimValidationLabel(claim)}
                          </span>
                          {(claim.source_numbers ?? []).map((sourceNumber) => (
                            <span className="source-badge" key={sourceNumber}>
                              [{sourceNumber}]
                            </span>
                          ))}
                        </div>
                        {claim.citations && claim.citations.length > 0 && (
                          <div className="claim-locations">
                            {claim.citations.map((citation, index) => (
                              <CitationLocation
                                compact
                                key={`${citation.source_number ?? 'source'}-${citation.block_id ?? index}`}
                                location={citation}
                              />
                            ))}
                          </div>
                        )}
                      </article>
                    ))}
                  </div>
                ) : (
                  <div className="empty-sources">
                    <ListChecks size={26} />
                    <strong>검증 문장이 없습니다.</strong>
                    <p>답변 문장을 분리하지 못했거나 검색 근거가 부족합니다.</p>
                  </div>
                )}
              </>
            )}

            {sourceTab === 'sources' && (
              <>
                {selectedAnswer.results && selectedAnswer.results.length > 0 ? (
                  <div className="citation-list">
                    {selectedAnswer.results.map((result, index) => {
                      const locationSummary = resultLocationSummary(
                        result,
                        MAX_SOURCE_LOCATIONS,
                      )
                      const fileName =
                        result.source_title ??
                        result.file_name ??
                        result.relative_path ??
                        '제목 없는 문서'
                      const sourcePath =
                        result.relative_path ?? result.source_path ?? ''
                      return (
                        <article className="citation-card" key={result.chunk_id || `${fileName}-${index}`}>
                          <div className="citation-topline">
                            <span>{result.institution ?? '기관 미상'}</span>
                            <strong>
                              [{result.source_number ?? index + 1}] {resultScoreLabel(result)}
                            </strong>
                          </div>
                          <h3>{fileName}</h3>
                          {sourcePath && <p className="location">{sourcePath}</p>}
                          {result.preview && <p>{cleanPreview(result.preview)}</p>}
                          {locationSummary.locations.length > 0 && (
                            <div className="result-locations">
                              {locationSummary.locations.map((location, locationIndex) => (
                                <CitationLocation
                                  compact
                                  key={`${result.chunk_id}-location-${locationIndex}`}
                                  location={location}
                                />
                              ))}
                              {locationSummary.hiddenCount > 0 && (
                                <span className="location-overflow">
                                  외 {locationSummary.hiddenCount.toLocaleString()}개 위치
                                </span>
                              )}
                            </div>
                          )}
                          <dl>
                            <div>
                              <dt>Chunk</dt>
                              <dd>{result.chunk_index ?? '—'}</dd>
                            </div>
                            <div>
                              <dt>Length</dt>
                              <dd>
                                {typeof result.char_count === 'number'
                                  ? `${result.char_count}자`
                                  : '—'}
                              </dd>
                            </div>
                          </dl>
                          {result.source_url && (
                            <a
                              className="source-link"
                              href={result.source_url}
                              rel="noreferrer"
                              target="_blank"
                            >
                              원문 열기
                            </a>
                          )}
                          {result.download_url &&
                            result.download_url !== result.source_url && (
                              <a
                                className="source-link"
                                href={result.download_url}
                                rel="noreferrer"
                                target="_blank"
                              >
                                첨부 열기
                              </a>
                            )}
                        </article>
                      )
                    })}
                  </div>
                ) : (
                  <div className="empty-sources">
                    <ShieldCheck size={26} />
                    <strong>검색된 근거가 없습니다.</strong>
                    <p>기관 범위를 넓히거나 질문 표현을 바꿔 다시 검색해 주세요.</p>
                  </div>
                )}
              </>
            )}

            {sourceTab === 'locations' && (
              <div className="location-list">
                {selectedAnswer.results?.map((result, index) => {
                  const locationSummary = resultLocationSummary(
                    result,
                    MAX_DETAIL_LOCATIONS,
                  )
                  const fileName =
                    result.file_name ?? result.relative_path ?? '제목 없는 문서'
                  return (
                    <article className="location-card" key={result.chunk_id || `${fileName}-${index}`}>
                      <strong>
                        [{result.source_number ?? index + 1}] {fileName}
                      </strong>
                      <span>{result.institution ?? '기관 미상'}</span>
                      {(result.relative_path || result.source_path) && (
                        <p>{result.relative_path ?? result.source_path}</p>
                      )}
                      {locationSummary.locations.length > 0 ? (
                        <div className="location-details">
                          {locationSummary.locations.map((location, locationIndex) => (
                            <CitationLocation
                              key={`${result.chunk_id}-detail-${locationIndex}`}
                              location={location}
                            />
                          ))}
                          {locationSummary.hiddenCount > 0 && (
                            <span className="location-overflow">
                              외 {locationSummary.hiddenCount.toLocaleString()}개 위치
                            </span>
                          )}
                        </div>
                      ) : (
                        <p className="no-location">세부 위치 정보가 제공되지 않았습니다.</p>
                      )}
                      <dl>
                        <div>
                          <dt>Chunk</dt>
                          <dd>{result.chunk_index ?? '—'}</dd>
                        </div>
                        <div>
                          <dt>Score</dt>
                          <dd>{resultScoreLabel(result)}</dd>
                        </div>
                      </dl>
                    </article>
                  )
                })}
                {!selectedAnswer.results?.length && (
                  <div className="empty-sources">
                    <FileText size={26} />
                    <strong>표시할 원문 위치가 없습니다.</strong>
                    <p>검색 결과가 없는 답변입니다.</p>
                  </div>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="empty-sources">
            <Clock3 size={26} />
            <strong>아직 선택된 답변이 없습니다.</strong>
            <p>질문을 보내면 관련 문서와 chunk 미리보기가 여기에 표시됩니다.</p>
          </div>
        )}
      </aside>
    </main>
  )
}

export default App
