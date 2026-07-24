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
  getPipelineStages,
  getProviderCapabilities,
  getResultLocations,
  normalizeGeneration,
  providerDisplayName,
  RagApiError,
} from './api/rag'
import type {
  ChatTrace,
  Claim,
  GenerationInfo,
  GenerationProvider,
  HealthResponse,
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
  error?: {
    code: string
    retryable: boolean
    requestId?: string
  }
  request?: {
    question: string
    institution?: string
    provider: GenerationProvider
  }
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
  if (['ready', 'healthy', 'available', 'configured', 'complete'].includes(normalized)) {
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

function isWeakAnswer(message: Message) {
  if (message.status === 'error') {
    return false
  }

  const claimCount = message.claims?.length ?? 0
  const supportedCount = getSupportedClaimCount(message)
  const sourceCount = message.results?.length ?? 0

  return sourceCount === 0 || (claimCount > 0 && supportedCount < Math.ceil(claimCount * 0.6))
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [institution, setInstitution] = useState(allInstitutions)
  const [institutions, setInstitutions] = useState(defaultInstitutions)
  const [provider, setProvider] = useState<GenerationProvider>('auto')
  const [pendingProvider, setPendingProvider] = useState<GenerationProvider | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [isHealthRefreshing, setIsHealthRefreshing] = useState(true)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null)
  const [sourceTab, setSourceTab] = useState<SourceTab>('claims')
  const [isLoading, setIsLoading] = useState(false)
  const messageStreamRef = useRef<HTMLDivElement | null>(null)
  const activeRequestRef = useRef<AbortController | null>(null)
  const defaultProviderAppliedRef = useRef(false)

  const assistantMessages = messages.filter((message) => message.role === 'assistant')
  const providerCapabilities = useMemo(
    () => getProviderCapabilities(health),
    [health],
  )
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
    setIsHealthRefreshing(true)
    const [healthResult, institutionResult] = await Promise.allSettled([
      getHealth(signal),
      getInstitutions(signal),
    ])
    if (signal?.aborted) {
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
  }, [])

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
    institutionOverride = activeInstitution,
    providerOverride = provider,
    appendUser = true,
  ) {
    const trimmed = question.trim()
    if (!trimmed || activeRequestRef.current || health?.ready !== true) {
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

    const request = {
      question: trimmed,
      institution: institutionOverride,
      provider: providerOverride,
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
    setPendingProvider(providerOverride)

    const controller = new AbortController()
    activeRequestRef.current = controller

    try {
      const data = await requestChat(
        {
          ...request,
          top_k: 8,
        },
        controller.signal,
      )
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
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab('sources')
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null
        setIsLoading(false)
        setPendingProvider(null)
      }
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
            error={healthError}
            health={health}
            onRefresh={() => void refreshStatus()}
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
                disabled={isLoading || health?.ready !== true}
                key={item.label}
                onClick={() => {
                  setInstitution(item.institution)
                  void submitQuestion(item.label, item.institution)
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
            <span className="scope-pill">{institution}</span>
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
                  : health?.ready
                    ? '검색 파이프라인 준비 완료'
                    : '검색 파이프라인 준비 필요'}
              </span>
              <h2>질문을 입력하면 파싱된 공문서 chunk에서 근거 후보를 찾아옵니다</h2>
              <div className="suggestion-grid">
                {availableSuggestedQuestions.map((item) => (
                  <button
                    disabled={isLoading || health?.ready !== true}
                    key={item.question}
                    onClick={() => {
                      setInstitution(item.institution)
                      void submitQuestion(item.question, item.institution)
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
                const requestTrace = traceSummary(message.trace)
                const retrieval = retrievalSummary(message.retrieval)
                const hasFallback = Boolean(message.generation?.fallback_reason)

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
                              : `검증된 근거 ${supportedClaimCount}개`}
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
                          <span>근거 후보 {sourceCount}개</span>
                          <span>검증 문장 {claimCount ? `${supportedClaimCount}/${claimCount}` : '0개'}</span>
                          {message.generation && (
                            <span>
                              {providerDisplayName(message.generation.used)}
                              {message.generation.model ? ` · ${message.generation.model}` : ''}
                            </span>
                          )}
                          {retrieval && <span>{retrieval}</span>}
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

                      {message.role === 'assistant' && isWeakAnswer(message) && (
                        <div className="answer-hint">
                          <AlertTriangle size={15} />
                          <span>근거가 약한 문장이 있을 수 있습니다. 오른쪽 검증 탭에서 확인하거나 기관을 좁혀 다시 질문해 보세요.</span>
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
                                disabled={isLoading || health?.ready !== true}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  void submitQuestion(
                                    message.request?.question ?? '',
                                    message.request?.institution,
                                    message.request?.provider ?? 'auto',
                                    false,
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
                      {providerDisplayName(pendingProvider ?? provider)} 경로로 검색,
                      답변 생성, 인용 검증을 요청했습니다.
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
              disabled={isLoading}
              onChange={setProvider}
              providers={providerCapabilities}
              value={provider}
            />
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
          {health?.ready !== true && (
            <div className="composer-notice" role="status">
              <AlertTriangle size={14} />
              <span>
                {isHealthRefreshing
                  ? '검색 파이프라인 상태를 확인하고 있습니다.'
                  : healthError ?? '검색 파이프라인이 준비되지 않았습니다.'}
              </span>
            </div>
          )}
          <div className="input-row">
            <textarea
              aria-label="질문 입력"
              disabled={isLoading || health?.ready !== true}
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
                health?.ready !== true ||
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
                <strong>{selectedSourceCount}개 후보</strong>
                <span>
                  검증 {selectedSupportedCount}/{selectedClaimCount || 0}
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
                            {claim.supported ? '근거 확인' : '근거 부족'}
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
                        result.file_name ?? result.relative_path ?? '제목 없는 문서'
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
