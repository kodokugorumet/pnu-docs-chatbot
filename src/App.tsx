import {
  AlertTriangle,
  BookOpen,
  ArrowDown,
  ArrowUpRight,
  Building2,
  Check,
  ChevronDown,
  GraduationCap,
  Landmark,
  PanelLeftOpen,
  Settings2,
  CheckCircle2,
  Clock3,
  Copy,
  FileSearch,
  FileText,
  Library,
  ListChecks,
  LoaderCircle,
  Menu,
  PanelRightOpen,
  RotateCcw,
  ShieldCheck,
  X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  chat as requestChat,
  getHealth,
  getInstitutions,
  getParserProfileCapabilities,
  getProviderCapabilities,
  getRetrievalModeCapabilities,
  normalizeGeneration,
  parseRoleOptions,
  parserProfileDisplayName,
  providerDisplayName,
  retrievalModeDisplayName,
  RagApiError,
  unloadLocalModel,
} from './api/rag'
import type {
  ChatRequest,
  GenerationProvider,
  HealthResponse,
  ParserProfile,
  RetrievalMode,
  RoleOption,
} from './api/rag'
import CitationLocation from './components/CitationLocation'
import PipelineStatus from './components/PipelineStatus'
import ProviderSelect from './components/ProviderSelect'
import AnswerContent from './components/AnswerContent'
import ChatComposer from './components/ChatComposer'
import ChatSidebar from './components/ChatSidebar'
import Dialog from './components/Dialog'
import useConversations from './hooks/useConversations'
import type { Message } from './types/chat'
import {
  cleanPreview,
  resultScoreLabel,
  resultLocationSummary,
  uniqueDocumentCount,
  retrievalSummary,
  contextDeduplicationSummary,
  claimValidationLabel,
  traceSummary,
  getSupportedClaimCount,
  weakAnswerHint,
  generationModelDisplayName,
  generationFallbackHint,
} from './chat/presentation'
import { positiveIntegerSetting } from './chat/settings'
import './App.css'

type MobilePanel = 'nav' | 'sources' | null
type SourceTab = 'claims' | 'sources' | 'locations'

type SubmitQuestionOptions = {
  role?: string
  institution?: string
  provider?: GenerationProvider
  topK?: number
  model?: string | null
  parserProfile?: ParserProfile
  retrievalMode?: RetrievalMode
  appendUser?: boolean
}

const MAX_QUESTION_CHARS = positiveIntegerSetting(
  import.meta.env.VITE_MAX_QUESTION_CHARS,
  1000,
)
const MAX_ROLE_CHARS = 120
const noRole = 'none'
const customRoleId = 'custom'
// 서버 /health가 프리셋을 내려주지 못할 때의 예비 목록 (role_router와 동일).
const fallbackRoleOptions: RoleOption[] = [
  { id: 'pnu-student', label: '부산대학교 학생' },
  { id: 'pnu-staff', label: '부산대학교 행정직원' },
  { id: 'pnu-researcher', label: '부산대학교 연구자' },
  { id: 'fss-staff', label: '금융감독원 직원' },
  { id: 'bok-staff', label: '한국은행 직원' },
  { id: 'krx-staff', label: '한국거래소 직원' },
  { id: 'ksd-staff', label: '한국예탁결제원 직원' },
  { id: 'kisa-staff', label: '한국인터넷진흥원 직원' },
  { id: 'kiost-staff', label: '한국해양과학기술원 직원' },
]
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
    label: '휴학은 어떻게 신청하나요?',
    institution: '부산대학교',
    category: '학교 생활',
    question: '부산대학교 휴학 신청 절차와 주의할 점을 알려줘',
    icon: GraduationCap,
  },
  {
    label: '상장폐지 제도가 궁금해요',
    institution: '한국거래소',
    category: '제도 이해',
    question: '상장폐지 제도 개선 내용을 핵심만 알려줘',
    icon: Landmark,
  },
  {
    label: '신탁 현황을 요약해 주세요',
    institution: '금융감독원',
    category: '핵심 요약',
    question: '신탁 수탁고 현황을 찾아서 요약해줘',
    icon: FileText,
  },
  {
    label: '지급결제 리스크를 알려줘요',
    institution: '한국은행',
    category: '문서 탐색',
    question: '한국은행 지급결제 리스크 관련 내용을 설명해줘',
    icon: BookOpen,
  },
]

function makeId() {
  return crypto.randomUUID()
}

function App() {
  const {
    messages,
    setMessages,
    conversations,
    activeId,
    selectConversation,
    deleteConversation,
    deleted,
    restoreConversation,
    dismissDeleted,
    storageError,
  } = useConversations()
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [highlightedSource, setHighlightedSource] = useState<number | null>(
    null,
  )
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)
  const [toast, setToast] = useState('')
  const [showScrollButton, setShowScrollButton] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const stickToBottomRef = useRef(true)
  const [input, setInput] = useState('')
  const [institution, setInstitution] = useState(
    () =>
      conversations
        .find((conversation) => conversation.id === activeId)
        ?.messages.findLast((message) => message.request)?.request
        ?.institution ?? allInstitutions,
  )
  const [institutions, setInstitutions] = useState(defaultInstitutions)
  const [roleChoice, setRoleChoice] = useState(noRole)
  const [customRole, setCustomRole] = useState('')
  const [provider, setProvider] = useState<GenerationProvider>('auto')
  const [parserProfile, setParserProfile] = useState<ParserProfile>('cascade')
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('bm25')
  const [localModelPreference, setLocalModelPreference] = useState<
    string | null
  >(null)
  const [frontierModelPreference, setFrontierModelPreference] = useState<
    string | null
  >(null)
  const [topK, setTopK] = useState<EvidenceTopK>(defaultEvidenceTopK)
  const [pendingProvider, setPendingProvider] =
    useState<GenerationProvider | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [isHealthRefreshing, setIsHealthRefreshing] = useState(true)
  const [isLocalModelUnloading, setIsLocalModelUnloading] = useState(false)
  const [localModelUnloadError, setLocalModelUnloadError] = useState<
    string | null
  >(null)
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(
    null,
  )
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null)
  const [sourceTab, setSourceTab] = useState<SourceTab>('claims')
  const [isLoading, setIsLoading] = useState(false)
  const [requestElapsedSeconds, setRequestElapsedSeconds] = useState(0)
  const messageStreamRef = useRef<HTMLDivElement | null>(null)
  const activeRequestRef = useRef<AbortController | null>(null)
  const localModelUnloadRequestRef = useRef(false)
  const healthRefreshSequenceRef = useRef(0)
  const defaultProviderAppliedRef = useRef(false)
  const defaultParserProfileAppliedRef = useRef(false)
  const defaultRetrievalModeAppliedRef = useRef(false)

  const assistantMessages = messages.filter(
    (message) => message.role === 'assistant',
  )
  const roleOptions = useMemo(() => {
    const parsed = parseRoleOptions(health?.roles).filter(
      (option) => option.id !== 'general',
    )
    return parsed.length ? parsed : fallbackRoleOptions
  }, [health])
  const activeRole =
    roleChoice === noRole
      ? undefined
      : roleChoice === customRoleId
        ? customRole.trim().slice(0, MAX_ROLE_CHARS) || undefined
        : roleChoice
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
    health?.ready === true && selectedParserProfileCapability?.ready === true
  const retrievalModeCapabilities = useMemo(
    () => getRetrievalModeCapabilities(health, parserProfile),
    [health, parserProfile],
  )
  const selectedRetrievalModeCapability = useMemo(
    () =>
      retrievalModeCapabilities.find(
        (capability) => capability.id === retrievalMode,
      ),
    [retrievalMode, retrievalModeCapabilities],
  )
  const selectedRetrievalModeReady =
    selectedRetrievalModeCapability?.ready === true
  const localProviderCapability = useMemo(
    () => providerCapabilities.find((capability) => capability.id === 'local'),
    [providerCapabilities],
  )
  const frontierProviderCapability = useMemo(
    () =>
      providerCapabilities.find((capability) => capability.id === 'frontier'),
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
  const selectedFrontierModel = useMemo(() => {
    const models = frontierProviderCapability?.models ?? []
    const preferred = frontierModelPreference
      ? models.find(
          (model) => model.id === frontierModelPreference && model.available,
        )
      : undefined
    if (preferred) {
      return preferred.id
    }

    const configured =
      frontierProviderCapability?.defaultModel ??
      frontierProviderCapability?.model
    const configuredModel = configured
      ? models.find((model) => model.id === configured)
      : undefined
    if (configured && (!configuredModel || configuredModel.available)) {
      return configured
    }
    return models.find((model) => model.available)?.id ?? ''
  }, [frontierModelPreference, frontierProviderCapability])
  const availableSuggestedQuestions = useMemo(
    () =>
      suggestedQuestions.filter((item) =>
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

  const refreshStatus = useCallback(
    async (signal?: AbortSignal) => {
      const sequence = healthRefreshSequenceRef.current + 1
      healthRefreshSequenceRef.current = sequence
      setIsHealthRefreshing(true)
      const [healthResult, institutionResult] = await Promise.allSettled([
        getHealth(signal),
        getInstitutions(parserProfile, signal),
      ])
      if (signal?.aborted || sequence !== healthRefreshSequenceRef.current) {
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
              (capability) => capability.id === current && capability.available,
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
              : (nextParserProfiles.find((capability) => capability.ready)
                  ?.id ?? preferredParserProfile),
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
        const availableInstitutions = [
          allInstitutions,
          ...institutionResult.value.filter(
            (value) => value !== allInstitutions,
          ),
        ]
        setInstitutions(availableInstitutions)
        setInstitution((current) =>
          availableInstitutions.includes(current) ? current : allInstitutions,
        )
      }
      setIsHealthRefreshing(false)
    },
    [parserProfile],
  )

  useEffect(() => {
    if (!health) {
      return
    }
    const preferred =
      selectedParserProfileCapability?.defaultRetrievalMode ??
      health?.default_retrieval_mode ??
      'bm25'
    const available = retrievalModeCapabilities.filter(
      (capability) => capability.ready,
    )
    if (!defaultRetrievalModeAppliedRef.current) {
      setRetrievalMode(
        available.some((capability) => capability.id === preferred)
          ? preferred
          : (available[0]?.id ?? 'bm25'),
      )
      defaultRetrievalModeAppliedRef.current = true
      return
    }
    setRetrievalMode((current) =>
      available.some((capability) => capability.id === current)
        ? current
        : (available.find((capability) => capability.id === 'bm25')?.id ??
          available[0]?.id ??
          'bm25'),
    )
  }, [
    health,
    health?.default_retrieval_mode,
    retrievalModeCapabilities,
    selectedParserProfileCapability?.defaultRetrievalMode,
  ])

  useEffect(() => {
    const controller = new AbortController()
    const initialTimer = window.setTimeout(
      () => void refreshStatus(controller.signal),
      0,
    )
    const timer = window.setInterval(
      () => void refreshStatus(controller.signal),
      30000,
    )
    return () => {
      controller.abort()
      window.clearTimeout(initialTimer)
      window.clearInterval(timer)
    }
  }, [refreshStatus])

  useEffect(
    () => () => {
      activeRequestRef.current?.abort()
      activeRequestRef.current = null
      healthRefreshSequenceRef.current += 1
    },
    [],
  )

  useEffect(() => {
    if (!isLoading) {
      return
    }
    const timer = window.setInterval(
      () => setRequestElapsedSeconds((current) => current + 1),
      1000,
    )
    return () => window.clearInterval(timer)
  }, [isLoading])

  useEffect(() => {
    const stream = messageStreamRef.current
    if (!stream) {
      return
    }

    if (!stickToBottomRef.current) return
    const frame = requestAnimationFrame(() => {
      stream.scrollTop = stream.scrollHeight
    })
    return () => cancelAnimationFrame(frame)
  }, [isLoading, messages])

  useEffect(() => {
    if (mobilePanel !== 'sources' || highlightedSource === null) return
    const frame = requestAnimationFrame(() => {
      const source = document.getElementById(`source-${highlightedSource}`)
      source?.scrollIntoView({ block: 'nearest' })
      source?.focus({ preventScroll: true })
    })
    return () => cancelAnimationFrame(frame)
  }, [mobilePanel, highlightedSource, selectedMessageId])

  useEffect(() => {
    if (!toast && !copiedMessageId) return
    const timer = window.setTimeout(() => {
      setToast('')
      setCopiedMessageId(null)
    }, 2600)
    return () => window.clearTimeout(timer)
  }, [toast, copiedMessageId])

  const selectedInstitution = institutions.includes(institution)
    ? institution
    : allInstitutions
  const activeInstitution =
    selectedInstitution === allInstitutions ? undefined : selectedInstitution

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
    const roleOverride = Object.prototype.hasOwnProperty.call(options, 'role')
      ? options.role
      : activeRole
    const topKOverride = options.topK ?? topK
    const parserProfileOverride = options.parserProfile ?? parserProfile
    const retrievalModeOverride = options.retrievalMode ?? retrievalMode
    const modelOverride = Object.prototype.hasOwnProperty.call(options, 'model')
      ? options.model
      : providerOverride === 'local'
        ? selectedLocalModel
        : providerOverride === 'frontier'
          ? selectedFrontierModel
          : null
    const appendUser = options.appendUser ?? true
    const trimmed = question.trim()
    if (
      !trimmed ||
      activeRequestRef.current ||
      health?.ready !== true ||
      !parserProfileCapabilities.some(
        (capability) =>
          capability.id === parserProfileOverride && capability.ready,
      ) ||
      !getRetrievalModeCapabilities(health, parserProfileOverride).some(
        (capability) =>
          capability.id === retrievalModeOverride && capability.ready,
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
      ...(roleOverride ? { role: roleOverride } : {}),
      provider: providerOverride,
      top_k: topKOverride,
      parser_profile: parserProfileOverride,
      retrieval_mode: retrievalModeOverride,
      ...((providerOverride === 'local' || providerOverride === 'frontier') &&
      normalizedModel
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
    const requestStartedAt = window.performance.now()
    setInput('')
    stickToBottomRef.current = true
    setShowScrollButton(false)
    setRequestElapsedSeconds(0)
    setIsLoading(true)
    if (providerOverride === 'local' || providerOverride === 'auto') {
      setLocalModelUnloadError(null)
    }
    setPendingProvider(providerOverride)

    const controller = new AbortController()
    activeRequestRef.current = controller

    try {
      const data = await requestChat(request, controller.signal)
      if (activeRequestRef.current !== controller) return
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'search',
        content: data.cited_answer ?? data.answer,
        claims: data.claims ?? [],
        results: data.results,
        resolvedRole: data.role,
        generation: normalizeGeneration(data, providerOverride),
        trace: data.trace,
        retrieval: data.retrieval,
        parserProfile: data.parser_profile ?? parserProfileOverride,
        request,
        durationMs: window.performance.now() - requestStartedAt,
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab(answer.claims?.length ? 'claims' : 'sources')
    } catch (error) {
      if (activeRequestRef.current !== controller) return
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
        durationMs: window.performance.now() - requestStartedAt,
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab('sources')
    } finally {
      if (activeRequestRef.current === controller) {
        activeRequestRef.current = null
        setIsLoading(false)
        setPendingProvider(null)
        if (providerOverride === 'local' || providerOverride === 'auto') {
          void refreshStatus()
        }
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

  function openSources(messageId: string, source: number | null = null) {
    setSelectedMessageId(messageId)
    setHighlightedSource(source)
    setSourceTab('sources')
    setMobilePanel('sources')
  }

  function changeConversation(id: string | null) {
    if (activeRequestRef.current) return
    if (id) {
      const lastRequest = conversations
        .find((conversation) => conversation.id === id)
        ?.messages.findLast((message) => message.request)?.request
      const savedInstitution = lastRequest?.institution ?? allInstitutions
      setInstitution(
        institutions.includes(savedInstitution)
          ? savedInstitution
          : allInstitutions,
      )
    }
    selectConversation(id)
    setInput('')
    setSelectedMessageId(null)
    setMobilePanel(null)
    stickToBottomRef.current = true
    setShowScrollButton(false)
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  function openSettings() {
    setMobilePanel(null)
    setIsSettingsOpen(true)
  }

  async function copyAnswer(message: Message) {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopiedMessageId(message.id)
    } catch {
      setToast('복사하지 못했어요. 답변을 선택해서 복사해 주세요.')
    }
  }

  const selectedSupportedCount = selectedAnswer
    ? getSupportedClaimCount(selectedAnswer)
    : 0
  const selectedClaimCount = selectedAnswer?.claims?.length ?? 0
  const selectedSourceCount = selectedAnswer?.results?.length ?? 0
  const selectedDocumentCount = uniqueDocumentCount(selectedAnswer?.results)
  const selectedUnsupportedCount = Math.max(
    selectedClaimCount - selectedSupportedCount,
    0,
  )

  const ready = selectedParserProfileReady && selectedRetrievalModeReady
  const sidebarProps = {
    conversations,
    activeId,
    disabled: isLoading,
    ready,
    checking: isHealthRefreshing && !health,
    onNew: () => changeConversation(null),
    onSelect: changeConversation,
    onDelete: (id: string) => {
      if (activeRequestRef.current) return
      deleteConversation(id)
      setMobilePanel(null)
      if (id === activeId) {
        setSelectedMessageId(null)
        setInput('')
      }
    },
    onSettings: openSettings,
  }

  return (
    <main
      className={`app-shell ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}
    >
      <a
        className="skip-link"
        href="#question-input-area"
        onClick={(event) => {
          event.preventDefault()
          inputRef.current?.focus()
        }}
      >
        질문 입력으로 건너뛰기
      </a>
      <div className="sidebar-desktop">
        <ChatSidebar
          {...sidebarProps}
          onClose={() => setIsSidebarCollapsed(true)}
        />
      </div>
      <Dialog
        className="navigation-dialog"
        labelledBy="mobile-nav-heading"
        onClose={() => setMobilePanel(null)}
        open={mobilePanel === 'nav'}
      >
        <h2 className="sr-only" id="mobile-nav-heading">
          대화 목록
        </h2>
        <ChatSidebar
          {...sidebarProps}
          mobile
          onClose={() => setMobilePanel(null)}
        />
      </Dialog>
      <section className="chat-column">
        <header className="chat-header">
          <div className="header-title-group">
            <button
              aria-label="대화 목록 열기"
              className="icon-button mobile-only"
              onClick={() => setMobilePanel('nav')}
              type="button"
            >
              <Menu size={20} />
            </button>
            {isSidebarCollapsed && (
              <button
                aria-label="사이드바 펼치기"
                className="icon-button desktop-only"
                onClick={() => setIsSidebarCollapsed(false)}
                title="사이드바 펼치기"
                type="button"
              >
                <PanelLeftOpen size={20} />
              </button>
            )}
            <h1>공문서 도우미</h1>
            <button
              aria-label="답변 모델 설정"
              className="model-trigger"
              onClick={openSettings}
              type="button"
            >
              {provider === 'auto'
                ? '자동 선택'
                : provider === 'local'
                  ? '로컬 모델'
                  : 'Gemini'}
              <ChevronDown size={13} />
            </button>
          </div>
          <div className="header-actions">
            <button
              aria-label="근거 문서 열기"
              className="sources-trigger"
              disabled={!selectedAnswer}
              onClick={() => selectedAnswer && openSources(selectedAnswer.id)}
              type="button"
            >
              <PanelRightOpen size={17} />
              <span>근거 문서</span>
            </button>
            <button
              aria-label="설정 열기"
              className="icon-button"
              onClick={openSettings}
              title="설정"
              type="button"
            >
              <Settings2 size={18} />
            </button>
          </div>
        </header>
        <div
          className={`chat-workspace ${messages.length ? 'has-messages' : 'is-empty'}`}
        >
          {messages.length === 0 && (
            <section className="welcome">
              <div className="welcome-symbol">
                <img alt="" src={sanjiniSrc} />
                <span>
                  <BookOpen size={13} />
                  문서에 근거한 답변
                </span>
              </div>
              <h2>
                복잡한 공문서,
                <br />
                <span>쉽게 물어보세요.</span>
              </h2>
              <p>
                학교 규정부터 기관 자료까지,
                <br className="mobile-only" /> 필요한 내용을 출처와 함께
                찾아드려요.
              </p>
            </section>
          )}
          {messages.length > 0 && (
            <div
              role="log"
              aria-label="대화 내용"
              aria-relevant="additions"
              aria-busy={isLoading}
              className="message-stream"
              onScroll={(event) => {
                const stream = event.currentTarget
                const nearBottom =
                  stream.scrollHeight - stream.scrollTop - stream.clientHeight <
                  100
                stickToBottomRef.current = nearBottom
                setShowScrollButton(!nearBottom)
              }}
              ref={messageStreamRef}
            >
              <div className="message-list">
                {messages.map((message) => {
                  const supportedClaimCount = getSupportedClaimCount(message)
                  const claimCount = message.claims?.length ?? 0
                  const sourceCount = message.results?.length ?? 0
                  const documentCount = uniqueDocumentCount(message.results)
                  const requestTrace = traceSummary(message.trace)
                  const retrieval = retrievalSummary(message.retrieval)
                  const deduplication = contextDeduplicationSummary(
                    message.retrieval,
                  )
                  const hasFallback = Boolean(
                    message.generation?.fallback_reason,
                  )
                  const weakAnswer = weakAnswerHint(message)
                  const originalRequest = message.request

                  return (
                    <article
                      className={`message-row ${message.role}`}
                      key={message.id}
                    >
                      {message.role === 'assistant' && (
                        <img className="avatar" src={sanjiniSrc} alt="산지니" />
                      )}
                      <div className="message-bubble">
                        {message.role === 'assistant' && (
                          <div className="answer-meta">
                            {message.status === 'error' ? (
                              <ShieldCheck size={16} />
                            ) : (
                              <CheckCircle2 size={16} />
                            )}
                            <span>
                              {message.status === 'error'
                                ? message.error?.code === 'cancelled'
                                  ? '요청 취소됨'
                                  : '요청 처리 실패'
                                : 'PNU Docs'}
                            </span>
                            {message.status !== 'error' &&
                              documentCount > 0 && (
                                <span className="answer-grounding">
                                  문서 {documentCount}개 참고
                                </span>
                              )}
                            <span className="sr-only">
                              {message.status !== 'error'
                                ? `검증된 문장 ${supportedClaimCount}개`
                                : ''}
                            </span>
                          </div>
                        )}

                        {message.role === 'assistant' ? (
                          <AnswerContent
                            content={message.content}
                            sourceNumbers={(message.results ?? []).map(
                              (result, index) =>
                                result.source_number ?? index + 1,
                            )}
                            onCitationSelect={(source) =>
                              openSources(message.id, source)
                            }
                          />
                        ) : (
                          <p>{message.content}</p>
                        )}

                        {message.role === 'assistant' &&
                          message.status !== 'error' && (
                            <details className="answer-details">
                              <summary>답변 정보</summary>
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
                                {message.request?.retrieval_mode && (
                                  <span>
                                    {retrievalModeDisplayName(
                                      message.request.retrieval_mode,
                                    )}
                                  </span>
                                )}
                                {message.resolvedRole?.requested && (
                                  <span className="role-badge">
                                    역할{' '}
                                    {message.resolvedRole.id === 'general'
                                      ? '일반 사용자 (매핑 안 됨)'
                                      : message.resolvedRole.label}
                                  </span>
                                )}
                                <span>근거 chunk {sourceCount}개</span>
                                <span>고유 문서 {documentCount}개</span>
                                <span>
                                  검증 문장{' '}
                                  {claimCount
                                    ? `${supportedClaimCount}/${claimCount}`
                                    : '0개'}
                                </span>
                                {message.generation && (
                                  <span>
                                    {providerDisplayName(
                                      message.generation.used,
                                    )}
                                    {message.generation.model
                                      ? ` · ${generationModelDisplayName(message.generation.model)}`
                                      : ''}
                                  </span>
                                )}
                                {typeof message.durationMs === 'number' && (
                                  <span>
                                    총 {(message.durationMs / 1000).toFixed(1)}
                                    초
                                  </span>
                                )}
                                {retrieval && <span>{retrieval}</span>}
                                {deduplication && <span>{deduplication}</span>}
                                {requestTrace && <span>{requestTrace}</span>}
                              </div>
                            </details>
                          )}

                        {message.role === 'assistant' && hasFallback && (
                          <div className="answer-hint generation-fallback">
                            <AlertTriangle size={15} />
                            <span>
                              {generationFallbackHint(message.generation)}
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
                                void copyAnswer(message)
                              }}
                              type="button"
                            >
                              {copiedMessageId === message.id ? (
                                <Check size={15} />
                              ) : (
                                <Copy size={15} />
                              )}
                              {copiedMessageId === message.id
                                ? '복사 완료'
                                : '복사'}
                            </button>
                            {message.status !== 'error' && (
                              <button
                                onClick={(event) => {
                                  event.stopPropagation()
                                  openSources(message.id)
                                }}
                                type="button"
                              >
                                <FileSearch size={15} />
                                출처 {documentCount}개
                              </button>
                            )}
                            {(message.status !== 'error' ||
                              message.error?.retryable) &&
                              originalRequest && (
                                <button
                                  disabled={
                                    isLoading ||
                                    !selectedParserProfileReady ||
                                    !selectedRetrievalModeReady
                                  }
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    void submitQuestion(
                                      originalRequest.question,
                                      {
                                        institution:
                                          originalRequest.institution,
                                        role: originalRequest.role,
                                        provider: originalRequest.provider,
                                        topK:
                                          originalRequest.top_k ??
                                          defaultEvidenceTopK,
                                        model: originalRequest.model ?? null,
                                        parserProfile:
                                          originalRequest.parser_profile,
                                        retrievalMode:
                                          originalRequest.retrieval_mode,
                                        appendUser: false,
                                      },
                                    )
                                  }}
                                  type="button"
                                >
                                  <RotateCcw size={15} />
                                  {message.status === 'error'
                                    ? '다시 시도'
                                    : '다시 답변'}
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
                  <article className="message-row assistant is-loading">
                    <img className="avatar" src={sanjiniSrc} alt="" />
                    <div className="message-bubble loading-bubble">
                      <div className="answer-meta">
                        <span>PNU Docs</span>
                        <span className="loading-time">
                          {requestElapsedSeconds}초
                        </span>
                      </div>
                      <div className="request-progress">
                        <LoaderCircle className="loading-icon" size={17} />
                        <span>
                          {requestElapsedSeconds < 20
                            ? '문서를 확인하고 답변을 준비하고 있어요.'
                            : '답변에 시간이 조금 더 걸리고 있어요.'}
                        </span>
                      </div>
                      {requestElapsedSeconds >= 20 && (
                        <p className="loading-explanation">
                          잠시 기다리거나, 아래 중지 버튼을 눌러 다시 질문할 수
                          있어요.
                        </p>
                      )}
                    </div>
                  </article>
                )}
              </div>
            </div>
          )}
          <div className="composer-dock" id="question-input-area">
            {showScrollButton && (
              <button
                aria-label="최신 답변으로 이동"
                className="scroll-bottom-button"
                onClick={() => {
                  const stream = messageStreamRef.current!
                  stream.scrollTo({
                    top: stream.scrollHeight,
                    behavior: window.matchMedia(
                      '(prefers-reduced-motion: reduce)',
                    ).matches
                      ? 'instant'
                      : 'smooth',
                  })
                  stickToBottomRef.current = true
                  setShowScrollButton(false)
                }}
                type="button"
              >
                <ArrowDown size={17} />
              </button>
            )}
            <ChatComposer
              value={input}
              onChange={setInput}
              onSubmit={() => void submitQuestion(input)}
              onCancel={cancelRequest}
              onSettings={openSettings}
              onRefresh={() => void refreshStatus()}
              inputRef={inputRef}
              loading={isLoading}
              ready={ready}
              checking={isHealthRefreshing && !health}
              institution={selectedInstitution}
              institutions={institutions}
              onInstitutionChange={setInstitution}
              maxLength={MAX_QUESTION_CHARS}
              hasMessages={messages.length > 0}
            />
          </div>
          {messages.length === 0 && (
            <section aria-label="추천 질문" className="suggestion-area">
              <div className="suggestion-heading">
                <span>이렇게 질문해 보세요</span>
                <span>질문을 눌러 시작하기</span>
              </div>
              <div className="suggestion-grid">
                {availableSuggestedQuestions.map((item) => (
                  <button
                    className="suggestion-card"
                    key={item.question}
                    onClick={() => {
                      setInstitution(item.institution)
                      setInput(item.question)
                      inputRef.current?.focus()
                    }}
                    type="button"
                  >
                    <span className="suggestion-category">
                      <item.icon size={17} />
                      {item.category}
                    </span>
                    <strong>{item.label}</strong>
                    <span className="suggestion-footer">
                      {item.institution}
                      <ArrowUpRight size={15} />
                    </span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </div>
        {messages.length === 0 && (
          <footer className="workspace-footer">
            <Building2 size={13} />
            부산대학교와 공공기관의 문서를 한곳에서
          </footer>
        )}
        {storageError && (
          <div className="storage-warning" role="status">
            이 브라우저에 대화를 저장할 수 없어요. 필요한 답변은 복사해 보관해
            주세요.
          </div>
        )}
      </section>
      <Dialog
        className="settings-dialog"
        labelledBy="settings-heading"
        onClose={() => setIsSettingsOpen(false)}
        open={isSettingsOpen}
      >
        <div className="dialog-header">
          <div>
            <span className="eyebrow">나에게 맞는 답변</span>
            <h2 id="settings-heading">답변 설정</h2>
          </div>
          <button
            aria-label="설정 닫기"
            className="icon-button"
            onClick={() => setIsSettingsOpen(false)}
            type="button"
          >
            <X size={20} />
          </button>
        </div>
        <div className="settings-body">
          <p className="settings-intro">
            기본 설정으로 바로 질문할 수 있어요. 필요할 때 원하는 방식으로 바꿔
            보세요.
          </p>
          <div className="institution-control role-control">
            <label htmlFor="role-choice">
              내 역할 <span className="optional-label">선택 사항</span>
            </label>
            <select
              aria-label="내 역할 선택"
              id="role-choice"
              disabled={isLoading}
              onChange={(event) => setRoleChoice(event.target.value)}
              value={roleChoice}
            >
              <option value={noRole}>역할 없음</option>
              {roleOptions.map((option) => (
                <option key={option.id} value={option.id}>
                  {option.label}
                </option>
              ))}
              <option value={customRoleId}>직접 입력…</option>
            </select>
            {roleChoice === customRoleId ? (
              <input
                aria-label="역할 직접 입력"
                disabled={isLoading}
                maxLength={MAX_ROLE_CHARS}
                onChange={(event) => setCustomRole(event.target.value)}
                placeholder="예: 부산대 대학원생, 금감원 직원"
                type="text"
                value={customRole}
              />
            ) : null}
            <span>
              {roleChoice === noRole
                ? '역할을 알려주면 그 역할에 맞는 기관 문서를 우선합니다.'
                : '역할에 맞는 기관 문서를 우선하되 다른 기관 문서도 검색합니다.'}
            </span>
          </div>

          <div className="composer-options">
            <ProviderSelect
              activeRequestProvider={isLoading ? pendingProvider : null}
              disabled={isLoading}
              frontierModel={selectedFrontierModel}
              model={selectedLocalModel}
              onChange={setProvider}
              onFrontierModelChange={setFrontierModelPreference}
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
                  : (selectedParserProfileCapability?.reason ??
                    '선택한 파서 인덱스를 확인할 수 없습니다.')}
              </small>
            </label>
            <label className="retrieval-mode-select">
              <span>검색 방식</span>
              <select
                aria-label="검색 방식"
                disabled={isLoading}
                onChange={(event) =>
                  setRetrievalMode(event.target.value as RetrievalMode)
                }
                value={retrievalMode}
              >
                {retrievalModeCapabilities.map((capability) => (
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
                {selectedRetrievalModeCapability?.ready
                  ? selectedRetrievalModeCapability.id === 'bm25'
                    ? '키워드 일치 기반 비교 기준선'
                    : `${selectedRetrievalModeCapability.dimensions ?? '-'}차원 · ${
                        selectedRetrievalModeCapability.modelLoaded
                          ? `${selectedRetrievalModeCapability.device ?? 'device'} 로드됨`
                          : '최초 검색 시 모델 로드'
                      }`
                  : (selectedRetrievalModeCapability?.reason ??
                    '선택한 검색 인덱스를 확인할 수 없습니다.')}
              </small>
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

          <details className="service-details">
            <summary>
              <Library size={16} />
              검색 서비스 연결 상태
            </summary>
            <PipelineStatus
              chunkCount={selectedParserProfileCapability?.chunkCount}
              error={healthError}
              health={health}
              onRefresh={() => void refreshStatus()}
              profileLabel={parserProfileDisplayName(parserProfile, true)}
              profileReady={selectedParserProfileCapability?.ready}
              refreshing={isHealthRefreshing}
            />
          </details>
        </div>
        <div className="dialog-footer">
          <span>변경한 설정은 다음 질문부터 적용돼요.</span>
          <button
            className="primary-button"
            onClick={() => setIsSettingsOpen(false)}
            type="button"
          >
            완료
          </button>
        </div>
      </Dialog>
      <Dialog
        className="source-dialog"
        labelledBy="source-heading"
        onClose={() => setMobilePanel(null)}
        open={mobilePanel === 'sources'}
      >
        <div className="source-panel">
          <div className="source-panel-header">
            <div>
              <span className="eyebrow">답변의 출처를 직접 확인하세요</span>
              <h2 id="source-heading">근거 문서</h2>
            </div>
            <button
              aria-label="근거 패널 닫기"
              className="icon-button"
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
                  <strong>참고한 근거 {selectedSourceCount}개</strong>
                  <span>
                    문서 {selectedDocumentCount}개 · 문장 검증{' '}
                    {selectedSupportedCount}/{selectedClaimCount || 0}
                    {selectedUnsupportedCount > 0
                      ? `, 근거 부족 ${selectedUnsupportedCount}`
                      : ''}
                  </span>
                </div>
              </div>

              <div
                className="source-tabs"
                role="tablist"
                aria-label="근거 보기 방식"
                onKeyDown={(event) => {
                  const tabs: SourceTab[] = ['claims', 'sources', 'locations']
                  const current = tabs.indexOf(sourceTab)
                  const next =
                    event.key === 'ArrowRight'
                      ? (current + 1) % 3
                      : event.key === 'ArrowLeft'
                        ? (current + 2) % 3
                        : event.key === 'Home'
                          ? 0
                          : event.key === 'End'
                            ? 2
                            : -1
                  if (next < 0) return
                  event.preventDefault()
                  setSourceTab(tabs[next])
                  const buttons =
                    event.currentTarget.querySelectorAll<HTMLButtonElement>(
                      '[role="tab"]',
                    )
                  buttons[next]?.focus()
                }}
              >
                <button
                  aria-selected={sourceTab === 'claims'}
                  aria-controls="source-tab-content"
                  id="source-tab-claims"
                  tabIndex={sourceTab === 'claims' ? 0 : -1}
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
                  aria-controls="source-tab-content"
                  id="source-tab-sources"
                  tabIndex={sourceTab === 'sources' ? 0 : -1}
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
                  aria-controls="source-tab-content"
                  id="source-tab-locations"
                  tabIndex={sourceTab === 'locations' ? 0 : -1}
                  className={sourceTab === 'locations' ? 'is-active' : ''}
                  onClick={() => setSourceTab('locations')}
                  role="tab"
                  type="button"
                >
                  <FileText size={15} />
                  위치
                </button>
              </div>

              <div
                role="tabpanel"
                id="source-tab-content"
                aria-labelledby={`source-tab-${sourceTab}`}
              >
                {sourceTab === 'claims' && (
                  <>
                    {selectedAnswer.claims &&
                    selectedAnswer.claims.length > 0 ? (
                      <div className="claim-list">
                        {selectedAnswer.claims.map((claim) => (
                          <article className="claim-card" key={claim.text}>
                            <p>{claim.text}</p>
                            <div className="claim-meta">
                              <span
                                className={
                                  claim.supported
                                    ? 'is-supported'
                                    : 'is-unsupported'
                                }
                              >
                                {claimValidationLabel(claim)}
                              </span>
                              {(claim.source_numbers ?? []).map(
                                (sourceNumber) => (
                                  <span
                                    className="source-badge"
                                    key={sourceNumber}
                                  >
                                    [{sourceNumber}]
                                  </span>
                                ),
                              )}
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
                        <p>
                          답변 문장을 분리하지 못했거나 검색 근거가 부족합니다.
                        </p>
                      </div>
                    )}
                  </>
                )}

                {sourceTab === 'sources' && (
                  <>
                    {selectedAnswer.results &&
                    selectedAnswer.results.length > 0 ? (
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
                            <article
                              className={`citation-card ${highlightedSource === (result.source_number ?? index + 1) ? 'is-highlighted' : ''}`}
                              id={`source-${result.source_number ?? index + 1}`}
                              tabIndex={-1}
                              key={result.chunk_id || `${fileName}-${index}`}
                            >
                              <div className="citation-topline">
                                <span>{result.institution ?? '기관 미상'}</span>
                                <strong>
                                  출처 {result.source_number ?? index + 1}
                                </strong>
                              </div>
                              <h3>{fileName}</h3>
                              {sourcePath && (
                                <p className="location">{sourcePath}</p>
                              )}
                              {result.preview && (
                                <p>{cleanPreview(result.preview)}</p>
                              )}
                              {locationSummary.locations.length > 0 && (
                                <div className="result-locations">
                                  {locationSummary.locations.map(
                                    (location, locationIndex) => (
                                      <CitationLocation
                                        compact
                                        key={`${result.chunk_id}-location-${locationIndex}`}
                                        location={location}
                                      />
                                    ),
                                  )}
                                  {locationSummary.hiddenCount > 0 && (
                                    <span className="location-overflow">
                                      외{' '}
                                      {locationSummary.hiddenCount.toLocaleString()}
                                      개 위치
                                    </span>
                                  )}
                                </div>
                              )}
                              <dl>
                                <div>
                                  <dt>문단</dt>
                                  <dd>{result.chunk_index ?? '—'}</dd>
                                </div>
                                <div>
                                  <dt>분량</dt>
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
                        <p>
                          기관 범위를 넓히거나 질문 표현을 바꿔 다시 검색해
                          주세요.
                        </p>
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
                        result.source_title ??
                        result.file_name ??
                        result.relative_path ??
                        '제목 없는 문서'
                      return (
                        <article
                          className="location-card"
                          key={result.chunk_id || `${fileName}-${index}`}
                        >
                          <strong>
                            [{result.source_number ?? index + 1}] {fileName}
                          </strong>
                          <span>{result.institution ?? '기관 미상'}</span>
                          {(result.relative_path || result.source_path) && (
                            <p>{result.relative_path ?? result.source_path}</p>
                          )}
                          {locationSummary.locations.length > 0 ? (
                            <div className="location-details">
                              {locationSummary.locations.map(
                                (location, locationIndex) => (
                                  <CitationLocation
                                    key={`${result.chunk_id}-detail-${locationIndex}`}
                                    location={location}
                                  />
                                ),
                              )}
                              {locationSummary.hiddenCount > 0 && (
                                <span className="location-overflow">
                                  외{' '}
                                  {locationSummary.hiddenCount.toLocaleString()}
                                  개 위치
                                </span>
                              )}
                            </div>
                          ) : (
                            <p className="no-location">
                              세부 위치 정보가 제공되지 않았습니다.
                            </p>
                          )}
                          <dl>
                            <div>
                              <dt>문단</dt>
                              <dd>{result.chunk_index ?? '—'}</dd>
                            </div>
                            <div>
                              <dt>관련도</dt>
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
              </div>
            </>
          ) : (
            <div className="empty-sources">
              <Clock3 size={26} />
              <strong>아직 선택된 답변이 없습니다.</strong>
              <p>
                질문을 보내면 답변에 참고한 문서와 원문 위치를 확인할 수 있어요.
              </p>
            </div>
          )}
        </div>
      </Dialog>

      {(toast || deleted) && (
        <div className="toast" role="status">
          <span>{toast || '대화를 삭제했어요.'}</span>
          {deleted && !toast && (
            <button
              disabled={isLoading}
              onClick={() => {
                restoreConversation()
                setSelectedMessageId(null)
                stickToBottomRef.current = true
              }}
              type="button"
            >
              되돌리기
            </button>
          )}
          <button
            aria-label="알림 닫기"
            onClick={() => {
              setToast('')
              dismissDeleted()
            }}
            type="button"
          >
            <X size={14} />
          </button>
        </div>
      )}
    </main>
  )
}

export default App
