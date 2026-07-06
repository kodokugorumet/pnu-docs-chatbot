import {
  AlertTriangle,
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Clock3,
  Copy,
  Database,
  FileSearch,
  FileText,
  History,
  Library,
  ListChecks,
  LoaderCircle,
  Menu,
  PanelRightOpen,
  Search,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  X,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Role = 'user' | 'assistant'
type MobilePanel = 'nav' | 'sources' | null
type SourceTab = 'claims' | 'sources' | 'locations'

type SearchResult = {
  source_number: number
  chunk_id: string
  doc_id: string
  chunk_index: number
  institution: string
  file_name: string
  source_path: string
  relative_path: string
  char_count: number
  score: number
  preview: string
}

type Claim = {
  text: string
  supported: boolean
  confidence: number
  source_ids: string[]
  source_numbers: number[]
}

type Message = {
  id: string
  role: Role
  content: string
  results?: SearchResult[]
  claims?: Claim[]
  status?: 'search' | 'error'
}

type ChatResponse = {
  answer: string
  cited_answer?: string
  claims?: Claim[]
  results: SearchResult[]
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
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
  '상장폐지 공시',
  '신탁 수탁고 현황',
  '지급결제 리스크',
  '휴학 신청',
]

const loadingSteps = [
  {
    label: '문서 검색',
    detail: 'BM25 인덱스에서 관련 chunk를 찾고 있습니다.',
  },
  {
    label: '답변 생성',
    detail: '검색된 근거를 바탕으로 Gemini가 초안을 작성하고 있습니다.',
  },
  {
    label: '근거 검증',
    detail: '답변 문장을 문서 후보와 다시 맞춰보고 있습니다.',
  },
]

function makeId() {
  return crypto.randomUUID()
}

function cleanPreview(value: string) {
  return value.replace(/\s+/g, ' ').trim()
}

function scoreLabel(score: number) {
  return Math.abs(score).toFixed(2)
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

function renderWithCitations(text: string) {
  return text.split(/(\[\d+\])/g).map((part, index) =>
    /^\[\d+\]$/.test(part) ? (
      <span className="citation-token" key={`${part}-${index}`}>
        {part}
      </span>
    ) : (
      part
    ),
  )
}

function AnswerContent({ content }: { content: string }) {
  const lines = normalizeAnswerLines(content)

  return (
    <div className="answer-content">
      {lines.map((line, index) =>
        isBulletLine(line) ? (
          <div className="answer-bullet" key={`${line}-${index}`}>
            <span aria-hidden="true" />
            <p>{renderWithCitations(getBulletText(line))}</p>
          </div>
        ) : (
          <p key={`${line}-${index}`}>{renderWithCitations(line)}</p>
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
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null)
  const [sourceTab, setSourceTab] = useState<SourceTab>('claims')
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStepIndex, setLoadingStepIndex] = useState(0)
  const [apiReady, setApiReady] = useState<boolean | null>(null)
  const messageStreamRef = useRef<HTMLDivElement | null>(null)

  const assistantMessages = messages.filter((message) => message.role === 'assistant')
  const selectedAnswer = useMemo(
    () =>
      assistantMessages.find((message) => message.id === selectedMessageId) ??
      assistantMessages.at(-1) ??
      null,
    [assistantMessages, selectedMessageId],
  )

  useEffect(() => {
    async function loadStatus() {
      try {
        const health = await fetch(`${API_BASE_URL}/health`)
        setApiReady(health.ok)

        const response = await fetch(`${API_BASE_URL}/institutions`)
        if (!response.ok) {
          return
        }
        const data = (await response.json()) as { institutions?: string[] }
        if (data.institutions?.length) {
          setInstitutions([allInstitutions, ...data.institutions])
        }
      } catch {
        setApiReady(false)
      }
    }

    void loadStatus()
  }, [])

  useEffect(() => {
    const stream = messageStreamRef.current
    if (!stream) {
      return
    }

    requestAnimationFrame(() => {
      stream.scrollTop = stream.scrollHeight
    })
  }, [isLoading, messages])

  useEffect(() => {
    if (!isLoading) {
      return
    }

    const timer = window.setInterval(() => {
      setLoadingStepIndex((current) => Math.min(current + 1, loadingSteps.length - 1))
    }, 1200)

    return () => window.clearInterval(timer)
  }, [isLoading])

  const activeInstitution = institution === allInstitutions ? undefined : institution

  async function submitQuestion(question: string, institutionOverride = activeInstitution) {
    const trimmed = question.trim()
    if (!trimmed || isLoading) {
      return
    }

    const userMessage: Message = {
      id: makeId(),
      role: 'user',
      content: trimmed,
    }

    setMessages((current) => [...current, userMessage])
    setInput('')
    setLoadingStepIndex(0)
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          institution: institutionOverride,
          top_k: 8,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const data = (await response.json()) as ChatResponse
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'search',
        content: data.cited_answer ?? data.answer,
        claims: data.claims ?? [],
        results: data.results,
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab(answer.claims?.length ? 'claims' : 'sources')
    } catch {
      const answer: Message = {
        id: makeId(),
        role: 'assistant',
        status: 'error',
        content:
          '검색 API에 연결하지 못했습니다. 로컬에서 `python scripts\\search_api.py`를 실행한 뒤 다시 질문해 주세요.',
        results: [],
      }
      setMessages((current) => [...current, answer])
      setSelectedMessageId(answer.id)
      setSourceTab('sources')
    } finally {
      setIsLoading(false)
    }
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
            인덱스 상태
          </div>
          <div className="status-list">
            <span>
              <Database size={15} />
              BM25 SQLite
            </span>
            <span className={apiReady ? 'is-ready' : 'is-offline'}>
              <Server size={15} />
              {apiReady === null ? '확인 중' : apiReady ? 'API 연결됨' : 'API 꺼짐'}
            </span>
          </div>
        </section>

        <section className="rail-section history-section">
          <div className="section-title">
            <History size={15} />
            빠른 검색
          </div>
          <div className="conversation-list">
            {recentQueries.map((query) => (
              <button key={query} onClick={() => void submitQuestion(query)} type="button">
                {query}
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
              <span>검색 인덱스 준비 완료</span>
              <h2>질문을 입력하면 파싱된 공문서 chunk에서 근거 후보를 찾아옵니다</h2>
              <div className="suggestion-grid">
                {suggestedQuestions.map((item) => (
                  <button
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
                              ? '연결 필요'
                              : `검증된 근거 ${supportedClaimCount}개`}
                          </span>
                        </div>
                      )}

                      {message.role === 'assistant' ? <AnswerContent content={message.content} /> : <p>{message.content}</p>}

                      {message.role === 'assistant' && message.status !== 'error' && (
                        <div className="answer-stats">
                          <span>근거 후보 {sourceCount}개</span>
                          <span>검증 문장 {claimCount ? `${supportedClaimCount}/${claimCount}` : '0개'}</span>
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
                        </div>
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
                      <span>{loadingSteps[loadingStepIndex].label} 중</span>
                    </div>
                    <p>{loadingSteps[loadingStepIndex].detail}</p>
                    <ol className="loading-steps">
                      {loadingSteps.map((step, index) => (
                        <li
                          className={
                            index < loadingStepIndex
                              ? 'is-done'
                              : index === loadingStepIndex
                                ? 'is-active'
                                : ''
                          }
                          key={step.label}
                        >
                          {step.label}
                        </li>
                      ))}
                    </ol>
                    <div className="typing-dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </article>
              )}
            </>
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-options">
            <label className="latest-toggle">
              <input checked readOnly type="checkbox" />
              <SlidersHorizontal size={15} />
              <span>BM25 1차 검색</span>
            </label>
            <select
              aria-label="검색 기관"
              onChange={(event) => setInstitution(event.target.value)}
              value={institution}
            >
              {institutions.map((option) => (
                <option key={option}>{option}</option>
              ))}
            </select>
          </div>
          <div className="input-row">
            <textarea
              aria-label="질문 입력"
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
            <button aria-label="질문 보내기" className="send-button" disabled={isLoading} type="submit">
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
                          {claim.source_numbers.map((sourceNumber) => (
                            <span className="source-badge" key={sourceNumber}>
                              [{sourceNumber}]
                            </span>
                          ))}
                        </div>
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
                    {selectedAnswer.results.map((result, index) => (
                      <article className="citation-card" key={result.chunk_id}>
                        <div className="citation-topline">
                          <span>{result.institution}</span>
                          <strong>[{result.source_number ?? index + 1}] {scoreLabel(result.score)}</strong>
                        </div>
                        <h3>{result.file_name}</h3>
                        <p className="location">{result.source_path}</p>
                        <p>{cleanPreview(result.preview)}</p>
                        <dl>
                          <div>
                            <dt>Chunk</dt>
                            <dd>{result.chunk_index}</dd>
                          </div>
                          <div>
                            <dt>Length</dt>
                            <dd>{result.char_count}자</dd>
                          </div>
                        </dl>
                      </article>
                    ))}
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
                {selectedAnswer.results?.map((result) => (
                  <article className="location-card" key={result.chunk_id}>
                    <strong>[{result.source_number}] {result.file_name}</strong>
                    <span>{result.institution}</span>
                    <p>{result.source_path}</p>
                    <dl>
                      <div>
                        <dt>Chunk</dt>
                        <dd>{result.chunk_index}</dd>
                      </div>
                      <div>
                        <dt>Score</dt>
                        <dd>{scoreLabel(result.score)}</dd>
                      </div>
                    </dl>
                  </article>
                ))}
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
