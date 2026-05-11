import {
  ArrowUp,
  BookOpen,
  CheckCircle2,
  Clock3,
  Copy,
  ExternalLink,
  FileSearch,
  History,
  Library,
  Menu,
  PanelRightOpen,
  Search,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Role = 'user' | 'assistant'

type Citation = {
  id: string
  title: string
  type: string
  location: string
  revisedDate: string
  effectiveDate: string
  quote: string
  confidence: number
}

type Message = {
  id: string
  role: Role
  content: string
  conditions?: string[]
  citations?: Citation[]
  confidence?: number
  safety?: boolean
}

type MobilePanel = 'nav' | 'sources' | null

const sanjiniSrc = '/sanjini.webp'

const scopeOptions = ['전체', '부산대 규정', '학사 행정', '공공기관 문서']
const docTypes = ['규정', '지침', '공지', '행정 서식', 'FAQ']

const suggestedQuestions = [
  '휴학 신청 기준 알려줘',
  '장학금 지급 제한 규정은?',
  '교원 복무 관련 최신 규정 찾아줘',
  '학사 행정 서식은 어디서 확인해?',
]

const conversations = [
  '휴학 신청 기준',
  '장학금 지급 제한',
  '교원 복무 규정',
  '행정 서식 위치',
]

const citationLibrary: Record<string, Citation[]> = {
  leave: [
    {
      id: 'leave-1',
      title: '부산대학교 학칙',
      type: '규정',
      location: '제47조 휴학',
      revisedDate: '2026.03.01',
      effectiveDate: '2026.03.01',
      quote: '일반휴학, 질병휴학, 군휴학 등 사유별 신청 요건을 확인해야 합니다.',
      confidence: 94,
    },
    {
      id: 'leave-2',
      title: '학사운영 안내',
      type: '공지',
      location: '휴학 및 복학 절차',
      revisedDate: '2026.02.15',
      effectiveDate: '2026.03.01',
      quote: '학기 개시 전후 신청 기간과 증빙 서류 제출 여부가 달라질 수 있습니다.',
      confidence: 89,
    },
  ],
  scholarship: [
    {
      id: 'scholarship-1',
      title: '장학금 지급 규정',
      type: '규정',
      location: '제9조 지급 제한',
      revisedDate: '2025.12.20',
      effectiveDate: '2026.01.01',
      quote: '학사경고, 징계, 등록 요건 미충족 등은 장학금 지급 제한 사유가 될 수 있습니다.',
      confidence: 92,
    },
    {
      id: 'scholarship-2',
      title: '학생지원 업무 처리 지침',
      type: '지침',
      location: '장학 심사 기준',
      revisedDate: '2026.01.18',
      effectiveDate: '2026.01.18',
      quote: '성적 기준과 소득 구간, 중복 수혜 여부를 함께 검토합니다.',
      confidence: 87,
    },
  ],
  service: [
    {
      id: 'service-1',
      title: '교원 복무 규정',
      type: '규정',
      location: '제12조 근무 및 출장',
      revisedDate: '2026.02.28',
      effectiveDate: '2026.03.01',
      quote: '출장, 연가, 병가 등 복무 사항은 사전 승인과 증빙 절차를 따릅니다.',
      confidence: 90,
    },
    {
      id: 'service-2',
      title: '공무원 복무 관련 행정 기준',
      type: '공공기관 문서',
      location: '복무 관리 일반',
      revisedDate: '2026.01.05',
      effectiveDate: '2026.01.05',
      quote: '상위 행정 기준과 대학 내부 규정을 함께 확인해야 합니다.',
      confidence: 83,
    },
  ],
  form: [
    {
      id: 'form-1',
      title: '학사 행정 서식 모음',
      type: '행정 서식',
      location: '휴복학/제증명 서식',
      revisedDate: '2026.04.02',
      effectiveDate: '2026.04.02',
      quote: '민원 성격에 따라 학과 사무실 또는 학사과 제출 서식이 다릅니다.',
      confidence: 88,
    },
  ],
}

const fallbackCitations: Citation[] = [
  {
    id: 'fallback-1',
    title: '부산대학교 규정집',
    type: '규정',
    location: '통합 검색 결과',
    revisedDate: '2026.03.01',
    effectiveDate: '2026.03.01',
    quote: '질문과 가장 가까운 규정 후보를 찾았지만 세부 조항 확인이 필요합니다.',
    confidence: 76,
  },
]

function getCitationSet(question: string) {
  const normalized = question.toLowerCase()

  if (question.includes('휴학') || question.includes('복학')) {
    return citationLibrary.leave
  }

  if (question.includes('장학') || question.includes('지급')) {
    return citationLibrary.scholarship
  }

  if (question.includes('교원') || question.includes('복무') || question.includes('출장')) {
    return citationLibrary.service
  }

  if (question.includes('서식') || question.includes('양식') || normalized.includes('form')) {
    return citationLibrary.form
  }

  return fallbackCitations
}

function isAdversarial(question: string) {
  const lowered = question.toLowerCase()
  return [
    '시스템 프롬프트',
    '프롬프트 공개',
    '이전 지시',
    '무시하고',
    '탈옥',
    'jailbreak',
    'ignore previous',
    'developer message',
  ].some((keyword) => lowered.includes(keyword.toLowerCase()))
}

function buildMockAnswer(question: string): Message {
  if (isAdversarial(question)) {
    return {
      id: crypto.randomUUID(),
      role: 'assistant',
      safety: true,
      confidence: 98,
      content:
        '요청하신 내용은 문서 검색 범위를 벗어나거나 시스템 지시를 변경하려는 요청으로 보여 답변할 수 없습니다. 부산대 규정, 학사 행정, 공공기관 문서에 관한 질문으로 다시 물어봐 주세요.',
      conditions: [
        '내부 지시나 보안 정책은 공개하지 않습니다.',
        '검색 대상 문서에 근거한 행정 정보만 답변합니다.',
      ],
      citations: [],
    }
  }

  const citations = getCitationSet(question)
  const topCitation = citations[0]

  return {
    id: crypto.randomUUID(),
    role: 'assistant',
    confidence: topCitation.confidence,
    content: `${topCitation.title}의 ${topCitation.location}을 기준으로 보면, 질문하신 사안은 신청 사유와 제출 시점, 증빙 서류 여부를 함께 확인해야 합니다. 최종 처리는 소속 학과나 담당 부서의 최신 공지와 원문 규정을 함께 확인하는 방식이 안전합니다.`,
    conditions: [
      '개정일이 더 최신인 문서가 있으면 최신 문서를 우선합니다.',
      '개인별 학적 상태나 소속 기관에 따라 적용 기준이 달라질 수 있습니다.',
      '근거가 부족한 경우에는 답변 대신 추가 확인이 필요하다고 표시합니다.',
    ],
    citations,
  }
}

function confidenceLabel(score = 0) {
  if (score >= 90) {
    return '높음'
  }

  if (score >= 80) {
    return '보통'
  }

  return '확인 필요'
}

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [scope, setScope] = useState(scopeOptions[0])
  const [latestOnly, setLatestOnly] = useState(true)
  const [activeTypes, setActiveTypes] = useState<string[]>(['규정', '지침', '공지'])
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null)
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>(null)

  const assistantMessages = messages.filter((message) => message.role === 'assistant')

  const selectedAnswer = useMemo(() => {
    return (
      assistantMessages.find((message) => message.id === selectedMessageId) ??
      assistantMessages.at(-1) ??
      null
    )
  }, [assistantMessages, selectedMessageId])

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const question = input.trim()

    if (!question) {
      return
    }

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: question,
    }
    const answer = buildMockAnswer(question)

    setMessages((current) => [...current, userMessage, answer])
    setSelectedMessageId(answer.id)
    setInput('')
  }

  const submitSuggestedQuestion = (question: string) => {
    const answer = buildMockAnswer(question)
    setMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: question,
      },
      answer,
    ])
    setSelectedMessageId(answer.id)
  }

  const toggleDocType = (docType: string) => {
    setActiveTypes((current) =>
      current.includes(docType)
        ? current.filter((item) => item !== docType)
        : [...current, docType],
    )
  }

  return (
    <main className="app-shell">
      <aside className={`left-rail ${mobilePanel === 'nav' ? 'is-open' : ''}`}>
        <div className="brand">
          <img src={sanjiniSrc} alt="부산대학교 마스코트 산지니" />
          <div>
            <strong>산지니 문서봇</strong>
            <span>부산대 규정 검색</span>
          </div>
        </div>

        <button className="new-chat-button" type="button">
          <Sparkles size={18} />
          새 질문
        </button>

        <section className="rail-section">
          <div className="section-title">
            <Search size={15} />
            검색 범위
          </div>
          <div className="segmented-list">
            {scopeOptions.map((option) => (
              <button
                className={scope === option ? 'is-active' : ''}
                key={option}
                onClick={() => setScope(option)}
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
            문서 유형
          </div>
          <div className="filter-list">
            {docTypes.map((docType) => (
              <label key={docType}>
                <input
                  checked={activeTypes.includes(docType)}
                  onChange={() => toggleDocType(docType)}
                  type="checkbox"
                />
                <span>{docType}</span>
              </label>
            ))}
          </div>
        </section>

        <section className="rail-section history-section">
          <div className="section-title">
            <History size={15} />
            최근 대화
          </div>
          <div className="conversation-list">
            {conversations.map((conversation) => (
              <button key={conversation} type="button">
                {conversation}
              </button>
            ))}
          </div>
        </section>
      </aside>

      {mobilePanel && (
        <button
          aria-label="모바일 패널 닫기"
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
            <h1>부산대 규정과 행정 문서를 물어보세요</h1>
          </div>

          <div className="header-actions">
            <span className="scope-pill">{scope}</span>
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

        <div className="message-stream">
          {messages.length === 0 ? (
            <section className="empty-state">
              <img src={sanjiniSrc} alt="" />
              <span>안녕하세요, 산지니예요.</span>
              <h2>필요한 규정과 행정 문서를 바로 찾아볼게요.</h2>
              <div className="suggestion-grid">
                {suggestedQuestions.map((question) => (
                  <button
                    key={question}
                    onClick={() => submitSuggestedQuestion(question)}
                    type="button"
                  >
                    <BookOpen size={17} />
                    {question}
                  </button>
                ))}
              </div>
            </section>
          ) : (
            messages.map((message) => (
              <article
                className={`message-row ${message.role}`}
                key={message.id}
                onClick={() => {
                  if (message.role === 'assistant') {
                    setSelectedMessageId(message.id)
                  }
                }}
              >
                {message.role === 'assistant' && (
                  <img className="avatar" src={sanjiniSrc} alt="산지니" />
                )}
                <div className="message-bubble">
                  {message.role === 'assistant' && (
                    <div className="answer-meta">
                      {message.safety ? <ShieldCheck size={16} /> : <CheckCircle2 size={16} />}
                      <span>
                        {message.safety
                          ? '보안 응답'
                          : `신뢰도 ${confidenceLabel(message.confidence)} ${message.confidence}%`}
                      </span>
                    </div>
                  )}
                  <p>{message.content}</p>
                  {message.conditions && (
                    <ul className="condition-list">
                      {message.conditions.map((condition) => (
                        <li key={condition}>{condition}</li>
                      ))}
                    </ul>
                  )}
                  {message.role === 'assistant' && (
                    <div className="message-actions">
                      <button type="button">
                        <Copy size={15} />
                        복사
                      </button>
                      <button type="button">
                        <FileSearch size={15} />
                        근거 {message.citations?.length ?? 0}개
                      </button>
                    </div>
                  )}
                </div>
              </article>
            ))
          )}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-options">
            <label className="latest-toggle">
              <input
                checked={latestOnly}
                onChange={(event) => setLatestOnly(event.target.checked)}
                type="checkbox"
              />
              <span>최신 문서 우선</span>
            </label>
            <select
              aria-label="검색 범위"
              onChange={(event) => setScope(event.target.value)}
              value={scope}
            >
              {scopeOptions.map((option) => (
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
              placeholder="예: 휴학 신청 기준 알려줘"
              rows={1}
              value={input}
            />
            <button aria-label="질문 보내기" className="send-button" type="submit">
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
                <strong>{confidenceLabel(selectedAnswer.confidence)}</strong>
                <span>
                  {selectedAnswer.safety
                    ? '문서 검색 범위 밖 요청입니다.'
                    : `답변 신뢰도 ${selectedAnswer.confidence}%`}
                </span>
              </div>
            </div>

            {selectedAnswer.citations && selectedAnswer.citations.length > 0 ? (
              <div className="citation-list">
                {selectedAnswer.citations.map((citation) => (
                  <article className="citation-card" key={citation.id}>
                    <div className="citation-topline">
                      <span>{citation.type}</span>
                      <strong>{citation.confidence}%</strong>
                    </div>
                    <h3>{citation.title}</h3>
                    <p className="location">{citation.location}</p>
                    <p>{citation.quote}</p>
                    <dl>
                      <div>
                        <dt>개정일</dt>
                        <dd>{citation.revisedDate}</dd>
                      </div>
                      <div>
                        <dt>시행일</dt>
                        <dd>{citation.effectiveDate}</dd>
                      </div>
                    </dl>
                    <button type="button">
                      원문 보기
                      <ExternalLink size={15} />
                    </button>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-sources">
                <ShieldCheck size={26} />
                <strong>근거 문서를 제공하지 않았습니다.</strong>
                <p>내부 지시나 보안 정보를 요구하는 질문은 문서 검색으로 처리하지 않습니다.</p>
              </div>
            )}
          </>
        ) : (
          <div className="empty-sources">
            <Clock3 size={26} />
            <strong>아직 선택된 답변이 없습니다.</strong>
            <p>질문을 보내면 관련 문서와 조항이 여기에 표시됩니다.</p>
          </div>
        )}
      </aside>
    </main>
  )
}

export default App
