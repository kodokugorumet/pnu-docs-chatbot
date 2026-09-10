import type {
  ChatResponse,
  HealthResponse,
  SearchResult,
} from '../../src/api/rag'
import type { Message } from '../../src/types/chat'

export function health(
  overrides: Partial<HealthResponse> = {},
): HealthResponse {
  const modes = [
    { id: 'bm25', ready: true },
    {
      id: 'kure_hybrid',
      ready: true,
      dimensions: 1024,
      model_loaded: true,
      device: 'cpu',
    },
    { id: 'snowflake_hybrid', ready: true },
  ]
  return {
    ready: true,
    default_provider: 'auto',
    default_parser_profile: 'cascade',
    default_retrieval_mode: 'bm25',
    roles: [
      { id: 'general', label: '일반' },
      { id: 'pnu-student', label: '부산대학교 학생' },
    ],
    parser_profiles: ['baseline', 'challenger', 'cascade'].map((id) => ({
      id,
      ready: true,
      document_count: 10,
      chunk_count: 100,
      default_retrieval_mode: 'bm25',
      retrieval_modes: modes,
    })),
    retrieval_modes: modes,
    providers: ['local', 'frontier'].map((id) => ({
      id,
      available: true,
      default_model: `${id}-a`,
      model: `${id}-a`,
      models: [
        { id: `${id}-a`, label: `${id} A`, available: true },
        { id: `${id}-b`, label: `${id} B`, available: true },
      ],
      runtime_state: 'loaded',
      loaded_model: `${id}-a`,
      worker_running: true,
      unload_supported: true,
    })),
    ...overrides,
  }
}
export const sources: SearchResult[] = [
  {
    chunk_id: '1',
    source_number: 1,
    source_title: '학적 안내',
    file_name: '학적.pdf',
    institution: '부산대학교',
    relative_path: 'docs/a',
    source_path: '/docs/a',
    document_id: 'a',
    preview: '신청 절차와 근거를 확인하세요.',
    score: 3,
    char_count: 100,
    chunk_index: 0,
    source_url: 'https://example.org/a',
    download_url: 'https://example.org/a.pdf',
    locations: Array.from({ length: 10 }, (_, index) => ({
      page: index + 1,
      block_id: `block-${index}`,
    })),
    location_count: 12,
  },
  {
    chunk_id: '2',
    source_number: 2,
    file_name: '서류 안내',
    institution: '부산대학교',
    doc_id: 'b',
    preview: '제출 서류 안내',
    source_url: 'https://example.org/b',
    download_url: 'https://example.org/b',
  },
]
export function response(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    answer:
      '## 신청 안내\n\n**신청 서류**를 확인하세요. [1]\n\n- 제출 서류 [2]',
    results: sources,
    claims: [
      {
        text: '신청 서류를 확인하세요.',
        supported: true,
        source_numbers: [1],
        citations: [{ source_number: 1, page: 1, block_id: 'block-1' }],
      },
      {
        text: '지원 대상은 확인이 필요합니다.',
        supported: false,
        source_numbers: [2],
        validation_reason: 'low_lexical_overlap',
      },
    ],
    generation: { requested: 'auto', used: 'local', model: 'local-a' },
    parser_profile: 'cascade',
    role: {
      requested: 'pnu-student',
      id: 'pnu-student',
      label: '부산대학교 학생',
    },
    retrieval: {
      strategy: 'bm25',
      context_deduplication: { removed_count: 1, kept_count: 2 },
    },
    trace: { stages: [{ id: 'search', state: 'ready', duration_ms: 10 }] },
    ...overrides,
  }
}
export function storedAnswer(overrides: Partial<Message> = {}): Message {
  const data = response()
  return {
    id: 'answer',
    role: 'assistant',
    content: data.answer,
    results: data.results,
    claims: data.claims,
    request: {
      question: '휴학 질문',
      provider: 'auto',
      parser_profile: 'cascade',
      retrieval_mode: 'bm25',
      institution: '부산대학교',
    },
    status: 'search',
    ...overrides,
  }
}
export function seed(messages: Message[], activeId: string | null = 'saved') {
  localStorage.setItem(
    'pnu-docs.conversations.v1',
    JSON.stringify({
      activeId,
      conversations: [
        { id: 'saved', title: '저장된 대화', updatedAt: 1, messages },
      ],
    }),
  )
}
export function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
