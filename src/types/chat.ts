import type {
  ChatRequest,
  ChatTrace,
  Claim,
  GenerationInfo,
  ParserProfile,
  ResolvedRole,
  SearchResult,
} from '../api/rag'

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  results?: SearchResult[]
  claims?: Claim[]
  status?: 'search' | 'error'
  generation?: GenerationInfo
  resolvedRole?: ResolvedRole
  trace?: ChatTrace
  retrieval?: Record<string, unknown>
  parserProfile?: ParserProfile
  error?: { code: string; retryable: boolean; requestId?: string }
  request?: ChatRequest
  durationMs?: number
}

export type Conversation = {
  id: string
  title: string
  updatedAt: number
  messages: Message[]
}
