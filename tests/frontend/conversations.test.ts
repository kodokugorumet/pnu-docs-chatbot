import assert from 'node:assert/strict'
import { test } from 'vitest'
import {
  conversationReducer,
  emptyConversationState,
  MAX_CONVERSATIONS,
  parseConversations,
} from '../../src/state/conversations.ts'
import type { ConversationState } from '../../src/state/conversations.ts'
import type { Message } from '../../src/types/chat.ts'

const question: Message = {
  id: 'question-1',
  role: 'user',
  content: '휴학 신청 절차를 알려줘',
}
const answer: Message = {
  id: 'answer-1',
  role: 'assistant',
  content: '서류를 확인하세요. [2]',
  results: [
    {
      chunk_id: 'source-2',
      source_number: 2,
      source_title: '학적 안내',
      page: 5,
    },
  ],
  request: {
    question: question.content,
    provider: 'auto',
    parser_profile: 'cascade',
    retrieval_mode: 'bm25',
    role: 'pnu-student',
  },
}
const append = (
  state: ConversationState,
  messages: Message[],
  id: string,
  now = 1,
) =>
  conversationReducer(state, {
    type: 'messages',
    id,
    now,
    update: (current) => [...current, ...messages],
  })

// These sequences exercise persistence and navigation contracts, including API replies arriving after submission.
test('a pending question and its reply stay in one conversation, including after reload', () => {
  const pending = append(emptyConversationState, [question], 'chat-1')
  const completed = append(pending, [answer], 'unused-id', 2)
  const restored = parseConversations(JSON.stringify(completed))
  assert.equal(restored.activeId, 'chat-1')
  assert.equal(restored.conversations.length, 1)
  assert.deepEqual(restored.conversations[0].messages, [question, answer])
  assert.equal(
    restored.conversations[0].messages[1].results?.[0].source_number,
    2,
  )
  assert.equal(
    restored.conversations[0].messages[1].request?.role,
    'pnu-student',
  )
})

test('starting a new conversation and switching back preserves both transcripts', () => {
  let state = append(emptyConversationState, [question, answer], 'chat-1')
  state = conversationReducer(state, { type: 'select', id: null })
  assert.equal(state.conversations.length, 1)
  state = append(
    state,
    [{ ...question, id: 'question-2', content: '다른 기관의 문서를 찾아줘' }],
    'chat-2',
  )
  state = conversationReducer(state, { type: 'select', id: 'chat-1' })
  state = append(
    state,
    [{ ...question, id: 'question-3', content: '추가 질문' }],
    'unused-id',
  )
  assert.equal(state.conversations.length, 2)
  assert.equal(state.conversations[0].messages.length, 3)
  assert.equal(
    state.conversations[1].messages[0].content,
    '다른 기관의 문서를 찾아줘',
  )
})

test('deleting the active conversation and undoing restores its sources and selection', () => {
  const initial = append(emptyConversationState, [question, answer], 'chat-1')
  const deleted = conversationReducer(initial, { type: 'delete', id: 'chat-1' })
  assert.equal(deleted.activeId, null)
  assert.equal(deleted.conversations.length, 0)
  const restored = conversationReducer(deleted, { type: 'restore' })
  assert.equal(restored.activeId, 'chat-1')
  assert.deepEqual(restored.conversations, initial.conversations)
  assert.equal(restored.deleted, null)
})

test('a failed request and retry do not duplicate the user question', () => {
  let state = append(emptyConversationState, [question], 'chat-1')
  state = append(
    state,
    [
      {
        id: 'error-1',
        role: 'assistant',
        content: '다시 시도해 주세요',
        status: 'error',
        request: answer.request,
      },
    ],
    'unused-id',
  )
  state = append(state, [answer], 'unused-id')
  assert.equal(
    state.conversations[0].messages.filter((item) => item.role === 'user')
      .length,
    1,
  )
  assert.equal(state.conversations[0].messages.at(-1)?.content, answer.content)
})

test('malformed browser storage is ignored without breaking the app', () => {
  for (const raw of [null, '{broken', 'null', '[]', '{"conversations":null}']) {
    assert.deepEqual(parseConversations(raw), emptyConversationState)
  }
  const valid = append(emptyConversationState, [question, answer], 'chat-1')
  const malformed = {
    ...valid.conversations[0],
    id: 'bad-chat',
    messages: [{ ...answer, results: {} }],
  }
  const restored = parseConversations(
    JSON.stringify({
      activeId: 'bad-chat',
      conversations: [malformed, ...valid.conversations],
    }),
  )
  assert.equal(restored.conversations.length, 1)
  assert.equal(restored.activeId, null)
})

test('only the most recent 30 conversations are retained without trimming their messages', () => {
  let state = emptyConversationState
  for (let index = 0; index < MAX_CONVERSATIONS + 1; index++) {
    state = conversationReducer(state, { type: 'select', id: null })
    state = append(state, [question, answer], `chat-${index}`, index)
  }
  assert.equal(state.conversations.length, MAX_CONVERSATIONS)
  assert.equal(state.conversations[0].id, `chat-${MAX_CONVERSATIONS}`)
  assert.equal(state.conversations.at(-1)?.id, 'chat-1')
  assert.deepEqual(state.conversations[0].messages, [question, answer])
})
