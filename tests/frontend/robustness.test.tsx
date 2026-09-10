import { describe, expect, it } from 'vitest'
import fc from 'fast-check'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import AnswerContent from '../../src/components/AnswerContent'
import { positiveIntegerSetting } from '../../src/chat/settings'
import { validMessage } from '../../src/state/messageValidation'
import {
  conversationReducer,
  emptyConversationState,
  parseConversations,
  MAX_CONVERSATIONS,
} from '../../src/state/conversations'
import type { ConversationState } from '../../src/state/conversations'
import { storedAnswer, response } from './fixtures'

it('rejects unusable build-time limits', () => {
  for (const value of [
    undefined,
    '',
    0,
    -1,
    1.5,
    Infinity,
    'bad',
    Number.MAX_SAFE_INTEGER + 1,
  ])
    expect(positiveIntegerSetting(value, 1000)).toBe(1000)
  expect(positiveIntegerSetting('500', 1000)).toBe(500)
})

it('normalizes arbitrary answer heading levels without skipping or changing the text', () => {
  const html = renderToStaticMarkup(
    createElement(AnswerContent, {
      content:
        '###### First\n\n# Root\n\n### Child\n\n###### Deep\n\n## Parent',
      sourceNumbers: [],
      onCitationSelect: () => {},
    }),
  )
  expect(
    [...html.matchAll(/<h(\d)>(.*?)<\/h\d>/g)].map((match) => [
      Number(match[1]),
      match[2],
    ]),
  ).toEqual([
    [2, 'First'],
    [2, 'Root'],
    [3, 'Child'],
    [4, 'Deep'],
    [3, 'Parent'],
  ])
})

const damaged = [
  null,
  [],
  {},
  { id: 5, role: 'user', content: 'x' },
  { id: '1', role: ['assistant'], content: 'x' },
  { id: '1', role: 'assistant', content: {} },
  ...[
    { results: {} },
    { results: [null] },
    { results: [{ chunk_id: 1 }] },
    { results: [{ chunk_id: '1', preview: {} }] },
    { results: [{ chunk_id: '1', section_path: [4] }] },
    { results: [{ chunk_id: '1', locations: [null] }] },
    { results: [{ chunk_id: '1', scores: { rank: 'bad' } }] },
    { claims: {} },
    { claims: [null] },
    { claims: [{ text: 'x', supported: true, source_numbers: [{}] }] },
    { claims: [{ text: 'x', supported: true, citations: [null] }] },
    {
      claims: [
        { text: 'x', supported: false, missing_critical_values: [false] },
      ],
    },
    { trace: null },
    { trace: { stages: {} } },
    { trace: { stages: [null] } },
    { trace: { stages: [{ duration_ms: 'a' }] } },
    { generation: { requested: 'auto', used: 'local', model: {} } },
    { request: { question: 'x' } },
    { resolvedRole: { id: 'x', label: {} } },
    { error: { code: 'bad', retryable: {} } },
    { durationMs: NaN },
    { retrieval: [] },
    { status: 'unknown' },
  ].map((extra) => ({ ...storedAnswer(), ...extra })),
]
describe('untrusted saved messages', () => {
  it.each(damaged)('rejects malformed nested data %#', (value) => {
    expect(validMessage(value)).toBe(false)
  })
  it('accepts complete typed evidence including nullable metadata', () => {
    const data = response()
    const message = storedAnswer({
      generation: {
        requested: 'local',
        used: 'local',
        model: null,
        fallback_reason: null,
        attempts: [
          {
            provider: 'local',
            model: 'x',
            status: 'ok',
            error: null,
            elapsed_ms: null,
            duration_ms: 1,
          },
        ],
      },
      resolvedRole: {
        requested: null,
        id: 'general',
        label: '일반',
        institutions: [],
      },
      trace: { stages: [{ id: 'search', duration_ms: null, detail: null }] },
      results: [
        {
          chunk_id: 'a',
          section_path: '장',
          location: null,
          locations: [{ section_path: ['장'], page: null }],
          metadata: { x: 1 },
          scores: { rank: null },
          locations_truncated: false,
        },
        {
          chunk_id: 'b',
          section_path: [],
          scores: { rank: 2 },
          location: { page: 1 },
        },
      ],
      claims: data.claims,
    })
    expect(validMessage(message)).toBe(true)
  })
  it('filters corrupted conversations while preserving sound ones and removing duplicate IDs', () => {
    const good = {
      id: 'good',
      title: '대화',
      updatedAt: 1,
      messages: [storedAnswer()],
    }
    const parsed = parseConversations(
      JSON.stringify({
        activeId: 'good',
        conversations: [
          ...damaged.map((message, index) => ({
            ...good,
            id: `bad-${index}`,
            messages: [message],
          })),
          {
            ...good,
            id: 'dupe-message',
            messages: [storedAnswer(), storedAnswer()],
          },
          null,
          { ...good, updatedAt: 'invalid' },
          good,
          good,
        ],
      }),
    )
    expect(parsed.conversations).toEqual([good])
    expect(parsed.activeId).toBe('good')
  })
  it('survives 1000 arbitrary stored payloads and validates all retained messages', () => {
    fc.assert(
      fc.property(fc.jsonValue(), (value) => {
        const parsed = parseConversations(JSON.stringify(value))
        expect(parsed.conversations.length).toBeLessThanOrEqual(
          MAX_CONVERSATIONS,
        )
        for (const conversation of parsed.conversations)
          expect(conversation.messages.every(validMessage)).toBe(true)
      }),
      { numRuns: 1000, seed: 20260910 },
    )
  })
})

it('preserves selection and unique IDs through randomized save, delete, switch and undo sequences', () => {
  fc.assert(
    fc.property(
      fc.array(
        fc.record({
          kind: fc.integer({ min: 0, max: 6 }),
          target: fc.nat(50),
          text: fc.string(),
        }),
        { maxLength: 160 },
      ),
      (actions) => {
        let state: ConversationState = emptyConversationState
        for (const [index, action] of actions.entries()) {
          const id = `conversation-${action.target}`
          if (action.kind === 0)
            state = conversationReducer(state, {
              type: 'messages',
              id: `created-${index}`,
              now: index + 1,
              update: (messages) => [
                ...messages,
                { id: `message-${index}`, role: 'user', content: action.text },
              ],
            })
          if (action.kind === 1)
            state = conversationReducer(state, { type: 'select', id: null })
          if (action.kind === 2)
            state = conversationReducer(state, {
              type: 'select',
              id:
                state.conversations[
                  action.target % Math.max(1, state.conversations.length)
                ]?.id ?? id,
            })
          if (action.kind === 3)
            state = conversationReducer(state, {
              type: 'delete',
              id:
                state.conversations[
                  action.target % Math.max(1, state.conversations.length)
                ]?.id ?? id,
            })
          if (action.kind === 4)
            state = conversationReducer(state, { type: 'restore' })
          if (action.kind === 5)
            state = conversationReducer(state, { type: 'dismiss-delete' })
          if (action.kind === 6)
            state = conversationReducer(state, {
              type: 'storage-error',
              value: Boolean(action.target % 2),
            })
          expect(state.conversations.length).toBeLessThanOrEqual(
            MAX_CONVERSATIONS,
          )
          expect(new Set(state.conversations.map((item) => item.id)).size).toBe(
            state.conversations.length,
          )
          expect(
            state.activeId === null ||
              state.conversations.some((item) => item.id === state.activeId),
          ).toBe(true)
          expect(
            parseConversations(JSON.stringify(state)).conversations,
          ).toEqual(state.conversations)
        }
      },
    ),
    { numRuns: 300, seed: 9142026 },
  )
})

it('restores an older deleted conversation even when the history limit has been reached', () => {
  let state = conversationReducer(emptyConversationState, {
    type: 'messages',
    id: 'old',
    now: 0,
    update: [storedAnswer()],
  })
  state = conversationReducer(state, { type: 'delete', id: 'old' })
  for (let index = 0; index < 30; index++) {
    state = conversationReducer(state, { type: 'select', id: null })
    state = conversationReducer(state, {
      type: 'messages',
      id: `new-${index}`,
      now: index + 1,
      update: [storedAnswer()],
    })
  }
  state = conversationReducer(state, { type: 'restore' })
  expect(state.conversations).toHaveLength(30)
  expect(state.activeId).toBe('old')
  expect(
    state.conversations.find((item) => item.id === 'old')?.messages,
  ).toEqual([storedAnswer()])
  state = conversationReducer(state, {
    type: 'messages',
    id: 'unused',
    now: 40,
    update: [],
  })
  expect(state.activeId).toBeNull()
})
