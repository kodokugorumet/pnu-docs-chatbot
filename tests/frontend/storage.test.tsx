import { StrictMode } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { createConversationStore } from '../../src/state/conversationStorage'
import {
  CONVERSATION_STORAGE_KEY,
  MAX_CONVERSATIONS,
} from '../../src/state/conversations'
import useConversations from '../../src/hooks/useConversations'
import { seed, storedAnswer } from './fixtures'

type Store = ReturnType<typeof createConversationStore>
const cleanups: (() => void)[] = []
afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup()
})
function store() {
  const result = createConversationStore()
  cleanups.push(result.subscribe(() => {}))
  return result
}
function append(
  target: Store,
  id: string,
  targetId?: string | null,
  now = Date.now(),
) {
  return target.dispatch({
    type: 'messages',
    id,
    now,
    targetId,
    update: (messages) => [...messages, { id, role: 'user', content: id }],
  })
}
function saved() {
  return JSON.parse(localStorage.getItem(CONVERSATION_STORAGE_KEY)!)
}
async function settled() {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}
function announce(
  key: string | null = CONVERSATION_STORAGE_KEY,
  storageArea: Storage = localStorage,
) {
  window.dispatchEvent(new StorageEvent('storage', { key, storageArea }))
}

it('keeps independent conversations from two writers and each tab selection', async () => {
  const first = store()
  const second = store()
  append(first, 'a')
  append(second, 'b')
  await settled()
  expect(
    saved()
      .conversations.map((item: { id: string }) => item.id)
      .sort(),
  ).toEqual(['a', 'b'])
  announce()
  expect(first.getSnapshot().activeId).toBe('a')
  expect(second.getSnapshot().activeId).toBe('b')
  expect(store().getSnapshot().conversations).toHaveLength(2)
})

it('merges simultaneous messages in the same conversation exactly once', async () => {
  seed([storedAnswer()])
  const first = store()
  const second = store()
  append(first, 'a', 'saved')
  append(second, 'b', 'saved')
  await settled()
  expect(
    saved().conversations[0].messages.map(
      (message: { id: string }) => message.id,
    ),
  ).toEqual(['answer', 'a', 'b'])
  announce()
  expect(first.getSnapshot().conversations[0].messages).toHaveLength(3)
})

it('does not revive a deleted conversation from a stale tab or a late response', async () => {
  seed([storedAnswer()])
  const first = store()
  const stale = store()
  first.dispatch({ type: 'delete', id: 'saved' })
  await settled()
  append(stale, 'stale-answer', 'saved')
  await settled()
  expect(saved().conversations).toEqual([])
  expect(stale.getSnapshot().activeId).toBeNull()
  append(stale, 'late-answer', 'saved')
  expect(stale.getSnapshot().conversations).toEqual([])
})

it('undo includes messages saved by another tab before deletion', async () => {
  seed([storedAnswer()])
  const first = store()
  const second = store()
  append(second, 'latest', 'saved')
  first.dispatch({ type: 'delete', id: 'saved' })
  first.dispatch({ type: 'restore' })
  await settled()
  expect(
    saved().conversations[0].messages.map(
      (message: { id: string }) => message.id,
    ),
  ).toEqual(['answer', 'latest'])
  expect(first.getSnapshot().activeId).toBe('saved')
})

it('an old undo cannot supersede a later deletion from another tab', async () => {
  seed([storedAnswer()])
  const first = store()
  const second = store()
  first.dispatch({ type: 'delete', id: 'saved' })
  await settled()
  second.dispatch({ type: 'delete', id: 'saved' })
  await settled()
  first.dispatch({ type: 'restore' })
  await settled()
  expect(saved().conversations).toEqual([])
  expect(first.getSnapshot().activeId).toBeNull()
})

it('keeps the restored conversation even when thirty newer conversations exist', async () => {
  seed([storedAnswer()])
  const target = store()
  target.dispatch({ type: 'delete', id: 'saved' })
  for (let index = 0; index < MAX_CONVERSATIONS; index++) {
    target.dispatch({ type: 'select', id: null })
    append(target, `new-${index}`, undefined, index + 2)
  }
  target.dispatch({ type: 'restore' })
  await settled()
  expect(saved().conversations).toHaveLength(MAX_CONVERSATIONS)
  expect(
    saved().conversations.some(
      (item: { id: string }) => item.id === 'saved',
    ),
  ).toBe(true)
  expect(target.getSnapshot().activeId).toBe('saved')
})

it('does not resurrect cleared storage when an old tab appends a message', async () => {
  seed([storedAnswer()])
  const target = store()
  localStorage.clear()
  append(target, 'stale', 'saved')
  await settled()
  expect(saved().conversations).toEqual([])
  append(target, 'new')
  await settled()
  localStorage.clear()
  announce(null)
  expect(target.getSnapshot().conversations).toEqual([])
})

it('handles storage events without rewriting snapshots or switching tabs', async () => {
  seed([storedAnswer()])
  const target = store()
  const write = vi.spyOn(Storage.prototype, 'setItem')
  announce('unrelated')
  announce(CONVERSATION_STORAGE_KEY, sessionStorage)
  announce()
  expect(write).not.toHaveBeenCalled()
  append(target, 'queued')
  announce()
  await settled()
  expect(saved().conversations[0].messages).toHaveLength(2)
  const read = vi
    .spyOn(Storage.prototype, 'getItem')
    .mockImplementation(() => {
      throw new Error('blocked')
    })
  announce()
  expect(target.getSnapshot().storageError).toBe(true)
  read.mockRestore()
  announce()
  expect(target.getSnapshot().storageError).toBe(false)
})

it('retains unsaved changes across quota failure and retries them with new writes', async () => {
  const target = store()
  const write = vi
    .spyOn(Storage.prototype, 'setItem')
    .mockImplementation(() => {
      throw new Error('quota')
    })
  append(target, 'first')
  await settled()
  expect(target.getSnapshot().storageError).toBe(true)
  expect(target.getSnapshot().conversations[0].messages).toHaveLength(1)
  write.mockRestore()
  append(target, 'second')
  await settled()
  expect(saved().conversations[0].messages).toHaveLength(2)
  expect(target.getSnapshot().storageError).toBe(false)
})

it('keeps in-memory chat usable when cross-tab locks are unavailable or rejected', async () => {
  const locks = navigator.locks
  Object.defineProperty(navigator, 'locks', {
    configurable: true,
    value: undefined,
  })
  const target = store()
  append(target, 'offline')
  await settled()
  expect(target.getSnapshot().storageError).toBe(true)
  expect(localStorage.getItem(CONVERSATION_STORAGE_KEY)).toBeNull()
  Object.defineProperty(navigator, 'locks', {
    configurable: true,
    value: locks,
  })
  vi.mocked(locks.request).mockRejectedValueOnce(new Error('denied'))
  append(target, 'retry')
  await settled()
  expect(target.getSnapshot().storageError).toBe(true)
  append(target, 'available')
  await settled()
  expect(saved().conversations[0].messages).toHaveLength(3)
})

it('validates legacy storage and deletion tokens without trusting prototype properties', async () => {
  localStorage.setItem(CONVERSATION_STORAGE_KEY, '{broken')
  expect(store().getSnapshot().conversations).toEqual([])
  localStorage.setItem(
    CONVERSATION_STORAGE_KEY,
    JSON.stringify({ deletions: { bad: 2, valid: 'token' } }),
  )
  const target = store()
  append(target, '__proto__')
  await settled()
  target.dispatch({ type: 'delete', id: '__proto__' })
  await settled()
  target.dispatch({ type: 'restore' })
  await settled()
  expect(saved().conversations[0].id).toBe('__proto__')
  expect(saved().deletions).toEqual({ valid: 'token' })
})

it('does not duplicate writes or listeners during StrictMode mounting', async () => {
  const view = renderHook(() => useConversations(), { wrapper: StrictMode })
  act(() => {
    view.result.current.setMessages([
      { id: 'one', role: 'user', content: 'one' },
    ])
  })
  await settled()
  expect(saved().conversations).toHaveLength(1)
  expect(saved().conversations[0].messages).toHaveLength(1)
  view.unmount()
  const reload = renderHook(() => useConversations(), {
    wrapper: StrictMode,
  })
  await waitFor(() =>
    expect(reload.result.current.messages).toHaveLength(1),
  )
})

it('keeps storage synchronization alive until the final subscriber leaves', () => {
  const target = createConversationStore()
  const first = vi.fn()
  const second = vi.fn()
  const stopFirst = target.subscribe(first)
  const stopSecond = target.subscribe(second)
  seed([storedAnswer()])
  announce()
  expect(first).toHaveBeenCalledOnce()
  expect(second).toHaveBeenCalledOnce()
  stopFirst()
  localStorage.clear()
  announce(null)
  expect(first).toHaveBeenCalledOnce()
  expect(second).toHaveBeenCalledTimes(2)
  expect(target.getSnapshot().conversations).toEqual([])
  stopSecond()
  announce()
  expect(second).toHaveBeenCalledTimes(2)
})

it('retains archived messages when clearing the current view and ignores stale delete controls', async () => {
  seed([storedAnswer()])
  const target = store()
  target.dispatch({ type: 'messages', update: [], id: 'unused', now: 2 })
  await settled()
  expect(store().getSnapshot().activeId).toBeNull()
  expect(saved().conversations[0].messages).toHaveLength(1)
  target.dispatch({ type: 'delete', id: 'saved' })
  target.dispatch({ type: 'delete', id: 'already-gone' })
  await settled()
  announce()
  expect(target.getSnapshot().deleted).toBeNull()
  target.dispatch({ type: 'restore' })
  expect(target.getSnapshot().conversations).toEqual([])
})
