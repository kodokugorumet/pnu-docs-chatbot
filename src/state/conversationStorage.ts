import {
  CONVERSATION_STORAGE_KEY,
  MAX_CONVERSATIONS,
  conversationReducer,
  emptyConversationState,
  parseConversations,
} from './conversations'
import type { ConversationAction, ConversationState } from './conversations'
import type { Conversation } from '../types/chat'
import { isRecord } from './messageValidation'

type Snapshot = {
  conversations: Conversation[]
  activeId: string | null
  deletions: Record<string, string>
}
type Removal = {
  type: 'delete'
  id: string
  token: string
  conversation: Conversation
}
type Change =
  | {
      type: 'save'
      conversation: Conversation
      changed: Conversation['messages']
      create: boolean
    }
  | Removal
  | { type: 'restore'; removal: Removal }
  | { type: 'select' }

function readSnapshot(): Snapshot {
  const raw = localStorage.getItem(CONVERSATION_STORAGE_KEY)
  const parsed = parseConversations(raw)
  let deletions: Record<string, string> = {}
  try {
    const value: unknown = JSON.parse(raw ?? 'null')
    if (isRecord(value) && isRecord(value.deletions)) {
      deletions = Object.fromEntries(
        Object.entries(value.deletions).filter(
          (entry): entry is [string, string] =>
            typeof entry[1] === 'string',
        ),
      )
    }
  } catch {
    // Invalid legacy storage is already rejected by parseConversations.
  }
  return {
    conversations: parsed.conversations,
    activeId: parsed.activeId,
    deletions,
  }
}

function applyChange(snapshot: Snapshot, change: Change): Snapshot {
  const deletions: Record<string, string> = Object.assign(
    Object.create(null),
    snapshot.deletions,
  )
  let conversations = snapshot.conversations
  if (change.type === 'delete') {
    change.conversation =
      conversations.find((item) => item.id === change.id) ??
      change.conversation
    deletions[change.id] = change.token
    conversations = conversations.filter((item) => item.id !== change.id)
  }
  if (change.type === 'restore') {
    const { removal } = change
    if (deletions[removal.id] !== removal.token) return snapshot
    delete deletions[removal.id]
    conversations = [
      removal.conversation,
      ...conversations.filter((item) => item.id !== removal.id),
    ].slice(0, MAX_CONVERSATIONS)
  }
  if (change.type === 'save') {
    const incoming = change.conversation
    const current = conversations.find((item) => item.id === incoming.id)
    if (deletions[incoming.id] || (!current && !change.create))
      return snapshot
    const messages = new Map(
      current?.messages.map((message) => [message.id, message]),
    )
    for (const message of change.changed) messages.set(message.id, message)
    conversations = [
      {
        ...incoming,
        updatedAt: Math.max(incoming.updatedAt, current?.updatedAt ?? 0),
        messages: [...messages.values()],
      },
      ...conversations.filter((item) => item.id !== incoming.id),
    ]
  }
  return {
    ...snapshot,
    deletions,
    conversations: [...conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS),
  }
}

/** One store per mounted app; Web Locks serialize writes across browser tabs. */
export function createConversationStore() {
  let state: ConversationState
  try {
    state = { ...emptyConversationState, ...readSnapshot() }
  } catch {
    state = { ...emptyConversationState, storageError: true }
  }
  const listeners = new Set<() => void>()
  const pending: Change[] = []
  let removal: Removal | null = null
  let writing = false

  function publish(next: ConversationState) {
    state = next
    for (const listener of listeners) listener()
  }

  function synchronize(snapshot: Snapshot) {
    publish({
      conversations: snapshot.conversations,
      activeId: snapshot.conversations.some(
        (item) => item.id === state.activeId,
      )
        ? state.activeId
        : null,
      deleted: removal?.conversation ?? null,
      storageError: false,
    })
  }

  async function flush() {
    if (writing || !pending.length) return
    writing = true
    let succeeded = false
    try {
      // Without cross-tab locking, keep the transcript in memory instead of risking lost writes.
      if (!navigator.locks)
        throw new Error('Safe browser storage is unavailable')
      await navigator.locks.request(CONVERSATION_STORAGE_KEY, () => {
        let snapshot = readSnapshot()
        const count = pending.length
        for (const change of pending)
          snapshot = applyChange(snapshot, change)
        snapshot.activeId = snapshot.conversations.some(
          (item) => item.id === state.activeId,
        )
          ? state.activeId
          : null
        localStorage.setItem(
          CONVERSATION_STORAGE_KEY,
          JSON.stringify(snapshot),
        )
        pending.splice(0, count)
        synchronize(snapshot)
      })
      succeeded = true
    } catch {
      publish({ ...state, storageError: true })
    } finally {
      writing = false
    }
    if (succeeded && pending.length) void flush()
  }

  function onStorage(event: StorageEvent) {
    if (event.key !== CONVERSATION_STORAGE_KEY && event.key !== null) return
    if (pending.length) {
      void flush()
      return
    }
    try {
      if (event.storageArea !== localStorage) return
      synchronize(readSnapshot())
    } catch {
      publish({ ...state, storageError: true })
    }
  }

  return {
    getSnapshot: () => state,
    subscribe(listener: () => void) {
      listeners.add(listener)
      if (listeners.size === 1)
        window.addEventListener('storage', onStorage)
      return () => {
        listeners.delete(listener)
        if (!listeners.size)
          window.removeEventListener('storage', onStorage)
      }
    },
    dispatch(action: ConversationAction) {
      const previous = state
      const next = conversationReducer(previous, action)
      if (next === previous) return state.activeId
      if (action.type === 'messages') {
        const conversation = next.conversations.find(
          (item) => item.id === next.activeId,
        )
        if (conversation) {
          const current = previous.conversations.find(
            (item) => item.id === conversation.id,
          )
          pending.push({
            type: 'save',
            conversation,
            changed: conversation.messages.filter(
              (message) => !current?.messages.includes(message),
            ),
            create: !current,
          })
        } else {
          pending.push({ type: 'select' })
        }
      } else if (action.type === 'delete' && next.deleted) {
        removal = {
          type: 'delete',
          id: action.id,
          token: crypto.randomUUID(),
          conversation: next.deleted,
        }
        pending.push(removal)
      } else if (action.type === 'restore' && removal) {
        pending.push({ type: 'restore', removal })
        removal = null
      } else if (action.type === 'dismiss-delete') {
        removal = null
      } else if (action.type === 'select') {
        pending.push({ type: 'select' })
      }
      if (!next.deleted) removal = null
      publish(next)
      void flush()
      return state.activeId
    },
  }
}
