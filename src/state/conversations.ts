import type { Conversation, Message } from '../types/chat'
import { isRecord, validMessage } from './messageValidation'

export const CONVERSATION_STORAGE_KEY = 'pnu-docs.conversations.v1'
export const MAX_CONVERSATIONS = 30

export type ConversationState = {
  conversations: Conversation[]
  activeId: string | null
  deleted: Conversation | null
  storageError: boolean
}

export const emptyConversationState: ConversationState = {
  conversations: [],
  activeId: null,
  deleted: null,
  storageError: false,
}

type Action =
  | {
      type: 'messages'
      update: Message[] | ((messages: Message[]) => Message[])
      id: string
      now: number
    }
  | { type: 'select'; id: string | null }
  | { type: 'delete'; id: string }
  | { type: 'restore' }
  | { type: 'dismiss-delete' }
  | { type: 'storage-error'; value: boolean }

export function parseConversations(raw: string | null): ConversationState {
  if (!raw) return emptyConversationState
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!isRecord(parsed) || !Array.isArray(parsed.conversations))
      return emptyConversationState
    const seen = new Set<string>()
    const conversations = parsed.conversations
      .filter(
        (item): item is Conversation =>
          isRecord(item) &&
          typeof item.id === 'string' &&
          typeof item.title === 'string' &&
          typeof item.updatedAt === 'number' &&
          Number.isFinite(item.updatedAt) &&
          Array.isArray(item.messages) &&
          item.messages.length > 0 &&
          item.messages.every(validMessage) &&
          new Set(item.messages.map((message) => message.id)).size ===
            item.messages.length,
      )
      .filter((item) => {
        if (seen.has(item.id)) return false
        seen.add(item.id)
        return true
      })
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS)
    return {
      conversations,
      activeId: conversations.some((item) => item.id === parsed.activeId)
        ? (parsed.activeId as string)
        : null,
      deleted: null,
      storageError: false,
    }
  } catch {
    return emptyConversationState
  }
}

export function conversationReducer(
  state: ConversationState,
  action: Action,
): ConversationState {
  switch (action.type) {
    case 'messages': {
      const current = state.conversations.find(
        (item) => item.id === state.activeId,
      )
      const messages =
        typeof action.update === 'function'
          ? action.update(current?.messages ?? [])
          : action.update
      if (!messages.length) return { ...state, activeId: null }
      const conversation: Conversation = {
        id: current?.id ?? action.id,
        title:
          messages
            .find((message) => message.role === 'user')
            ?.content.replace(/\s+/g, ' ')
            .slice(0, 70) ?? '새 대화',
        updatedAt: action.now,
        messages,
      }
      return {
        ...state,
        activeId: conversation.id,
        conversations: [
          conversation,
          ...state.conversations.filter((item) => item.id !== conversation.id),
        ].slice(0, MAX_CONVERSATIONS),
      }
    }
    case 'select':
      return action.id === null ||
        state.conversations.some((item) => item.id === action.id)
        ? { ...state, activeId: action.id }
        : state
    case 'delete':
      return {
        ...state,
        deleted:
          state.conversations.find((item) => item.id === action.id) ?? null,
        activeId: state.activeId === action.id ? null : state.activeId,
        conversations: state.conversations.filter(
          (item) => item.id !== action.id,
        ),
      }
    case 'restore':
      return state.deleted
        ? {
            ...state,
            activeId: state.deleted.id,
            conversations: [
              state.deleted,
              ...state.conversations.filter(
                (item) => item.id !== state.deleted!.id,
              ),
            ]
              .slice(0, MAX_CONVERSATIONS)
              .sort((a, b) => b.updatedAt - a.updatedAt),
            deleted: null,
          }
        : state
    case 'dismiss-delete':
      return { ...state, deleted: null }
    case 'storage-error':
      return state.storageError === action.value
        ? state
        : { ...state, storageError: action.value }
  }
}
