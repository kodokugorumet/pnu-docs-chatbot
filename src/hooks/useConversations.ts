import { useCallback, useEffect, useReducer } from 'react'
import {
  CONVERSATION_STORAGE_KEY,
  conversationReducer,
  emptyConversationState,
  parseConversations,
} from '../state/conversations'
import type { Message } from '../types/chat'

export default function useConversations() {
  const [state, dispatch] = useReducer(conversationReducer, undefined, () => {
    try {
      return parseConversations(localStorage.getItem(CONVERSATION_STORAGE_KEY))
    } catch {
      return { ...emptyConversationState, storageError: true }
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(
        CONVERSATION_STORAGE_KEY,
        JSON.stringify({
          conversations: state.conversations,
          activeId: state.activeId,
        }),
      )
      dispatch({ type: 'storage-error', value: false })
    } catch {
      dispatch({ type: 'storage-error', value: true })
    }
  }, [state.conversations, state.activeId])

  const setMessages = useCallback(
    (update: Message[] | ((messages: Message[]) => Message[])) => {
      dispatch({
        type: 'messages',
        update,
        id: crypto.randomUUID(),
        now: Date.now(),
      })
    },
    [],
  )

  return {
    ...state,
    messages:
      state.conversations.find((item) => item.id === state.activeId)
        ?.messages ?? [],
    setMessages,
    selectConversation: (id: string | null) => dispatch({ type: 'select', id }),
    deleteConversation: (id: string) => dispatch({ type: 'delete', id }),
    restoreConversation: () => dispatch({ type: 'restore' }),
    dismissDeleted: () => dispatch({ type: 'dismiss-delete' }),
  }
}
