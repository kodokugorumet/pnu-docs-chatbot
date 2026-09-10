import { useCallback, useState, useSyncExternalStore } from 'react'
import { createConversationStore } from '../state/conversationStorage'
import type { Message } from '../types/chat'

export default function useConversations() {
  const [store] = useState(createConversationStore)
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot)
  const setMessages = useCallback(
    (
      update: Message[] | ((messages: Message[]) => Message[]),
      targetId?: string | null,
    ) =>
      store.dispatch({
        type: 'messages',
        update,
        targetId,
        id: crypto.randomUUID(),
        now: Date.now(),
      }),
    [store],
  )
  return {
    ...state,
    messages:
      state.conversations.find((item) => item.id === state.activeId)
        ?.messages ?? [],
    setMessages,
    selectConversation: (id: string | null) =>
      store.dispatch({ type: 'select', id }),
    deleteConversation: (id: string) =>
      store.dispatch({ type: 'delete', id }),
    restoreConversation: () => store.dispatch({ type: 'restore' }),
    dismissDeleted: () => store.dispatch({ type: 'dismiss-delete' }),
  }
}
