import {
  BookOpen,
  CircleHelp,
  MessageSquare,
  PanelLeftClose,
  Search,
  Settings2,
  SquarePen,
  Trash2,
  X,
} from 'lucide-react'
import { useState } from 'react'
import type { Conversation } from '../types/chat'

export default function ChatSidebar({
  conversations,
  activeId,
  disabled,
  ready,
  checking,
  onNew,
  onSelect,
  onDelete,
  onSettings,
  onClose,
  mobile = false,
}: {
  conversations: Conversation[]
  activeId: string | null
  disabled: boolean
  ready: boolean
  checking: boolean
  onNew: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onSettings: () => void
  onClose: () => void
  mobile?: boolean
}) {
  const [query, setQuery] = useState('')
  const filtered = conversations.filter((item) =>
    item.messages.some((message) =>
      message.content
        .toLocaleLowerCase()
        .includes(query.trim().toLocaleLowerCase()),
    ),
  )

  return (
    <aside aria-label="대화 목록" className="left-rail">
      <div className="rail-brand-row">
        <div className="brand">
          <img alt="산지니" src="/sanjini.webp" />
          <span className="brand-wordmark">
            PNU <strong>Docs</strong>
            <span className="brand-dot" />
          </span>
        </div>
        <button
          aria-label={mobile ? '대화 목록 닫기' : '사이드바 접기'}
          className="icon-button"
          onClick={onClose}
          title={mobile ? '닫기' : '사이드바 접기'}
          type="button"
        >
          {mobile ? <X size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>
      <button
        className="new-chat-button"
        disabled={disabled}
        onClick={onNew}
        type="button"
      >
        <SquarePen size={18} />
        <span>새 대화</span>
      </button>
      <label className="history-search">
        <Search aria-hidden="true" size={17} />
        <input
          aria-label="대화 검색"
          onChange={(event) => setQuery(event.target.value)}
          placeholder="대화 검색"
          type="search"
          value={query}
        />
      </label>
      <nav aria-label="저장된 대화" className="history-section">
        <div className="section-title">
          {query ? '검색 결과' : '최근 대화'}
          <span>{filtered.length || ''}</span>
        </div>
        {filtered.length ? (
          <div className="conversation-list">
            {filtered.map((conversation) => (
              <div
                className={`conversation-item ${conversation.id === activeId ? 'is-active' : ''}`}
                key={conversation.id}
              >
                <button
                  aria-current={
                    conversation.id === activeId ? 'page' : undefined
                  }
                  className="conversation-select"
                  disabled={disabled}
                  onClick={() => onSelect(conversation.id)}
                  title={conversation.title}
                  type="button"
                >
                  <MessageSquare aria-hidden="true" size={15} />
                  <span>{conversation.title}</span>
                </button>
                <button
                  aria-label={`${conversation.title} 대화 삭제`}
                  className="conversation-delete icon-button"
                  disabled={disabled}
                  onClick={() => onDelete(conversation.id)}
                  title="대화 삭제"
                  type="button"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="history-empty">
            <MessageSquare size={21} strokeWidth={1.5} />
            <p>
              {query ? '일치하는 대화가 없어요' : '첫 대화를 시작해 보세요'}
            </p>
            <span>
              {query
                ? '다른 단어로 검색해 보세요.'
                : '나눈 대화가 여기에 모여요.'}
            </span>
          </div>
        )}
      </nav>
      <div className="rail-bottom">
        <div className="library-note">
          <BookOpen size={18} />
          <div>
            <strong>답변에서 원문까지</strong>
            <span>출처를 눌러 근거를 확인하세요.</span>
          </div>
        </div>
        <button className="rail-settings" onClick={onSettings} type="button">
          <Settings2 size={17} />
          <span>설정 및 연결 상태</span>
          <span
            role="img"
            aria-label={
              ready ? '연결됨' : checking ? '연결 확인 중' : '연결 필요'
            }
            className={`status-dot ${ready ? 'is-ready' : checking ? 'is-checking' : 'is-offline'}`}
          />
        </button>
        <p className="storage-note">
          <CircleHelp size={13} />
          최근 30개 대화가 이 브라우저에 저장돼요.
        </p>
      </div>
    </aside>
  )
}
