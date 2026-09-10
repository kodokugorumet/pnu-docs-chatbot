import {
  ArrowUp,
  Building2,
  LoaderCircle,
  RefreshCw,
  SlidersHorizontal,
  Square,
} from 'lucide-react'
import { useLayoutEffect } from 'react'
import type { RefObject } from 'react'

export default function ChatComposer({
  value,
  onChange,
  onSubmit,
  onCancel,
  onSettings,
  onRefresh,
  inputRef,
  loading,
  ready,
  checking,
  institution,
  institutions,
  onInstitutionChange,
  maxLength,
  hasMessages,
}: {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  onCancel: () => void
  onSettings: () => void
  onRefresh: () => void
  inputRef: RefObject<HTMLTextAreaElement | null>
  loading: boolean
  ready: boolean
  checking: boolean
  institution: string
  institutions: string[]
  onInstitutionChange: (value: string) => void
  maxLength: number
  hasMessages: boolean
}) {
  useLayoutEffect(() => {
    const textarea = inputRef.current!
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, 56), 200)}px`
  }, [value, inputRef])

  return (
    <div className="composer-container">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit()
        }}
      >
        <textarea
          aria-label="질문 입력"
          maxLength={maxLength}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === 'Enter' &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing &&
              event.nativeEvent.keyCode !== 229
            ) {
              event.preventDefault()
              if (!loading && ready && value.trim())
                event.currentTarget.form?.requestSubmit()
            }
          }}
          placeholder={
            hasMessages
              ? '다른 궁금한 내용을 물어보세요'
              : '궁금한 내용을 편하게 물어보세요'
          }
          ref={inputRef}
          rows={2}
          value={value}
        />
        <div className="composer-toolbar">
          <div className="composer-tools">
            <label className="institution-picker">
              <Building2 aria-hidden="true" size={15} />
              <select
                aria-label="검색 기관"
                disabled={loading}
                onChange={(event) => onInstitutionChange(event.target.value)}
                value={institution}
              >
                {institutions.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
            <button
              aria-label="답변 설정 열기"
              className="icon-button composer-settings"
              onClick={onSettings}
              title="답변 설정"
              type="button"
            >
              <SlidersHorizontal size={17} />
            </button>
          </div>
          <div className="composer-send">
            {value.length > maxLength * 0.8 && (
              <span className="character-count">
                {value.length.toLocaleString()} / {maxLength.toLocaleString()}
              </span>
            )}
            {loading ? (
              <button
                aria-label="답변 생성 중지"
                className="send-button stop-button"
                onClick={onCancel}
                title="답변 생성 중지"
                type="button"
              >
                <Square fill="currentColor" size={14} />
              </button>
            ) : (
              <button
                aria-label="질문 보내기"
                className="send-button"
                disabled={!ready || !value.trim()}
                title={
                  ready ? '질문 보내기' : '검색 서버 연결 후 보낼 수 있어요'
                }
                type="submit"
              >
                <ArrowUp size={20} />
              </button>
            )}
          </div>
        </div>
      </form>
      {!ready && (
        <div className="connection-notice" role="status">
          {checking ? (
            <LoaderCircle className="loading-icon" size={14} />
          ) : (
            <span className="status-dot is-offline" />
          )}
          <span>
            {checking
              ? '검색 서비스에 연결하고 있어요.'
              : '검색 서비스에 연결할 수 없어요. 질문은 미리 작성할 수 있어요.'}
          </span>
          {!checking && (
            <button onClick={onRefresh} type="button">
              <RefreshCw size={13} />
              다시 연결
            </button>
          )}
        </div>
      )}
      <div className="composer-caption">
        <span>답변의 출처를 함께 확인해 주세요.</span>
        <span className="keyboard-hint">
          Enter 전송 <span>·</span> Shift + Enter 줄바꿈
        </span>
      </div>
    </div>
  )
}
