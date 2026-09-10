import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
} from '@testing-library/react'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import ChatComposer from '../../src/components/ChatComposer'
import ChatSidebar from '../../src/components/ChatSidebar'
import Dialog from '../../src/components/Dialog'
import AnswerContent from '../../src/components/AnswerContent'
import useConversations from '../../src/hooks/useConversations'
import { CONVERSATION_STORAGE_KEY } from '../../src/state/conversations'

const composerProps = () => ({
  value: '질문',
  onChange: vi.fn(),
  onSubmit: vi.fn(),
  onCancel: vi.fn(),
  onSettings: vi.fn(),
  onRefresh: vi.fn(),
  inputRef: createRef<HTMLTextAreaElement>(),
  loading: false,
  ready: true,
  checking: false,
  institution: '전체 기관',
  institutions: ['전체 기관', '부산대학교'],
  onInstitutionChange: vi.fn(),
  maxLength: 1000,
  hasMessages: false,
})

describe('chat composer', () => {
  it('sends Enter but protects Korean composition, shift-newlines and other keys', () => {
    const props = composerProps()
    render(<ChatComposer {...props} />)
    const input = screen.getByRole('textbox', { name: '질문 입력' })
    for (const event of [
      { key: 'a' },
      { key: 'Enter', shiftKey: true },
      { key: 'Enter', isComposing: true },
      { key: 'Enter', keyCode: 229 },
    ])
      fireEvent.keyDown(input, event)
    expect(props.onSubmit).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(props.onSubmit).toHaveBeenCalledTimes(1)
    fireEvent.change(input, { target: { value: '수정' } })
    expect(props.onChange).toHaveBeenCalledWith('수정')
    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: '부산대학교' },
    })
    expect(props.onInstitutionChange).toHaveBeenCalledWith('부산대학교')
    fireEvent.click(screen.getByRole('button', { name: '답변 설정 열기' }))
    expect(props.onSettings).toHaveBeenCalledOnce()
  })
  it('blocks empty, offline and busy submissions while preserving a draft', () => {
    const props = composerProps()
    const view = render(<ChatComposer {...props} ready={false} checking />)
    expect(screen.getByText('검색 서비스에 연결하고 있어요.')).toBeTruthy()
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(props.onSubmit).not.toHaveBeenCalled()
    view.rerender(<ChatComposer {...props} ready={false} />)
    fireEvent.click(screen.getByRole('button', { name: '다시 연결' }))
    expect(props.onRefresh).toHaveBeenCalledOnce()
    view.rerender(<ChatComposer {...props} value="   " />)
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(props.onSubmit).not.toHaveBeenCalled()
    view.rerender(
      <ChatComposer {...props} loading hasMessages value={'가'.repeat(950)} />,
    )
    expect(screen.getByText('950 / 1,000')).toBeTruthy()
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(props.onSubmit).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: '답변 생성 중지' }))
    expect(props.onCancel).toHaveBeenCalledOnce()
    expect(
      (screen.getByRole('textbox') as HTMLTextAreaElement).value.length,
    ).toBe(950)
  })
})

describe('conversation navigation', () => {
  it('filters by answer content, switches and deletes the intended conversation', () => {
    const props = {
      conversations: [
        {
          id: '1',
          title: '휴학',
          updatedAt: 1,
          messages: [{ id: 'u', role: 'user' as const, content: '휴학 절차' }],
        },
        {
          id: '2',
          title: '학교',
          updatedAt: 2,
          messages: [
            { id: 'a', role: 'assistant' as const, content: '지원 서류' },
          ],
        },
      ],
      activeId: '1',
      disabled: false,
      ready: true,
      checking: false,
      onNew: vi.fn(),
      onSelect: vi.fn(),
      onDelete: vi.fn(),
      onSettings: vi.fn(),
      onClose: vi.fn(),
    }
    const view = render(<ChatSidebar {...props} />)
    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: '지원' },
    })
    expect(screen.queryByRole('button', { name: '휴학' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: '학교' }))
    fireEvent.click(screen.getByRole('button', { name: '학교 대화 삭제' }))
    expect(props.onSelect).toHaveBeenCalledWith('2')
    expect(props.onDelete).toHaveBeenCalledWith('2')
    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: '없는 검색어' },
    })
    expect(screen.getByText('일치하는 대화가 없어요')).toBeTruthy()
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: '' } })
    view.rerender(
      <ChatSidebar
        {...props}
        conversations={[]}
        ready={false}
        checking
        mobile
      />,
    )
    expect(screen.getByText('첫 대화를 시작해 보세요')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '대화 목록 닫기' }))
    expect(props.onClose).toHaveBeenCalledOnce()
    view.rerender(
      <ChatSidebar
        {...props}
        conversations={[]}
        ready={false}
        checking={false}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: '새 대화' }))
    fireEvent.click(screen.getByRole('button', { name: /설정 및 연결 상태/ }))
    expect(props.onNew).toHaveBeenCalledOnce()
    expect(props.onSettings).toHaveBeenCalledOnce()
  })
})

describe('native dialog boundaries', () => {
  it('opens, closes, supports Escape and only dismisses clicks outside its bounds', () => {
    const onClose = vi.fn()
    const view = render(
      <Dialog open={false} labelledBy="title" onClose={onClose}>
        <h2 id="title">설정</h2>
      </Dialog>,
    )
    expect(screen.queryByRole('dialog')).toBeNull()
    view.rerender(
      <Dialog open labelledBy="title" onClose={onClose}>
        <h2 id="title">설정</h2>
      </Dialog>,
    )
    const dialog = screen.getByRole('dialog')
    vi.spyOn(dialog, 'getBoundingClientRect').mockReturnValue({
      left: 10,
      right: 110,
      top: 20,
      bottom: 120,
    } as DOMRect)
    fireEvent.click(screen.getByRole('heading'))
    fireEvent.click(dialog, { clientX: 50, clientY: 50 })
    expect(onClose).not.toHaveBeenCalled()
    for (const [clientX, clientY] of [
      [0, 50],
      [120, 50],
      [50, 0],
      [50, 130],
    ])
      fireEvent.click(dialog, { clientX, clientY })
    expect(onClose).toHaveBeenCalledTimes(4)
    fireEvent(dialog, new Event('cancel', { bubbles: true }))
    expect(onClose).toHaveBeenCalledTimes(5)
    view.rerender(
      <Dialog open={false} labelledBy="title" onClose={onClose}>
        <h2 id="title">설정</h2>
      </Dialog>,
    )
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})

it('citation buttons are real controls and remain focused through rerenders', () => {
  const select = vi.fn()
  const view = render(
    <AnswerContent
      content={
        '# 제목\n\n### 소제목\n\n**근거 [2]**\n\n![설명](https://example.org/pixel)'
      }
      sourceNumbers={[2]}
      onCitationSelect={select}
    />,
  )
  const citation = screen.getByRole('button', { name: '근거 2번 보기' })
  citation.focus()
  fireEvent.click(citation)
  expect(select).toHaveBeenCalledWith(2)
  view.rerender(
    <AnswerContent
      content={
        '# 제목\n\n### 소제목\n\n**근거 [2]**\n\n![설명](https://example.org/pixel)'
      }
      sourceNumbers={[2]}
      onCitationSelect={select}
    />,
  )
  expect(document.activeElement).toBe(citation)
})

it('storage errors keep the active transcript usable and recover after space is freed', () => {
  const get = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
    throw new Error('storage blocked')
  })
  const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('quota')
  })
  const { result } = renderHook(() => useConversations())
  expect(result.current.storageError).toBe(true)
  act(() =>
    result.current.setMessages([{ id: 'u', role: 'user', content: '질문' }]),
  )
  expect(result.current.messages[0].content).toBe('질문')
  get.mockRestore()
  set.mockRestore()
  act(() =>
    result.current.setMessages((messages) => [
      ...messages,
      { id: 'a', role: 'assistant', content: '답변' },
    ]),
  )
  expect(result.current.storageError).toBe(false)
  expect(
    JSON.parse(localStorage.getItem(CONVERSATION_STORAGE_KEY)!).conversations[0]
      .messages,
  ).toHaveLength(2)
  const id = result.current.activeId!
  act(() => result.current.selectConversation(null))
  expect(result.current.messages).toEqual([])
  act(() => result.current.selectConversation(id))
  act(() => result.current.deleteConversation(id))
  act(() => result.current.restoreConversation())
  expect(result.current.messages).toHaveLength(2)
  act(() => result.current.deleteConversation(id))
  act(() => result.current.dismissDeleted())
  expect(result.current.deleted).toBeNull()
})

it('names source location groups and renders page ranges, sections and zero-based table coordinates', async () => {
  const { default: CitationLocation } =
    await import('../../src/components/CitationLocation')
  const view = render(
    <CitationLocation
      location={{
        page: 1,
        page_end: 3,
        section_path: ['장', '', '절'],
        table_id: 'T1',
        row: 0,
        column: 1,
      }}
    />,
  )
  const location = screen.getByRole('group', { name: '원문 위치' })
  expect(location.textContent).toBe('1–3쪽장 › 절표 T1행 1열 2')
  view.rerender(
    <CitationLocation compact location={{ page: 2, page_end: 2 }} />,
  )
  expect(screen.getByText('2쪽')).toBeTruthy()
  view.rerender(<CitationLocation location={{ section_path: [], row: 0 }} />)
  expect(screen.getByText('행 1')).toBeTruthy()
  view.rerender(<CitationLocation location={{ column: 0 }} />)
  expect(screen.getByText('열 1')).toBeTruthy()
  view.rerender(<CitationLocation location={{ block_id: 'block-only' }} />)
  expect(screen.queryByRole('group', { name: '원문 위치' })).toBeNull()
})
