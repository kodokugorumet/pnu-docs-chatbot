import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import axe from 'axe-core'
import App from '../../src/App'
import * as api from '../../src/api/rag'
import {
  health,
  response,
  seed,
  storedAnswer,
  deferred,
  sources,
} from './fixtures'
import type { ChatResponse, HealthResponse } from '../../src/api/rag'

vi.mock('../../src/api/rag', async (original) => ({
  ...(await original<typeof import('../../src/api/rag')>()),
  getHealth: vi.fn(),
  getInstitutions: vi.fn(),
  chat: vi.fn(),
  unloadLocalModel: vi.fn(),
}))
const getHealth = vi.mocked(api.getHealth)
const getInstitutions = vi.mocked(api.getInstitutions)
const chat = vi.mocked(api.chat)
const unload = vi.mocked(api.unloadLocalModel)

beforeEach(() => {
  getHealth.mockReset().mockResolvedValue(health())
  getInstitutions
    .mockReset()
    .mockResolvedValue([
      '전체 기관',
      '부산대학교',
      '한국거래소',
      '금융감독원',
      '한국은행',
    ])
  chat.mockReset().mockResolvedValue(response())
  unload
    .mockReset()
    .mockResolvedValue({ ok: true, state: 'unloaded', released: true })
})
async function boot() {
  const view = render(<App />)
  await waitFor(() => expect(getHealth).toHaveBeenCalled())
  await waitFor(() =>
    expect(screen.queryByText('검색 서비스에 연결하고 있어요.')).toBeNull(),
  )
  return view
}
function input(value: string) {
  fireEvent.change(screen.getByRole('textbox', { name: '질문 입력' }), {
    target: { value },
  })
}
async function ask(value = '휴학 질문') {
  input(value)
  fireEvent.submit(
    screen.getByRole('textbox', { name: '질문 입력' }).closest('form')!,
  )
  await waitFor(() =>
    expect(screen.queryByRole('button', { name: '답변 생성 중지' })).toBeNull(),
  )
}
function settings() {
  fireEvent.click(screen.getByRole('button', { name: '설정 열기' }))
  return within(screen.getByRole('dialog', { name: '답변 설정' }))
}
function closeSettings() {
  fireEvent.click(screen.getByRole('button', { name: '완료' }))
}

it('starts from recommendations, sends the selected institution and opens the exact citation', async () => {
  await boot()
  fireEvent.click(screen.getByRole('link', { name: '질문 입력으로 건너뛰기' }))
  expect(document.activeElement).toBe(
    screen.getByRole('textbox', { name: '질문 입력' }),
  )
  fireEvent.click(screen.getByRole('button', { name: /학교 생활휴학은/ }))
  expect(
    (screen.getByRole('textbox', { name: '질문 입력' }) as HTMLTextAreaElement)
      .value,
  ).toContain('부산대학교')
  fireEvent.click(screen.getByRole('button', { name: '질문 보내기' }))
  await screen.findByRole('button', { name: '출처 2개' })
  expect(chat.mock.calls[0][0].institution).toBe('부산대학교')
  fireEvent.click(screen.getByRole('button', { name: '근거 1번 보기' }))
  const dialog = screen.getByRole('dialog', { name: '근거 문서' })
  await waitFor(() => expect(document.activeElement?.id).toBe('source-1'))
  expect(within(dialog).getByText('학적 안내')).toBeTruthy()
  const tab = within(dialog).getByRole('tab', { name: '문서' })
  fireEvent.keyDown(tab, { key: 'ArrowRight' })
  expect(
    within(dialog)
      .getByRole('tab', { name: '위치' })
      .getAttribute('aria-selected'),
  ).toBe('true')
  fireEvent.keyDown(within(dialog).getByRole('tab', { name: '위치' }), {
    key: 'End',
  })
  fireEvent.keyDown(within(dialog).getByRole('tab', { name: '위치' }), {
    key: 'ArrowLeft',
  })
  fireEvent.keyDown(within(dialog).getByRole('tab', { name: '문서' }), {
    key: 'Home',
  })
  fireEvent.keyDown(within(dialog).getByRole('tab', { name: '검증' }), {
    key: 'Tab',
  })
  expect(within(dialog).getByText('검색 근거와 연결 부족')).toBeTruthy()
  fireEvent.click(within(dialog).getByRole('tab', { name: '문서' }))
  fireEvent.click(within(dialog).getByRole('tab', { name: '위치' }))
  fireEvent.click(
    within(dialog).getByRole('button', { name: '근거 패널 닫기' }),
  )
  fireEvent.click(screen.getByRole('button', { name: '근거 문서 열기' }))
  fireEvent(
    screen.getByRole('dialog', { name: '근거 문서' }),
    new Event('cancel', { bubbles: true }),
  )
  fireEvent.click(screen.getByRole('button', { name: '복사' }))
  await screen.findByRole('button', { name: '복사 완료' })
  expect(navigator.clipboard.writeText).toHaveBeenCalledWith(response().answer)
  fireEvent.click(screen.getByRole('button', { name: '다시 답변' }))
  await waitFor(() => expect(chat).toHaveBeenCalledTimes(2))
  expect(
    screen.getByLabelText('대화 내용').querySelectorAll('.user'),
  ).toHaveLength(1)
})

it('persists real history, changes conversations and restores deleted transcripts', async () => {
  await boot()
  await ask()
  fireEvent.click(screen.getByRole('button', { name: '새 대화' }))
  await ask('다른 질문')
  fireEvent.click(screen.getByRole('button', { name: '휴학 질문' }))
  expect(
    (screen.getByRole('combobox', { name: '검색 기관' }) as HTMLSelectElement)
      .value,
  ).toBe('전체 기관')
  fireEvent.click(screen.getByRole('button', { name: '휴학 질문 대화 삭제' }))
  expect(screen.queryByRole('heading', { name: '신청 안내' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: '되돌리기' }))
  expect(screen.getByRole('heading', { name: '신청 안내' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '다른 질문 대화 삭제' }))
  fireEvent.click(screen.getByRole('button', { name: '알림 닫기' }))
  expect(screen.queryByText('대화를 삭제했어요.')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: '사이드바 접기' }))
  fireEvent.click(screen.getByRole('button', { name: '사이드바 펼치기' }))
  fireEvent.click(screen.getByRole('button', { name: '대화 목록 열기' }))
  fireEvent.click(
    within(screen.getByRole('dialog', { name: '대화 목록' })).getByRole(
      'button',
      { name: '대화 목록 닫기' },
    ),
  )
  fireEvent.click(screen.getByRole('button', { name: '대화 목록 열기' }))
  fireEvent(
    screen.getByRole('dialog', { name: '대화 목록' }),
    new Event('cancel', { bubbles: true }),
  )
})

it('uses chosen model, role, parser, retrieval and evidence options in the request', async () => {
  await boot()
  const dialog = settings()
  fireEvent.change(dialog.getByRole('combobox', { name: '내 역할 선택' }), {
    target: { value: 'custom' },
  })
  fireEvent.change(dialog.getByRole('textbox', { name: '역할 직접 입력' }), {
    target: { value: ' 대학원생 ' },
  })
  fireEvent.change(dialog.getByRole('combobox', { name: '답변 생성 제공자' }), {
    target: { value: 'local' },
  })
  fireEvent.change(dialog.getByRole('combobox', { name: '로컬 모델' }), {
    target: { value: 'local-b' },
  })
  fireEvent.change(dialog.getByRole('combobox', { name: '검색 파싱 버전' }), {
    target: { value: 'baseline' },
  })
  await waitFor(() =>
    expect(getInstitutions).toHaveBeenLastCalledWith(
      'baseline',
      expect.anything(),
    ),
  )
  fireEvent.change(dialog.getByRole('combobox', { name: '검색 방식' }), {
    target: { value: 'kure_hybrid' },
  })
  fireEvent.click(dialog.getByRole('button', { name: /확장/ }))
  closeSettings()
  await ask()
  expect(chat.mock.calls[0][0]).toMatchObject({
    role: '대학원생',
    provider: 'local',
    model: 'local-b',
    parser_profile: 'baseline',
    retrieval_mode: 'kure_hybrid',
    top_k: 12,
  })
  const other = settings()
  fireEvent.change(other.getByRole('combobox', { name: '답변 생성 제공자' }), {
    target: { value: 'frontier' },
  })
  fireEvent.change(other.getByRole('combobox', { name: 'Gemini 모델' }), {
    target: { value: 'frontier-b' },
  })
  fireEvent.change(other.getByRole('combobox', { name: '내 역할 선택' }), {
    target: { value: 'pnu-student' },
  })
  fireEvent.change(other.getByRole('combobox', { name: '검색 방식' }), {
    target: { value: 'snowflake_hybrid' },
  })
  closeSettings()
  await ask('다음 질문')
  expect(chat.mock.calls[1][0]).toMatchObject({
    role: 'pnu-student',
    model: 'frontier-b',
    retrieval_mode: 'snowflake_hybrid',
  })
})

it('retries errors with the original request and handles unknown failures', async () => {
  chat.mockRejectedValueOnce(
    new api.RagApiError('일시적 실패', {
      code: 'temporary',
      retryable: true,
      requestId: 'request-1',
    }),
  )
  await boot()
  await ask()
  expect(screen.getByText(/요청 ID request-1/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '다시 시도' }))
  await screen.findByRole('button', { name: '출처 2개' })
  expect(chat.mock.calls[1][0]).toEqual(chat.mock.calls[0][0])
  chat.mockRejectedValueOnce(new Error('unexpected'))
  await ask('실패')
  expect(
    screen.getByText('요청을 처리하지 못했습니다. 다시 시도해 주세요.'),
  ).toBeTruthy()
})

it('does not transmit blank or over-limit questions', async () => {
  await boot()
  await ask('   ')
  expect(chat).not.toHaveBeenCalled()
  await ask('가'.repeat(1001))
  expect(chat).not.toHaveBeenCalled()
  expect(screen.getByText('질문은 1,000자 이내로 입력해 주세요.')).toBeTruthy()
})

it('allows drafting offline, reconnects, and preserves malformed-status error detail', async () => {
  getHealth.mockRejectedValueOnce(new Error('offline'))
  getInstitutions.mockRejectedValueOnce(new Error('offline'))
  await boot()
  input('초안')
  fireEvent.submit(
    screen.getByRole('textbox', { name: '질문 입력' }).closest('form')!,
  )
  expect(chat).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: '다시 연결' }))
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: '질문 보내기' }) as HTMLButtonElement)
        .disabled,
    ).toBe(false),
  )
  expect(
    (screen.getByRole('textbox', { name: '질문 입력' }) as HTMLTextAreaElement)
      .value,
  ).toBe('초안')
  const dialog = settings()
  fireEvent.click(dialog.getByText('검색 서비스 연결 상태'))
  getHealth.mockRejectedValueOnce('unknown')
  fireEvent.click(
    dialog.getByRole('button', { name: '파이프라인 상태 새로고침' }),
  )
  await dialog.findByText('API 상태를 확인할 수 없습니다.')
  fireEvent.click(dialog.getByRole('button', { name: '설정 닫기' }))
  fireEvent.click(screen.getByRole('button', { name: '답변 모델 설정' }))
  fireEvent(
    screen.getByRole('dialog', { name: '답변 설정' }),
    new Event('cancel', { bubbles: true }),
  )
})

it('shows long-running progress, aborts the request and lets the user keep a next draft', async () => {
  await boot()
  const pending = deferred<ChatResponse>()
  chat.mockImplementation((_request, signal) => {
    signal?.addEventListener('abort', () =>
      pending.reject(
        new api.RagApiError('취소했습니다', {
          code: 'cancelled',
          retryable: false,
        }),
      ),
    )
    return pending.promise
  })
  vi.useFakeTimers()
  input('긴 요청')
  fireEvent.click(screen.getByRole('button', { name: '질문 보내기' }))
  await act(async () => {
    await vi.advanceTimersByTimeAsync(21000)
  })
  expect(screen.getByText('답변에 시간이 조금 더 걸리고 있어요.')).toBeTruthy()
  input('다음 초안')
  fireEvent.click(screen.getByRole('button', { name: '답변 생성 중지' }))
  await act(async () => {})
  expect(screen.getByText('요청 취소됨')).toBeTruthy()
  expect(
    (screen.getByRole('textbox', { name: '질문 입력' }) as HTMLTextAreaElement)
      .value,
  ).toBe('다음 초안')
})

it('aborts pending requests on unmount and ignores late health responses', async () => {
  const view = await boot()
  const pending = deferred<ChatResponse>()
  chat.mockReturnValue(pending.promise)
  input('요청')
  fireEvent.click(screen.getByRole('button', { name: '질문 보내기' }))
  const signal = chat.mock.calls[0][1]!
  view.unmount()
  expect(signal.aborted).toBe(true)
  await act(async () => pending.resolve(response()))
  const late = deferred<HealthResponse>()
  getHealth.mockReturnValueOnce(late.promise)
  const another = render(<App />)
  await waitFor(() => expect(getHealth).toHaveBeenCalledTimes(2))
  another.unmount()
  await act(async () => late.resolve(health()))
})

it('unloads local models, prevents duplicate unloads, and reports unload failures', async () => {
  await boot()
  const dialog = settings()
  const pending = deferred<api.LocalModelUnloadResponse>()
  unload.mockReturnValueOnce(pending.promise)
  fireEvent.click(dialog.getByRole('button', { name: /메모리에서 내리기/ }))
  expect(unload).toHaveBeenCalledOnce()
  await act(async () =>
    pending.resolve({ ok: true, state: 'unloaded', released: true }),
  )
  unload.mockRejectedValueOnce(new Error('해제 실패'))
  fireEvent.click(dialog.getByRole('button', { name: /메모리에서 내리기/ }))
  await dialog.findByText('해제 실패')
  unload.mockRejectedValueOnce('bad')
  fireEvent.click(dialog.getByRole('button', { name: /메모리에서 내리기/ }))
  await dialog.findByText('로컬 모델을 메모리에서 내리지 못했습니다.')
})

it('reports clipboard denial and dismisses transient copy messages', async () => {
  await boot()
  await ask()
  vi.mocked(navigator.clipboard.writeText).mockRejectedValueOnce(
    new Error('denied'),
  )
  fireEvent.click(screen.getByRole('button', { name: '복사' }))
  await screen.findByText('복사하지 못했어요. 답변을 선택해서 복사해 주세요.')
  vi.useFakeTimers()
  fireEvent.click(screen.getByRole('button', { name: '알림 닫기' }))
  fireEvent.click(screen.getByRole('button', { name: '복사' }))
  await act(async () => {})
  expect(screen.getByRole('button', { name: '복사 완료' })).toBeTruthy()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2700)
  })
  expect(screen.getByRole('button', { name: '복사' })).toBeTruthy()
})

it('keeps reading position and scrolls to latest with the reduced-motion preference', async () => {
  await boot()
  await ask()
  const stream = screen.getByLabelText('대화 내용')
  Object.defineProperties(stream, {
    scrollHeight: { configurable: true, value: 2000 },
    clientHeight: { configurable: true, value: 500 },
    scrollTop: { configurable: true, writable: true, value: 100 },
  })
  fireEvent.scroll(stream)
  input('읽는 중 초안')
  expect(stream.scrollTop).toBe(100)
  vi.mocked(window.matchMedia).mockReturnValue({
    matches: true,
  } as MediaQueryList)
  const scroll = vi.spyOn(stream, 'scrollTo')
  fireEvent.click(screen.getByRole('button', { name: '최신 답변으로 이동' }))
  expect(scroll).toHaveBeenCalledWith({ top: 2000, behavior: 'instant' })
  stream.scrollTop = 100
  fireEvent.scroll(stream)
  vi.mocked(window.matchMedia).mockReturnValue({
    matches: false,
  } as MediaQueryList)
  fireEvent.click(screen.getByRole('button', { name: '최신 답변으로 이동' }))
  expect(scroll).toHaveBeenLastCalledWith({ top: 2000, behavior: 'smooth' })
  stream.scrollTop = 1500
  fireEvent.scroll(stream)
  expect(
    screen.queryByRole('button', { name: '최신 답변으로 이동' }),
  ).toBeNull()
})

describe('stored answers with varied evidence', () => {
  it.each([
    { results: [], claims: [] },
    {
      results: [{ chunk_id: 'x' }],
      claims: undefined,
      generation: {
        requested: 'auto' as const,
        used: 'local',
        fallback_reason: 'other',
      },
      resolvedRole: { id: 'general', label: '일반', requested: 'other' },
    },
    {
      results: [
        {
          chunk_id: 'x',
          relative_path: 'fallback',
          source_path: 'path',
          preview: 'x',
          locations: [],
          download_url: 'https://example.org/file',
        },
      ],
      claims: [
        {
          text: '보류',
          supported: false,
          validation_reason: 'model_abstention',
          citations: [{ page: 1 }],
        },
      ],
    },
    {
      results: [{ chunk_id: '', source_path: 'legacy' }],
      claims: [
        {
          text: '미확인',
          supported: false,
          source_numbers: [1],
          citations: [{ source_number: 1, page: 1 }],
        },
      ],
      resolvedRole: {
        id: 'pnu-student',
        label: '학생',
        requested: 'pnu-student',
      },
    },
  ])('shows usable empty states and optional metadata %#', async (variant) => {
    seed([{ id: 'u', role: 'user', content: '저장' }, storedAnswer(variant)])
    await boot()
    fireEvent.click(screen.getByRole('button', { name: '근거 문서 열기' }))
    const dialog = within(screen.getByRole('dialog', { name: '근거 문서' }))
    fireEvent.click(dialog.getByRole('tab', { name: '검증' }))
    fireEvent.click(dialog.getByRole('tab', { name: '위치' }))
    fireEvent.click(dialog.getByRole('tab', { name: '문서' }))
    expect(screen.getByRole('dialog', { name: '근거 문서' })).toBeTruthy()
  })
  it('handles missing request info and deleted or unavailable institutions', async () => {
    seed(
      [
        { id: 'u', role: 'user', content: '저장' },
        storedAnswer({
          request: undefined,
          results: sources.map((source) => ({
            ...source,
            source_number: undefined,
            chunk_id: '',
          })),
        }),
      ],
      null,
    )
    await boot()
    fireEvent.click(screen.getByRole('button', { name: '저장된 대화' }))
    fireEvent.click(screen.getByRole('button', { name: '출처 2개' }))
    expect(screen.getByRole('dialog', { name: '근거 문서' })).toBeTruthy()
  })
})

it('passes automated accessibility rules on the welcome, settings and answer views', async () => {
  const view = await boot()
  const options = { rules: { 'color-contrast': { enabled: false } } }
  expect((await axe.run(view.container, options)).violations).toEqual([])
  settings()
  expect(
    (await axe.run(screen.getByRole('dialog', { name: '답변 설정' }), options))
      .violations,
  ).toEqual([])
  closeSettings()
  await ask()
  expect((await axe.run(view.container, options)).violations).toEqual([])
})

it('warns on storage quota failures while keeping the current answer usable', async () => {
  vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('quota')
  })
  await boot()
  await ask()
  expect(screen.getByText(/이 브라우저에 대화를 저장할 수 없어요/)).toBeTruthy()
  expect(screen.getByRole('button', { name: '복사' })).toBeTruthy()
})

it('keeps archived requests with no sources usable and falls back from retired institutions', async () => {
  seed(
    [
      storedAnswer({
        results: undefined,
        request: {
          question: '이전 질문',
          provider: 'auto',
          parser_profile: 'cascade',
          retrieval_mode: 'bm25',
          institution: '폐지 기관',
        },
      }),
    ],
    null,
  )
  await boot()
  fireEvent.click(screen.getByRole('button', { name: '저장된 대화' }))
  expect(
    (screen.getByRole('combobox', { name: '검색 기관' }) as HTMLSelectElement)
      .value,
  ).toBe('전체 기관')
  chat.mockResolvedValueOnce(
    response({
      claims: undefined,
      parser_profile: undefined,
      cited_answer: '옛 답변',
    }),
  )
  fireEvent.click(screen.getByRole('button', { name: '다시 답변' }))
  await screen.findByText('옛 답변')
  expect(chat.mock.calls[0][0].top_k).toBe(8)
  expect(chat.mock.calls[0][0].institution).toBe('폐지 기관')
})

it('ignores rejected requests after unmount without modifying the saved transcript', async () => {
  const view = await boot()
  const pending = deferred<ChatResponse>()
  chat.mockReturnValueOnce(pending.promise)
  input('진행')
  fireEvent.click(screen.getByRole('button', { name: '질문 보내기' }))
  const saved = localStorage.getItem('pnu-docs.conversations.v1')
  view.unmount()
  await act(async () => pending.reject(new Error('late failure')))
  expect(localStorage.getItem('pnu-docs.conversations.v1')).toBe(saved)
  expect(getHealth).toHaveBeenCalledOnce()
})

it('prevents same-frame double sends, navigation, deletion and model unloads during a request', async () => {
  await boot()
  await ask()
  const dialog = settings()
  const unloadButton = dialog.getByRole('button', { name: /메모리에서 내리기/ })
  closeSettings()
  const send = screen.getByRole('button', { name: '질문 보내기' }),
    newChat = screen.getByRole('button', { name: '새 대화' }),
    remove = screen.getByRole('button', { name: '휴학 질문 대화 삭제' })
  const pending = deferred<ChatResponse>()
  chat.mockReturnValueOnce(pending.promise)
  input('진행 중')
  act(() => {
    fireEvent.click(send)
    fireEvent.click(send)
    fireEvent.click(newChat)
    fireEvent.click(remove)
  })
  expect(chat).toHaveBeenCalledTimes(2)
  expect(screen.getByRole('button', { name: '휴학 질문' })).toBeTruthy()
  await act(async () => pending.resolve(response()))
  settings()
  const unloading = deferred<api.LocalModelUnloadResponse>()
  unload.mockReturnValueOnce(unloading.promise)
  act(() => {
    fireEvent.click(unloadButton)
    fireEvent.click(unloadButton)
  })
  expect(unload).toHaveBeenCalledOnce()
  await act(async () =>
    unloading.resolve({ ok: true, state: 'unloaded', released: true }),
  )
})

it('does not force scrolling when a pending answer arrives while older text is being read', async () => {
  await boot()
  const pending = deferred<ChatResponse>()
  chat.mockReturnValueOnce(pending.promise)
  input('긴 질문')
  fireEvent.click(screen.getByRole('button', { name: '질문 보내기' }))
  const stream = screen.getByLabelText('대화 내용')
  Object.defineProperties(stream, {
    scrollHeight: { configurable: true, value: 2000 },
    clientHeight: { configurable: true, value: 500 },
    scrollTop: { configurable: true, writable: true, value: 100 },
  })
  fireEvent.scroll(stream)
  await act(async () => pending.resolve(response()))
  await new Promise((resolve) => setTimeout(resolve, 30))
  expect(stream.scrollTop).toBe(100)
})

const retrievalModes = (id: string) => [{ id, ready: true, model_loaded: true }]
it.each([
  {
    name: 'unavailable defaults',
    status: health({
      default_provider: 'local',
      default_parser_profile: 'challenger',
      default_retrieval_mode: 'bm25',
      providers: [{ id: 'local', available: false }],
      parser_profiles: [
        {
          id: 'cascade',
          ready: true,
          retrieval_modes: retrievalModes('snowflake_hybrid'),
        },
      ],
    }),
    parser: 'cascade',
    mode: 'snowflake_hybrid',
  },
  {
    name: 'missing defaults',
    status: health({
      default_provider: undefined,
      default_parser_profile: undefined,
      default_retrieval_mode: undefined,
      parser_profiles: [{ id: 'cascade', ready: true }],
      retrieval_modes: retrievalModes('bm25'),
    }),
    parser: 'cascade',
    mode: 'bm25',
  },
  {
    name: 'not ready',
    status: health({
      ready: false,
      parser_profiles: [{ id: 'cascade', ready: false }],
      retrieval_modes: [],
    }),
    parser: 'cascade',
    mode: 'bm25',
  },
  {
    name: 'no ready retrieval modes',
    status: health({
      parser_profiles: [{ id: 'cascade', ready: true }],
      retrieval_modes: [],
    }),
    parser: 'cascade',
    mode: 'bm25',
  },
])('handles capability fallback: $name', async ({ status, parser, mode }) => {
  getHealth.mockResolvedValue(status)
  await boot()
  const dialog = settings()
  expect(
    (
      dialog.getByRole('combobox', {
        name: '검색 파싱 버전',
      }) as HTMLSelectElement
    ).value,
  ).toBe(parser)
  expect(
    (dialog.getByRole('combobox', { name: '검색 방식' }) as HTMLSelectElement)
      .value,
  ).toBe(mode)
  expect(
    (
      dialog.getByRole('combobox', {
        name: '답변 생성 제공자',
      }) as HTMLSelectElement
    ).value,
  ).toBe('auto')
})

it('periodically refreshes status and moves away from unavailable providers, profiles and retrieval modes', async () => {
  getHealth.mockResolvedValue(health({ default_provider: 'local' }))
  vi.useFakeTimers()
  render(<App />)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1)
  })
  const refresh = async (status: HealthResponse) => {
    getHealth.mockResolvedValue(status)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(30001)
    })
  }
  await refresh(
    health({
      providers: [],
      parser_profiles: [
        {
          id: 'baseline',
          ready: true,
          retrieval_modes: retrievalModes('snowflake_hybrid'),
        },
        { id: 'cascade', ready: false },
      ],
    }),
  )
  let dialog = settings()
  expect(
    (
      dialog.getByRole('combobox', {
        name: '답변 생성 제공자',
      }) as HTMLSelectElement
    ).value,
  ).toBe('auto')
  expect(
    (
      dialog.getByRole('combobox', {
        name: '검색 파싱 버전',
      }) as HTMLSelectElement
    ).value,
  ).toBe('baseline')
  expect(
    (dialog.getByRole('combobox', { name: '검색 방식' }) as HTMLSelectElement)
      .value,
  ).toBe('snowflake_hybrid')
  closeSettings()
  await refresh(
    health({
      parser_profiles: [
        {
          id: 'baseline',
          ready: true,
          retrieval_modes: retrievalModes('bm25'),
        },
      ],
    }),
  )
  dialog = settings()
  expect(
    (dialog.getByRole('combobox', { name: '검색 방식' }) as HTMLSelectElement)
      .value,
  ).toBe('bm25')
  closeSettings()
  await refresh(
    health({
      parser_profiles: [{ id: 'baseline', ready: true, retrieval_modes: [] }],
      retrieval_modes: [],
    }),
  )
  await refresh(health({ parser_profiles: [{ id: 'cascade', ready: false }] }))
  dialog = settings()
  expect(
    (
      dialog.getByRole('combobox', {
        name: '검색 파싱 버전',
      }) as HTMLSelectElement
    ).value,
  ).toBe('cascade')
})

it('selects available models when configured defaults or previous choices disappear', async () => {
  getHealth.mockResolvedValue(
    health({
      providers: ['local', 'frontier'].map((id) => ({
        id,
        available: true,
        default_model: 'gone',
        models: [
          { id: 'gone', available: false },
          { id: 'available', available: true },
        ],
      })),
    }),
  )
  await boot()
  const dialog = settings()
  fireEvent.change(dialog.getByRole('combobox', { name: '답변 생성 제공자' }), {
    target: { value: 'local' },
  })
  expect(
    (dialog.getByRole('combobox', { name: '로컬 모델' }) as HTMLSelectElement)
      .value,
  ).toBe('available')
  fireEvent.change(dialog.getByRole('combobox', { name: '답변 생성 제공자' }), {
    target: { value: 'frontier' },
  })
  expect(
    (dialog.getByRole('combobox', { name: 'Gemini 모델' }) as HTMLSelectElement)
      .value,
  ).toBe('available')
  fireEvent.change(dialog.getByRole('combobox', { name: 'Gemini 모델' }), {
    target: { value: 'available' },
  })
  getHealth.mockResolvedValue(health({ providers: [] }))
  fireEvent.click(
    dialog.getByRole('button', { name: '파이프라인 상태 새로고침' }),
  )
  await waitFor(() =>
    expect(
      (
        dialog.getByRole('combobox', {
          name: '답변 생성 제공자',
        }) as HTMLSelectElement
      ).value,
    ).toBe('auto'),
  )
})

it('keeps restored institution labels and outgoing filters consistent after an institution is retired', async () => {
  seed([
    storedAnswer({
      request: {
        question: '저장 질문',
        institution: '폐지 기관',
        provider: 'auto',
        parser_profile: 'cascade',
        retrieval_mode: 'bm25',
      },
    }),
  ])
  await boot()
  expect(
    (screen.getByRole('combobox', { name: '검색 기관' }) as HTMLSelectElement)
      .value,
  ).toBe('전체 기관')
  await ask('새 질문')
  expect(chat.mock.calls[0][0].institution).toBeUndefined()
})
