import { afterEach, beforeEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
  configurable: true,
  value() {
    this.setAttribute('open', '')
    this.setAttribute('aria-modal', 'true')
  },
})
Object.defineProperty(HTMLDialogElement.prototype, 'close', {
  configurable: true,
  value() {
    this.removeAttribute('open')
    this.removeAttribute('aria-modal')
  },
})
Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
  configurable: true,
  value() {},
})
Object.defineProperty(HTMLElement.prototype, 'scrollTo', {
  configurable: true,
  value(options: ScrollToOptions) {
    this.scrollTop = options.top ?? 0
  },
})
Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  value: vi.fn().mockReturnValue({ matches: false }),
})
Object.defineProperty(navigator, 'clipboard', {
  configurable: true,
  value: { writeText: vi.fn().mockResolvedValue(undefined) },
})
Object.defineProperty(HTMLFormElement.prototype, 'requestSubmit', {
  configurable: true,
  value() {
    this.dispatchEvent(
      new Event('submit', { bubbles: true, cancelable: true }),
    )
  },
})

beforeEach(() => {
  localStorage.clear()
  let queue = Promise.resolve<unknown>(undefined)
  Object.defineProperty(navigator, 'locks', {
    configurable: true,
    value: {
      request: vi.fn((_name: string, run: () => unknown) => {
        const next = queue.then(run)
        queue = next.catch(() => {})
        return next
      }),
    },
  })
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})
