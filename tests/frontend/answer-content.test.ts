import assert from 'node:assert/strict'
import { test } from 'vitest'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import AnswerContent from '../../src/components/AnswerContent.tsx'

function render(content: string) {
  return renderToStaticMarkup(
    createElement(AnswerContent, {
      content,
      sourceNumbers: [1, 2],
      onCitationSelect: () => {},
    }),
  )
}

test('lists, emphasis, tables and real citations render as semantic elements', () => {
  const html = render(
    '## 신청 절차\n\n1. **신청** [1]\n2. 확인 [99]\n\n| 문서 | 출처 |\n| --- | --- |\n| 안내 | [2] |',
  )
  assert.match(html, /<ol>/)
  assert.match(html, /<strong>신청<\/strong>/)
  assert.match(html, /<table>/)
  assert.match(html, /aria-label="근거 1번 보기"/)
  assert.match(html, /aria-label="근거 2번 보기"/)
  assert.doesNotMatch(html, /aria-label="근거 99번 보기"/)
  assert.match(html, /\[99\]/)
})

test('code examples and literal external links do not become citation controls', () => {
  const html = render('`[1]`\n\n[링크 [2]](https://example.org)')
  assert.match(html, /<code>\[1\]<\/code>/)
  assert.doesNotMatch(html, /class="citation-token"/)
})

test('model output cannot inject executable HTML or load tracking images', () => {
  const html = render(
    '<script>alert(1)</script>\n\n[열기](javascript:alert%281%29)\n\n![이미지](https://example.org/tracker)',
  )
  assert.doesNotMatch(
    html,
    /<script|javascript:|<img|src="https:\/\/example.org\/tracker/,
  )
})
