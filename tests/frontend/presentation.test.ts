import { describe, expect, it } from 'vitest'
import * as display from '../../src/chat/presentation'
import type { Message } from '../../src/types/chat'
import type { GenerationInfo } from '../../src/api/rag'

describe('evidence presentation', () => {
  it('counts stable documents instead of repeated chunks', () => {
    expect(display.uniqueDocumentCount()).toBe(0)
    expect(display.uniqueDocumentCount([])).toBe(0)
    expect(
      display.uniqueDocumentCount([
        { chunk_id: '1', document_id: 'same' },
        { chunk_id: '2', doc_id: 'same' },
        { chunk_id: '3', relative_path: 'path' },
        { chunk_id: '4', source_path: 'source' },
        { chunk_id: '5', file_name: 'name', institution: 'PNU' },
        { chunk_id: '6', file_name: 'name' },
        { chunk_id: '7' },
      ]),
    ).toBe(6)
  })
  it('shows readable previews, score fallbacks, and capped source locations', () => {
    expect(display.cleanPreview(' a\n b  ')).toBe('a b')
    expect(display.resultScoreLabel({ chunk_id: 'a', score: -2.56 })).toBe(
      '2.56',
    )
    expect(
      display.resultScoreLabel({
        chunk_id: 'a',
        scores: { ignored: null, bm25: 5 },
      }),
    ).toBe('5.00')
    expect(display.resultScoreLabel({ chunk_id: 'a' })).toBe('—')
    expect(
      display.resultLocationSummary({ chunk_id: 'a', page: 2 }, 3).hiddenCount,
    ).toBe(0)
    expect(
      display.resultLocationSummary(
        {
          chunk_id: 'a',
          locations: [{ page: 1 }, { page: 2 }],
          location_count: 5,
        },
        1,
      ).hiddenCount,
    ).toBe(4)
  })
  it('uses retrieval metadata only when its structure is usable', () => {
    expect(display.retrievalSummary()).toBeNull()
    expect(display.retrievalSummary({ strategy: 'hybrid' })).toBe('hybrid')
    expect(display.retrievalSummary({ mode: 'bm25' })).toBe('bm25')
    expect(display.retrievalSummary({ fusion: 'rrf' })).toBe('rrf')
    expect(display.retrievalSummary({ strategy: '', result_count: 3 })).toBe(
      '검색 결과 3개',
    )
    expect(display.retrievalSummary({ results: 2 })).toBe('검색 결과 2개')
    expect(display.retrievalSummary({ top_k: 8 })).toBe('검색 결과 8개')
    expect(display.retrievalSummary({})).toBeNull()
    for (const value of [
      undefined,
      'bad',
      [],
      { removed_count: 'bad' },
      { removed_count: 2, kept_count: 'bad' },
    ])
      expect(
        display.contextDeduplicationSummary({ context_deduplication: value }),
      ).toBeNull()
    expect(display.contextDeduplicationSummary()).toBeNull()
    expect(
      display.contextDeduplicationSummary({
        context_deduplication: { removed_count: 2, kept_count: 3 },
      }),
    ).toBe('중복 근거 2개 제거 · 최종 3개')
  })
  it.each([
    ['academic_year:2026', '2026학년도'],
    ['semester:2', '2학기'],
    ['round:1', '1차'],
    ['date:2026-09-10', '2026-09-10'],
    ['month_day:09-10', '9월 10일'],
    ['month_day:bad', 'bad'],
    ['time_minutes:75', '01:15'],
    ['time_minutes:bad', 'time_minutes:bad'],
    ['amount_krw:1000', '1,000원'],
    ['amount_krw:bad', 'bad원'],
    ['percent:30', '30%'],
    ['quantity:2:명', '2명'],
    ['phone:051', '전화번호 051'],
    ['unknown:value', 'unknown:value'],
  ])('formats evidence mismatch value %s', (input, expected) =>
    expect(display.criticalValueLabel(input)).toBe(expected),
  )
  it('explains supported, missing, and unverified claims', () => {
    expect(display.claimValidationLabel({ text: 'x', supported: true })).toBe(
      '근거 확인',
    )
    expect(
      display.claimValidationLabel({
        text: 'x',
        supported: false,
        validation_reason: 'model_abstention',
      }),
    ).toContain('답변 보류')
    expect(
      display.claimValidationLabel({
        text: 'x',
        supported: false,
        validation_reason: 'critical_value_mismatch',
      }),
    ).toBe('핵심 값을 근거에서 확인하지 못함')
    expect(
      display.claimValidationLabel({
        text: 'x',
        supported: false,
        validation_reason: 'critical_value_mismatch',
        missing_critical_values: ['percent:3'],
      }),
    ).toContain('3%')
    expect(
      display.claimValidationLabel({
        text: 'x',
        supported: false,
        validation_reason: 'low_lexical_overlap',
      }),
    ).toBe('검색 근거와 연결 부족')
    expect(display.claimValidationLabel({ text: 'x', supported: false })).toBe(
      '근거 부족',
    )
    expect(display.traceSummary()).toBeNull()
    expect(display.traceSummary({ stages: [] })).toBeNull()
    expect(
      display.traceSummary({
        stages: [{ id: 'x', state: 'ready', duration_ms: 2000 }],
      }),
    ).toBe('1단계 · 2.0초')
    expect(
      display.traceSummary({ stages: [{ id: 'x', state: 'ready' }] }),
    ).toBe('처리 단계 1개')
  })
  it('separates missing evidence, abstention and filtered claims', () => {
    const message: Message = { id: '1', role: 'assistant', content: 'x' }
    expect(display.getSupportedClaimCount(message)).toBe(0)
    expect(display.weakAnswerHint({ ...message, status: 'error' })).toBeNull()
    expect(display.weakAnswerHint(message)).toContain('검색된 근거가 없습니다')
    const withSource = { ...message, results: [{ chunk_id: '1' }] }
    expect(display.weakAnswerHint(withSource)).toBeNull()
    expect(
      display.weakAnswerHint({
        ...withSource,
        claims: [{ text: 'x', supported: true }],
      }),
    ).toBeNull()
    expect(
      display.weakAnswerHint({
        ...withSource,
        claims: [
          {
            text: 'x',
            supported: false,
            validation_reason: 'model_abstention',
          },
        ],
      }),
    ).toContain('판단했습니다')
    expect(
      display.weakAnswerHint({
        ...withSource,
        claims: [{ text: 'x', supported: false }],
      }),
    ).toContain('최종 답변에서 제외')
  })
  it('describes model fallback without hiding failures', () => {
    expect(display.generationModelDisplayName()).toBe('안전 응답')
    expect(display.generationModelDisplayName('gemini-3.5-flash-lite')).toBe(
      'Gemini 3.5 Flash Lite',
    )
    expect(display.generationModelDisplayName('local-model')).toBe(
      'local-model',
    )
    expect(display.generationFallbackHint()).toContain('다른 경로')
    expect(display.generationFallbackHint({} as GenerationInfo)).toContain(
      '다른 경로',
    )
    for (const [reason, text] of [
      ['timeout', '제한 시간'],
      ['http_429', '요청 한도'],
      ['network_error', '연결이 불안정'],
      ['http_500', '호출에 실패'],
    ]) {
      expect(
        display.generationFallbackHint({
          requested: 'frontier',
          used: 'local',
          model: 'local',
          fallback_reason: `gemini:gemini-3.1-flash-lite:${reason}`,
        }),
      ).toContain(text)
    }
  })
})
