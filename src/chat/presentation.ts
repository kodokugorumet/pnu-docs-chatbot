import { getResultLocations } from '../api/rag'
import type { SearchResult, Claim, ChatTrace, GenerationInfo } from '../api/rag'
import type { Message } from '../types/chat'

export function cleanPreview(value: string) {
  return value.replace(/\s+/g, ' ').trim()
}

export function resultScoreLabel(result: SearchResult) {
  const score =
    typeof result.score === 'number'
      ? result.score
      : Object.values(result.scores ?? {}).find(
          (value): value is number => typeof value === 'number',
        )
  return typeof score === 'number' ? Math.abs(score).toFixed(2) : '—'
}

export function resultLocationSummary(result: SearchResult, limit: number) {
  const locations = getResultLocations(result)
  const reportedCount =
    typeof result.location_count === 'number'
      ? result.location_count
      : locations.length
  return {
    locations: locations.slice(0, limit),
    hiddenCount: Math.max(reportedCount - Math.min(locations.length, limit), 0),
  }
}

export function uniqueDocumentCount(results?: SearchResult[]) {
  if (!results?.length) {
    return 0
  }
  return new Set(
    results.map((result) => {
      const stableId = [
        result.document_id,
        result.doc_id,
        result.relative_path,
        result.source_path,
      ].find((value) => typeof value === 'string' && value.trim())
      if (stableId) {
        return stableId
      }
      if (result.file_name) {
        return `${result.institution ?? ''}:${result.file_name}`
      }
      return result.chunk_id
    }),
  ).size
}

export function retrievalSummary(retrieval?: Record<string, unknown>) {
  if (!retrieval) {
    return null
  }
  const strategy = [retrieval.strategy, retrieval.mode, retrieval.fusion].find(
    (value) => typeof value === 'string',
  )
  if (typeof strategy === 'string' && strategy.trim()) {
    return strategy
  }
  const resultCount = [
    retrieval.result_count,
    retrieval.results,
    retrieval.top_k,
  ].find((value) => typeof value === 'number')
  return typeof resultCount === 'number' ? `검색 결과 ${resultCount}개` : null
}

export function contextDeduplicationSummary(
  retrieval?: Record<string, unknown>,
) {
  const value = retrieval?.context_deduplication
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const diagnostics = value as Record<string, unknown>
  const removedCount = diagnostics.removed_count
  const keptCount = diagnostics.kept_count
  if (typeof removedCount !== 'number' || typeof keptCount !== 'number') {
    return null
  }
  return `중복 근거 ${removedCount}개 제거 · 최종 ${keptCount}개`
}

export function criticalValueLabel(value: string) {
  const [kind, ...parts] = value.split(':')
  const normalized = parts.join(':')
  if (kind === 'academic_year') return `${normalized}학년도`
  if (kind === 'semester') return `${normalized}학기`
  if (kind === 'round') return `${normalized}차`
  if (kind === 'date') return normalized
  if (kind === 'month_day') {
    const [month, day] = normalized.split('-').map(Number)
    return Number.isFinite(month) && Number.isFinite(day)
      ? `${month}월 ${day}일`
      : normalized
  }
  if (kind === 'time_minutes') {
    const minutes = Number(normalized)
    if (Number.isFinite(minutes)) {
      const hour = Math.floor(minutes / 60)
      const minute = minutes % 60
      return `${hour.toString().padStart(2, '0')}:${minute
        .toString()
        .padStart(2, '0')}`
    }
  }
  if (kind === 'amount_krw') {
    const amount = Number(normalized)
    return Number.isFinite(amount)
      ? `${amount.toLocaleString('ko-KR')}원`
      : `${normalized}원`
  }
  if (kind === 'percent') return `${normalized}%`
  if (kind === 'quantity') return normalized.replace(':', '')
  if (kind === 'phone') return `전화번호 ${normalized}`
  return value
}

export function claimValidationLabel(claim: Claim) {
  if (claim.supported) {
    return '근거 확인'
  }
  if (claim.validation_reason === 'model_abstention') {
    return '모델이 근거 부족으로 답변 보류'
  }
  if (claim.validation_reason === 'critical_value_mismatch') {
    const missing = (claim.missing_critical_values ?? [])
      .map(criticalValueLabel)
      .join(', ')
    return missing
      ? `핵심 값을 근거에서 확인하지 못함 · ${missing}`
      : '핵심 값을 근거에서 확인하지 못함'
  }
  if (claim.validation_reason === 'low_lexical_overlap') {
    return '검색 근거와 연결 부족'
  }
  return '근거 부족'
}

export function traceSummary(trace?: ChatTrace) {
  const stages = trace?.stages ?? []
  if (!stages.length) {
    return null
  }
  const totalDuration = stages.reduce(
    (sum, stage) =>
      sum + (typeof stage.duration_ms === 'number' ? stage.duration_ms : 0),
    0,
  )
  return totalDuration > 0
    ? `${stages.length}단계 · ${(totalDuration / 1000).toFixed(1)}초`
    : `처리 단계 ${stages.length}개`
}

export function getSupportedClaimCount(message: Message) {
  return message.claims?.filter((claim) => claim.supported).length ?? 0
}

export function weakAnswerHint(message: Message) {
  if (message.status === 'error') {
    return null
  }

  const claimCount = message.claims?.length ?? 0
  const supportedCount = getSupportedClaimCount(message)
  const sourceCount = message.results?.length ?? 0

  if (sourceCount === 0) {
    return '검색된 근거가 없습니다. 기관 범위나 질문 표현을 바꿔 다시 검색해 보세요.'
  }
  if (claimCount === 0 || supportedCount >= Math.ceil(claimCount * 0.6)) {
    return null
  }
  if (
    message.claims?.every(
      (claim) => claim.validation_reason === 'model_abstention',
    )
  ) {
    return '모델이 제공된 검색 근거만으로 답하기 어렵다고 판단했습니다. 검증 탭에서 판정 내용을 확인할 수 있습니다.'
  }
  return '모델 초안 중 근거가 부족한 문장은 최종 답변에서 제외했습니다. 검증 탭에서 판정 이유를 확인할 수 있습니다.'
}

export function generationModelDisplayName(model?: string | null) {
  const labels: Record<string, string> = {
    'gemini-3.1-flash-lite': 'Gemini 3.1 Flash Lite',
    'gemini-3.5-flash-lite': 'Gemini 3.5 Flash Lite',
  }
  return model ? (labels[model] ?? model) : '안전 응답'
}

export function generationFallbackHint(generation?: GenerationInfo) {
  const reason = generation?.fallback_reason ?? ''
  const geminiFailure = reason.match(
    /(?:^|,)gemini:([^,:]+):(timeout|network_error|http_\d+)/,
  )
  if (!geminiFailure) {
    return '요청한 생성 경로를 사용할 수 없어 다른 경로로 답변했습니다.'
  }

  const failedModel = generationModelDisplayName(geminiFailure[1])
  const usedModel = generationModelDisplayName(generation?.model)
  const failure = geminiFailure[2]
  if (failure === 'timeout') {
    return `${failedModel} 응답이 설정된 제한 시간 안에 오지 않아 ${usedModel}(으)로 자동 전환했습니다.`
  }
  if (failure === 'http_429') {
    return `${failedModel} 무료 API 요청 한도에 도달해 ${usedModel}(으)로 자동 전환했습니다.`
  }
  if (failure === 'network_error') {
    return `${failedModel} 연결이 불안정해 ${usedModel}(으)로 자동 전환했습니다.`
  }
  return `${failedModel} 호출에 실패해 ${usedModel}(으)로 자동 전환했습니다.`
}
