import {
  AlertTriangle,
  CheckCircle2,
  LoaderCircle,
  RefreshCw,
  ServerOff,
} from 'lucide-react'
import {
  getPipelineStages,
  type HealthResponse,
  type PipelineStage,
} from '../api/rag'

type PipelineStatusProps = {
  health: HealthResponse | null
  error?: string | null
  refreshing?: boolean
  onRefresh: () => void
}

function normalizedState(stage: PipelineStage) {
  return stage.state.toLowerCase()
}

function stageClass(stage: PipelineStage) {
  const state = normalizedState(stage)
  if (['ready', 'healthy', 'available', 'configured', 'complete'].includes(state)) {
    return 'is-ready'
  }
  if (['building', 'loading', 'running', 'pending', 'checking'].includes(state)) {
    return 'is-loading'
  }
  if (['degraded', 'fallback', 'partial'].includes(state)) {
    return 'is-degraded'
  }
  return 'is-offline'
}

function stageLabel(stage: PipelineStage) {
  const state = normalizedState(stage)
  if (['ready', 'healthy', 'available', 'configured', 'complete'].includes(state)) {
    return '준비됨'
  }
  if (['building', 'loading', 'running', 'pending', 'checking'].includes(state)) {
    return '준비 중'
  }
  if (['degraded', 'fallback', 'partial'].includes(state)) {
    return '제한적'
  }
  if (state === 'unknown') {
    return '확인 필요'
  }
  return '사용 불가'
}

export default function PipelineStatus({
  health,
  error,
  refreshing = false,
  onRefresh,
}: PipelineStatusProps) {
  const stages = getPipelineStages(health)
  const ready = health?.ready === true
  const degraded =
    ready &&
    (health?.status?.toLowerCase() === 'degraded' ||
      stages.some((stage) => stageClass(stage) === 'is-degraded'))

  return (
    <div className="pipeline-status">
      <div
        className={`pipeline-summary ${
          !health && refreshing
            ? 'is-loading'
            : !health || !ready
              ? 'is-offline'
              : degraded
                ? 'is-degraded'
                : 'is-ready'
        }`}
      >
        {!health && refreshing ? (
          <LoaderCircle className="loading-icon" size={16} />
        ) : !health ? (
          <ServerOff size={16} />
        ) : !ready ? (
          <AlertTriangle size={16} />
        ) : degraded ? (
          <AlertTriangle size={16} />
        ) : (
          <CheckCircle2 size={16} />
        )}
        <div>
          <strong>
            {!health && refreshing
              ? '상태 확인 중'
              : !health
              ? 'API 연결 안 됨'
              : !ready
                ? '파이프라인 준비 안 됨'
                : degraded
                  ? '제한적으로 사용 가능'
                  : '검색 파이프라인 준비됨'}
          </strong>
          <span>
            {error ??
              (typeof health?.chunk_count === 'number'
                ? `${health.chunk_count.toLocaleString()}개 chunk`
                : '서버 상태 기준')}
          </span>
        </div>
        <button
          aria-label="파이프라인 상태 새로고침"
          disabled={refreshing}
          onClick={onRefresh}
          type="button"
        >
          {refreshing ? (
            <LoaderCircle className="loading-icon" size={15} />
          ) : (
            <RefreshCw size={15} />
          )}
        </button>
      </div>

      {stages.length > 0 && (
        <ul className="pipeline-stage-list">
          {stages.map((stage) => (
            <li className={stageClass(stage)} key={stage.id} title={stage.detail}>
              <span>{stage.label}</span>
              <small>{stageLabel(stage)}</small>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
