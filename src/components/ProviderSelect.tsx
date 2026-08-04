import { Cpu, LoaderCircle, Power } from 'lucide-react'
import { providerDisplayName } from '../api/rag'
import type {
  GenerationProvider,
  ProviderCapability,
} from '../api/rag'

type ProviderSelectProps = {
  value: GenerationProvider
  providers: ProviderCapability[]
  onChange: (provider: GenerationProvider) => void
  model: string
  onModelChange: (model: string) => void
  frontierModel: string
  onFrontierModelChange: (model: string) => void
  onUnloadLocalModel: () => void
  unloadingLocalModel?: boolean
  unloadError?: string | null
  activeRequestProvider?: GenerationProvider | null
  disabled?: boolean
}

export default function ProviderSelect({
  value,
  providers,
  onChange,
  model,
  onModelChange,
  frontierModel,
  onFrontierModelChange,
  onUnloadLocalModel,
  unloadingLocalModel = false,
  unloadError = null,
  activeRequestProvider = null,
  disabled = false,
}: ProviderSelectProps) {
  const selected = providers.find((provider) => provider.id === value)
  const localProvider = providers.find((provider) => provider.id === 'local')
  const frontierProvider = providers.find((provider) => provider.id === 'frontier')
  const localModels = localProvider?.models ?? []
  const frontierModels = frontierProvider?.models ?? []
  const hasAvailableLocalModel = localModels.some((candidate) => candidate.available)
  const hasAvailableFrontierModel = frontierModels.some(
    (candidate) => candidate.available,
  )
  const selectedModel = localModels.find((candidate) => candidate.id === model)
  const selectedFrontierModel = frontierModels.find(
    (candidate) => candidate.id === frontierModel,
  )
  const runtimeState = localProvider?.runtimeState ?? 'unknown'
  const loadedModel = localProvider?.loadedModel
  const loadedModelLabel =
    localModels.find((candidate) => candidate.id === loadedModel)?.label ??
    loadedModel
  const runtimeBusy = ['busy', 'loading', 'starting', 'stopping'].includes(
    runtimeState,
  )
  const canUnload =
    localProvider?.unloadSupported === true &&
    (localProvider.workerRunning === true ||
      Boolean(loadedModel) ||
      ['ready', 'loaded', 'busy', 'loading', 'starting'].includes(runtimeState))
  let runtimeDetail = '로컬 모델 실행 상태를 확인할 수 없습니다.'
  if (unloadingLocalModel || runtimeState === 'stopping') {
    runtimeDetail = '로컬 모델을 메모리에서 내리는 중입니다.'
  } else if (runtimeState === 'loaded') {
    runtimeDetail = `${loadedModelLabel ?? '로컬 모델'} · 메모리 사용 중`
  } else if (runtimeState === 'busy') {
    runtimeDetail = `${loadedModelLabel ?? '로컬 모델'} · 답변 생성 중`
  } else if (runtimeState === 'loading' || runtimeState === 'starting') {
    runtimeDetail = '로컬 모델을 메모리로 불러오는 중입니다.'
  } else if (activeRequestProvider === 'local') {
    runtimeDetail = '로컬 모델을 불러오거나 답변을 생성 중입니다.'
  } else if (activeRequestProvider === 'auto') {
    runtimeDetail =
      '자동 선택 답변 생성 중 · 로컬 경로 사용 상태는 완료 후 반영됩니다.'
  } else if (runtimeState === 'ready') {
    runtimeDetail = '로컬 모델 서버가 대기 중입니다.'
  } else if (runtimeState === 'unloaded') {
    runtimeDetail =
      '메모리에서 내려감 · 로컬 경로를 사용하는 다음 질의에서 다시 불러옵니다.'
  } else if (runtimeState === 'external') {
    runtimeDetail = '외부 로컬 모델 서버에서 메모리를 관리하고 있습니다.'
  } else if (runtimeState === 'error') {
    runtimeDetail = '로컬 모델 실행 상태에 문제가 있습니다.'
  }
  let detail = selected?.reason ?? `${providerDisplayName(value)} 상태를 확인하고 있습니다.`
  if (value === 'auto') {
    detail = '서버가 사용 가능한 생성 경로를 선택합니다.'
  } else if (value === 'local') {
    detail = selectedModel
      ? `${selected?.label ?? providerDisplayName(value)} · ${selectedModel.label}`
      : selected?.reason ??
        localModels.find((candidate) => candidate.reason)?.reason ??
        '서버에서 사용 가능한 로컬 모델을 확인하지 못했습니다.'
  } else if (value === 'frontier' && selectedFrontierModel) {
    detail = `${selected?.label ?? providerDisplayName(value)} · ${selectedFrontierModel.label} 우선, 지연 시 다른 Gemini 모델로 전환`
  } else if (selected?.model) {
    detail = `${selected.label} · ${selected.model}`
  }

  return (
    <div className="provider-select">
      <div className="provider-select-controls">
        <label className="provider-select-control">
          <span className="provider-select-label">
            <Cpu size={15} />
            답변 생성
          </span>
          <select
            aria-describedby="provider-select-detail"
            aria-label="답변 생성 제공자"
            disabled={disabled}
            onChange={(event) => onChange(event.target.value as GenerationProvider)}
            value={value}
          >
            <option value="auto">{providerDisplayName('auto')}</option>
            {providers.map((provider) => (
              <option
                disabled={!provider.available}
                key={provider.id}
                value={provider.id}
              >
                {provider.label}
                {!provider.available ? ' (사용 불가)' : ''}
              </option>
            ))}
          </select>
        </label>
        {value === 'local' && (
          <label className="provider-model-select">
            <span>로컬 모델</span>
            <select
              aria-describedby="provider-select-detail"
              aria-label="로컬 모델"
              disabled={disabled || !hasAvailableLocalModel}
              onChange={(event) => onModelChange(event.target.value)}
              value={model}
            >
              {localModels.length === 0 ? (
                <option value="">사용 가능한 모델 없음</option>
              ) : (
                <>
                  {!model && (
                    <option disabled value="">
                      모델 선택
                    </option>
                  )}
                  {localModels.map((candidate) => (
                    <option
                      disabled={!candidate.available}
                      key={candidate.id}
                      title={candidate.reason}
                      value={candidate.id}
                    >
                      {candidate.label}
                      {!candidate.available ? ' (사용 불가)' : ''}
                    </option>
                  ))}
                </>
              )}
            </select>
          </label>
        )}
        {value === 'frontier' && (
          <label className="provider-model-select">
            <span>Gemini 모델</span>
            <select
              aria-describedby="provider-select-detail"
              aria-label="Gemini 모델"
              disabled={disabled || !hasAvailableFrontierModel}
              onChange={(event) => onFrontierModelChange(event.target.value)}
              value={frontierModel}
            >
              {frontierModels.length === 0 ? (
                <option value="">사용 가능한 모델 없음</option>
              ) : (
                <>
                  {!frontierModel && (
                    <option disabled value="">
                      모델 선택
                    </option>
                  )}
                  {frontierModels.map((candidate) => (
                    <option
                      disabled={!candidate.available}
                      key={candidate.id}
                      title={candidate.reason}
                      value={candidate.id}
                    >
                      {candidate.label}
                      {!candidate.available ? ' (사용 불가)' : ''}
                    </option>
                  ))}
                </>
              )}
            </select>
          </label>
        )}
      </div>
      <small id="provider-select-detail">{detail}</small>
      {localProvider && (
        <div className={`local-model-runtime is-${runtimeState}`}>
          <span aria-live="polite">{runtimeDetail}</span>
          {localProvider.unloadSupported === true && (
            <button
              aria-label={
                canUnload
                  ? `${loadedModelLabel ?? '로컬 모델'} 메모리에서 내리기`
                  : '로컬 모델이 메모리에서 내려가 있음'
              }
              disabled={
                !canUnload || disabled || unloadingLocalModel || runtimeBusy
              }
              onClick={onUnloadLocalModel}
              type="button"
            >
              {unloadingLocalModel ? (
                <LoaderCircle className="loading-icon" size={14} />
              ) : (
                <Power size={14} />
              )}
              {unloadingLocalModel
                ? '내리는 중'
                : canUnload
                  ? '메모리에서 내리기'
                  : '메모리에서 내려감'}
            </button>
          )}
        </div>
      )}
      {unloadError && (
        <small className="local-model-runtime-error" role="alert">
          {unloadError}
        </small>
      )}
    </div>
  )
}
