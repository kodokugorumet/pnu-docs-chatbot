import { Cpu } from 'lucide-react'
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
  disabled?: boolean
}

export default function ProviderSelect({
  value,
  providers,
  onChange,
  model,
  onModelChange,
  disabled = false,
}: ProviderSelectProps) {
  const selected = providers.find((provider) => provider.id === value)
  const localProvider = providers.find((provider) => provider.id === 'local')
  const localModels = localProvider?.models ?? []
  const hasAvailableLocalModel = localModels.some((candidate) => candidate.available)
  const selectedModel = localModels.find((candidate) => candidate.id === model)
  let detail = selected?.reason ?? `${providerDisplayName(value)} 상태를 확인하고 있습니다.`
  if (value === 'auto') {
    detail = '서버가 사용 가능한 생성 경로를 선택합니다.'
  } else if (value === 'local') {
    detail = selectedModel
      ? `${selected?.label ?? providerDisplayName(value)} · ${selectedModel.label}`
      : selected?.reason ??
        localModels.find((candidate) => candidate.reason)?.reason ??
        '서버에서 사용 가능한 로컬 모델을 확인하지 못했습니다.'
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
          <label className="local-model-select">
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
      </div>
      <small id="provider-select-detail">{detail}</small>
    </div>
  )
}
