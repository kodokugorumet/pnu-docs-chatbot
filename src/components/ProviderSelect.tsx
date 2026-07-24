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
  disabled?: boolean
}

export default function ProviderSelect({
  value,
  providers,
  onChange,
  disabled = false,
}: ProviderSelectProps) {
  const selected = providers.find((provider) => provider.id === value)
  const detail =
    value === 'auto'
      ? '서버가 사용 가능한 생성 경로를 선택합니다.'
      : selected?.model
        ? `${selected.label} · ${selected.model}`
        : selected?.reason ?? `${providerDisplayName(value)} 상태를 확인하고 있습니다.`

  return (
    <label className="provider-select">
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
      <small id="provider-select-detail">{detail}</small>
    </label>
  )
}
