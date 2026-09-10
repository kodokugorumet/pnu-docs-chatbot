import type { CitationLocation as CitationLocationData } from '../api/rag'

type CitationLocationProps = {
  location: CitationLocationData
  compact?: boolean
}

function pageLabel(location: CitationLocationData) {
  if (location.page == null) {
    return null
  }
  if (location.page_end != null && location.page_end !== location.page) {
    return `${location.page}–${location.page_end}쪽`
  }
  return `${location.page}쪽`
}

export default function CitationLocation({
  location,
  compact = false,
}: CitationLocationProps) {
  const page = pageLabel(location)
  const section = location.section_path?.filter(Boolean).join(' › ')
  const hasLocation =
    page ||
    section ||
    location.table_id ||
    location.row != null ||
    location.column != null

  if (!hasLocation) {
    return null
  }

  return (
    <div
      aria-label="원문 위치"
      role="group"
      className={`citation-location ${compact ? 'is-compact' : ''}`}
    >
      {page && <span>{page}</span>}
      {section && (
        <span className="section-path" title={section}>
          {section}
        </span>
      )}
      {location.table_id && (
        <span className="table-location" title={location.table_id}>
          표 {location.table_id}
        </span>
      )}
      {location.row != null && <span>행 {location.row + 1}</span>}
      {location.column != null && <span>열 {location.column + 1}</span>}
    </div>
  )
}
