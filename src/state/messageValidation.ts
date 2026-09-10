import {
  GENERATION_PROVIDERS,
  PARSER_PROFILES,
  RETRIEVAL_MODES,
} from '../api/rag'
import type { Message } from '../types/chat'

type Check = (value: unknown) => boolean
type Fields = Record<string, Check>
export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
const text: Check = (value) => typeof value === 'string'
const number: Check = (value) =>
  typeof value === 'number' && Number.isFinite(value)
const boolean: Check = (value) => typeof value === 'boolean'
const oneOf =
  (values: readonly unknown[]): Check =>
  (value) =>
    values.includes(value)
const nullable =
  (check: Check): Check =>
  (value) =>
    value === null || check(value)
const array =
  (check: Check): Check =>
  (value) =>
    Array.isArray(value) && value.every(check)
const fields = (check: Check, ...keys: string[]): Fields =>
  Object.fromEntries(keys.map((key) => [key, check]))
const shape =
  (required: Fields, optional: Fields = {}): Check =>
  (value) =>
    isRecord(value) &&
    Object.entries(required).every(([key, check]) => check(value[key])) &&
    Object.entries(optional).every(
      ([key, check]) => value[key] === undefined || check(value[key]),
    )
const locationFields: Fields = {
  ...fields(nullable(text), 'block_id', 'table_id'),
  ...fields(nullable(number), 'page', 'page_end', 'row', 'column'),
  section_path: nullable(array(text)),
}
const citation = shape(
  {},
  {
    ...locationFields,
    ...fields(
      text,
      'citation_id',
      'chunk_id',
      'document_id',
      'excerpt',
      'excerpt_sha256',
    ),
    ...fields(number, 'source_number', 'excerpt_start', 'excerpt_end'),
    ...fields(nullable(text), 'claim_sha256', 'source_title', 'source_url'),
  },
)
const result = shape(
  { chunk_id: text },
  {
    ...locationFields,
    ...fields(
      text,
      'doc_id',
      'document_id',
      'institution',
      'file_name',
      'source_path',
      'relative_path',
      'preview',
    ),
    ...fields(
      nullable(text),
      'source_title',
      'source_url',
      'download_url',
      'source_host',
      'fetched_at',
      'published_at',
      'category',
    ),
    ...fields(
      number,
      'source_number',
      'chunk_index',
      'char_count',
      'score',
      'location_count',
    ),
    ...fields(array(text), 'table_ids', 'block_ids'),
    page_start: nullable(number),
    section_path: nullable((value) => text(value) || array(text)(value)),
    location: nullable(shape({}, locationFields)),
    locations: array(shape({}, locationFields)),
    locations_truncated: boolean,
    metadata: isRecord,
    scores: (value) =>
      isRecord(value) && Object.values(value).every(nullable(number)),
  },
)
const claim = shape(
  { text, supported: boolean },
  {
    ...fields(number, 'confidence', 'best_score'),
    ...fields(array(text), 'source_ids', 'missing_critical_values'),
    source_numbers: array(number),
    citations: array(citation),
    validation_reason: text,
  },
)
const generation = shape(
  { requested: oneOf(GENERATION_PROVIDERS), used: text },
  {
    model: nullable(text),
    fallback_reason: nullable(text),
    attempts: array(
      shape(
        {},
        {
          ...fields(text, 'provider', 'status'),
          ...fields(nullable(text), 'model', 'error'),
          ...fields(nullable(number), 'elapsed_ms', 'duration_ms'),
        },
      ),
    ),
  },
)
const message = shape(
  { id: text, content: text, role: oneOf(['user', 'assistant']) },
  {
    results: array(result),
    claims: array(claim),
    generation,
    status: oneOf(['search', 'error']),
    parserProfile: oneOf(PARSER_PROFILES),
    durationMs: number,
    resolvedRole: shape(
      { id: text, label: text },
      { requested: nullable(text), institutions: array(text) },
    ),
    trace: shape(
      {},
      {
        request_id: text,
        stages: array(
          shape(
            {},
            {
              ...fields(text, 'id', 'label', 'state', 'status'),
              duration_ms: nullable(number),
              detail: nullable(text),
            },
          ),
        ),
      },
    ),
    retrieval: isRecord,
    error: shape({ code: text, retryable: boolean }, { requestId: text }),
    request: shape(
      {
        question: text,
        provider: oneOf(GENERATION_PROVIDERS),
        parser_profile: oneOf(PARSER_PROFILES),
        retrieval_mode: oneOf(RETRIEVAL_MODES),
      },
      {
        ...fields(text, 'institution', 'role', 'model'),
        top_k: number,
      },
    ),
  },
)

/** Validate every nested field used by rendering before trusting browser storage. */
export function validMessage(value: unknown): value is Message {
  return message(value)
}
