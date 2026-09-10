import {
  Children,
  cloneElement,
  createContext,
  createElement,
  isValidElement,
  useContext,
  useMemo,
} from 'react'
import type { ComponentProps, ReactNode } from 'react'
import Markdown from 'react-markdown'
import type { Components, ExtraProps } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Heading, Root } from 'mdast'
import { visit } from 'unist-util-visit'

// Generated Markdown can start at any heading level. Keep its hierarchy under
// the page heading without skipping levels for screen-reader navigation.
function normalizeAnswerHeadings() {
  return (tree: Root) => {
    const headings: Heading[] = []
    visit(tree, 'heading', (node) => {
      headings.push(node)
    })
    const minimum = Math.min(...headings.map((node) => node.depth))
    let previous = 1
    for (const node of headings) {
      node.depth = Math.min(
        6,
        previous + 1,
        node.depth - minimum + 2,
      ) as Heading['depth']
      previous = node.depth
    }
  }
}

function withCitations(
  children: ReactNode,
  onSelect: (source: number) => void,
  sources: Set<number>,
): ReactNode {
  return Children.map(children, (child) => {
    if (typeof child === 'string') {
      return child.split(/(\[\d+\])/g).map((part, index) => {
        const match = part.match(/^\[(\d+)\]$/)
        return match && sources.has(Number(match[1])) ? (
          <button
            aria-label={`근거 ${match[1]}번 보기`}
            className="citation-token"
            key={index}
            onClick={() => onSelect(Number(match[1]))}
            type="button"
          >
            {match[1]}
          </button>
        ) : (
          part
        )
      })
    }
    if (
      isValidElement<{ children?: ReactNode; node?: { tagName?: string } }>(
        child,
      ) &&
      child.type !== 'code' &&
      child.type !== 'a' &&
      !['a', 'code', 'pre'].includes(child.props.node?.tagName ?? '')
    ) {
      return cloneElement(
        child,
        {},
        withCitations(child.props.children, onSelect, sources),
      )
    }
    return child
  })
}

const CitationContext = createContext<{
  sources: Set<number>
  onSelect: (source: number) => void
} | null>(null)

function CitationText({ children }: { children: ReactNode }) {
  const { sources, onSelect } = useContext(CitationContext)!
  return withCitations(children, onSelect, sources)
}

function AnswerHeading({ node, children }: ComponentProps<'h2'> & ExtraProps) {
  return createElement(
    node!.tagName,
    {},
    <CitationText>{children}</CitationText>,
  )
}

// Stable component identities preserve focus and selection while status refreshes.
const markdownComponents: Components = {
  p: ({ children }) => (
    <p>
      <CitationText>{children}</CitationText>
    </p>
  ),
  li: ({ children }) => (
    <li>
      <CitationText>{children}</CitationText>
    </li>
  ),
  td: ({ children }) => (
    <td>
      <CitationText>{children}</CitationText>
    </td>
  ),
  th: ({ children }) => (
    <th>
      <CitationText>{children}</CitationText>
    </th>
  ),
  h2: AnswerHeading,
  h3: AnswerHeading,
  h4: AnswerHeading,
  h5: AnswerHeading,
  h6: AnswerHeading,
  table: ({ children }) => (
    <div
      aria-label="답변 표"
      className="answer-table"
      role="region"
      tabIndex={0}
    >
      <table>{children}</table>
    </div>
  ),
  a: ({ href, children }) => (
    <a href={href} rel="noreferrer" target="_blank">
      {children}
    </a>
  ),
  img: ({ alt }) => <span>{alt}</span>,
}

export default function AnswerContent({
  content,
  onCitationSelect,
  sourceNumbers,
}: {
  content: string
  onCitationSelect: (source: number) => void
  sourceNumbers: number[]
}) {
  const markdown = useMemo(
    () => (
      <Markdown
        remarkPlugins={[remarkGfm, normalizeAnswerHeadings]}
        skipHtml
        components={markdownComponents}
      >
        {content}
      </Markdown>
    ),
    [content],
  )
  return (
    <CitationContext.Provider
      value={{ sources: new Set(sourceNumbers), onSelect: onCitationSelect }}
    >
      <div className="answer-content">{markdown}</div>
    </CitationContext.Provider>
  )
}
