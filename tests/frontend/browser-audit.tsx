// Development-only accessibility harness, never included in the production entry.
import { createRoot } from 'react-dom/client'
import axe from 'axe-core'
import App from '../../src/App'
import '../../src/index.css'

createRoot(document.getElementById('root')!).render(<App />)
const button = document.createElement('button')
button.textContent = '접근성 검사 실행'
button.style.cssText =
  'position:fixed;right:12px;bottom:8px;z-index:1000;padding:8px;background:white;color:#111;border:1px solid #111;border-radius:8px'
const report = document.createElement('output')
report.id = 'accessibility-report'
report.hidden = true
async function runAudit() {
  if (button.disabled) return
  button.disabled = true
  await Promise.allSettled(
    document
      .getAnimations()
      .filter(
        (animation) => animation.effect?.getTiming().iterations !== Infinity,
      )
      .map((animation) => animation.finished),
  )
  const result = await axe.run(
    document.querySelector('dialog[open]') ?? document.getElementById('root')!,
  )
  report.textContent = JSON.stringify({
    violations: result.violations,
    passes: result.passes.map((item) => item.id),
    incomplete: result.incomplete,
  })
  button.disabled = false
  button.textContent = `접근성 검사: ${result.violations.length}건`
}
button.addEventListener('click', () => void runAudit())
document.getElementById('root')!.addEventListener('click', () => {
  window.setTimeout(() => void runAudit(), 180)
})
document.addEventListener('keydown', (event) => {
  if (event.key === 'F8') void runAudit()
})
document.body.append(button, report)
