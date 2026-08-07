#!/bin/zsh
# 3.1-flash-lite RPD 리셋(16:00 KST) 후 D44 n=3 잔여 재채점 자동 재개.
# resume-gen-eval-20260806.sh와 같은 프로브 방식.
cd ~/project/pnu-docs-chatbot
API_KEY=$(grep '^GEMINI_API_KEY' .env | head -1 | cut -d= -f2 | tr -d ' ')
MODEL=gemini-3.1-flash-lite

probe() {
  curl -s -o /dev/null -w "%{http_code}" --max-time 20 \
    -X POST "https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent" \
    -H "Content-Type: application/json" -H "x-goog-api-key: ${API_KEY}" \
    -d '{"contents":[{"parts":[{"text":"1"}]}],"generationConfig":{"maxOutputTokens":1}}'
}

while [ "$(date +%H%M)" -lt 1555 ]; do sleep 300; done
echo "=== $(date +%H:%M) 프로브 시작 (${MODEL}) ==="
until code=$(probe); [ "$code" = "200" ]; do
  echo "$(date +%H:%M) probe=${code} — 대기"
  sleep 600
done
echo "=== $(date +%H:%M) 리셋 확인 — 재채점 ==="

PY=.parser-tools/venvs/embedding/bin/python
for i in 2 3; do
  echo "=== rejudge run${i} ==="
  GEMINI_MODEL=$MODEL GEMINI_FALLBACK_MODELS=$MODEL \
    $PY scripts/evaluate_grant_generation.py --rejudge --sleep 5 \
    --out processed/eval/20260807-gen-nrf-default-run${i}.jsonl
done
echo "=== ALL DONE $(date +%H:%M) ==="
