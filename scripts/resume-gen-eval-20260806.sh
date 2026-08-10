#!/bin/zsh
# Gemini RPD 리셋 후 생성 평가 자동 재개 — 프로브 방식
# (문서상 midnight PT = 16:00 KST(PDT). 정확한 시각 논쟁을 피하려고
#  15:55부터 실제 호출이 성공할 때까지 10분마다 1회 프로브)
cd ~/project/pnu-docs-chatbot
API_KEY=$(grep '^GEMINI_API_KEY' .env | head -1 | cut -d= -f2 | tr -d ' ')
MODEL=$(grep '^GEMINI_MODEL' .env | head -1 | cut -d= -f2 | tr -d ' ')
MODEL=${MODEL:-gemini-3.5-flash-lite}

probe() {
  curl -s -o /dev/null -w "%{http_code}" --max-time 20 \
    -X POST "https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent" \
    -H "Content-Type: application/json" -H "x-goog-api-key: ${API_KEY}" \
    -d '{"contents":[{"parts":[{"text":"1"}]}],"generationConfig":{"maxOutputTokens":1}}'
}

while [ "$(date +%H%M)" -lt 1555 ]; do sleep 300; done
echo "=== $(date +%H:%M) 프로브 시작 (model=${MODEL}) ==="
until code=$(probe); [ "$code" = "200" ]; do
  echo "$(date +%H:%M) probe=${code} — 대기"
  sleep 600
done
echo "=== $(date +%H:%M) 리셋 확인(200) — 재개 ==="

PY=.parser-tools/venvs/embedding/bin/python
echo "=== rejudge G-base ==="
$PY scripts/evaluate_grant_generation.py --rejudge --sleep 4 --out processed/eval/20260806-gen-base.jsonl
echo "=== rejudge G-esnow ==="
$PY scripts/evaluate_grant_generation.py --rejudge --sleep 4 --out processed/eval/20260806-gen-esnow.jsonl
echo "=== resume G-esnow ==="
$PY scripts/evaluate_grant_generation.py --contexts processed/eval/20260806-ctx-gen-esnow.jsonl --litm-order --sleep 4 --out processed/eval/20260806-gen-esnow.jsonl
echo "=== ALL DONE $(date +%H:%M) ==="
