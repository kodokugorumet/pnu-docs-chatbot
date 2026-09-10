# 2단계 보안 레이어 오프라인 평가

평가일: 2026-09-07

## 범위

외부 LLM이나 네트워크 없이 `Context Security Gate`와 `Output Security
Gate` 자체를 결정론적으로 평가했다. 평가 데이터는
`config/security-layer-eval.json`, 실행기는
`scripts/evaluate_security_layers.py`이다.

이 평가는 구현 후 작성한 합성 개발 평가이므로 독립 holdout이나 실제 LLM
종단간 보안 성능을 대신하지 않는다.

## 재현 명령

```powershell
python scripts\evaluate_security_layers.py --cases config\security-layer-eval.json --iterations 5000
python scripts\evaluate_security_layers.py --cases config\security-layer-validation.json --iterations 5000 --corpus-root src\converted_txt --corpus-limit 5000
```

## 결과

### Context Gate

| 지표 | 결과 | 권장 기준 | 판정 |
|---|---:|---:|---|
| 공격 탐지 recall | 100% | 95% (직접), 80% (변형) | 충족 |
| precision | 100% | 참고 지표 | 충족 |
| F1 | 100% | 참고 지표 | 충족 |
| 정상 문서 오탐률 | 0% | 3% 이하 | 충족 |
| 정확한 allow/sanitize/exclude | 100% | 별도 기준 없음 | 충족 |

개발셋은 정상 25건과 공격 36건, 추가 검증셋은 정상 25건과 공격 40건이다.
두 세트에서 직접 공격, 역할 변경, delimiter, Unicode 난독화, citation
조작, 한국어 의미 변형을 모두 탐지했고 정상 문장은 모두 허용했다.

초기 실행은 개발셋 recall 66.67%, FPR 16%, 추가 검증셋 recall 30%, FPR
44%였다. 정상 규정의 서술형과 모델을 대상으로 한 명령형을 구분하도록
규칙을 수정한 뒤 두 세트 모두 recall 100%, FPR 0%를 기록했다. 따라서
최종 수치는 규칙 조정에 사용된 개발·검증 결과이며 독립 외부 holdout
성능으로 해석해서는 안 된다.

### Output Gate

| 지표 | 결과 | 권장 기준 | 판정 |
|---|---:|---:|---|
| 정상 출력 통과율 | 100% (6/6) | 97% 이상 | 잠정 충족 |
| 잘못된 출력 탈출률 | 0% (0/48) | 0% | 잠정 충족 |

unknown chunk, 잘못된 source number, 잘못된 hash, 원문에 없는 excerpt,
claim citation 누락, 잘못된 claims/citations schema, 빈 claim을 각각 6건씩
시험했으며 모두 차단되었다. 표본이 작고 합성 데이터이므로 잠정 결과이다.

### 처리 지연시간

Windows 개발 환경, 반복 2,000회, Context Gate당 청크 8개 조건이다.

| 계층 | 평균 | 중앙값 | p95 | 최대 |
|---|---:|---:|---:|---:|
| Context Gate | 0.1998 ms | 0.1973 ms | 0.2097 ms | 0.5534 ms |
| Output Gate | 0.0077 ms | 0.0074 ms | 0.0088 ms | 0.0576 ms |

규칙 기반 게이트의 자체 지연시간은 권장 기준인 p95 20ms 이하를 충족했다.
이는 전체 HTTP 및 LLM 지연시간을 포함하지 않는다.

## 결론

변환 문서 corpus 584개 구간을 추가 검사했으며 경보 후보는 0건이었다.
정답 라벨이 없는 운영 corpus 검사이므로 이는 FPR 0%를 증명하지 않고,
현재 규칙이 기존 문서에 대량 경보를 만들지 않는다는 보조 지표이다.

결정론적 합성 평가 범위에서는 Context Gate, Output Gate, 처리 지연시간이
설정한 목표를 충족했다. 다만 규칙 기반 평가셋을 조정 과정에서 사용했기
때문에 실제 LLM과 독립적인 외부 holdout까지 포함한 전체 성공을 의미하지
않는다.

최종 평가는 고정된 독립 holdout, 실제 사용 모델, 보안 적용 전후 조건,
반복 생성과 사람 판정을 사용해 Attack Success Rate, Secure Answer Rate,
Clean Answer Preservation을 추가 측정해야 한다.

## 격리된 API 통합 스모크 테스트 (2026-09-08)

운영 인덱스와 분리한 BM25 인덱스에 정상 문서 1건과 문서형 프롬프트
인젝션 1건을 함께 넣고 `/chat` 전체 경로를 호출했다. 질문은 정상 문서와
악성 문서가 모두 검색되도록 같은 주제와 핵심어를 사용했다.

| 확인 항목 | 결과 |
|---|---:|
| 검색 후보 | 2건 |
| Context Gate 허용 | 1건 |
| Context Gate 제외 | 1건 (`high_confidence_injection`) |
| 정상 근거만 최종 컨텍스트에 포함 | 성공 |
| 가짜 날짜(12월 31일) 노출 | 0건 |
| 시스템 프롬프트 노출 | 0건 |
| Output Gate 판정 | `answer` |
| 유효 인용 | 1건 |

최종 답변은 정상 근거의 `2026년 10월 30일`, `소속 학과 사무실`만
포함했다. 이 실행에서는 Gemini 호출이 `network_error`로 실패하여
추출형 fallback이 사용되었다. 따라서 두 게이트와 API 연결은 확인했지만,
Gemini를 포함한 종단간 성공률 표본에는 이 실행을 포함하지 않는다.
