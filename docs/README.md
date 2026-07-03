# 코드 분석 문서

AI VTuber 프로젝트 코드를 분석할 때는 **이 폴더**를 시작점으로 하세요.

| 문서 | 용도 |
|------|------|
| [CODEBASE_MAP.md](./CODEBASE_MAP.md) | **메인** — 모듈 구조, 포트, 호출 관계, 분석 범위 |
| [ANALYSIS_README.md](./ANALYSIS_README.md) | 모듈별 읽기 순서, 핵심 파일 설명 |
| [archive/](./archive/) | 구버전 분석 보고서 (참고용, 현재 코드와 불일치 가능) |

## 분석 범위 한 줄 요약

```
분석 대상  →  modules/{core,chat,live2d,broadcast,voice}  (+ music은 개발중)
분석 제외  →  old/, training/, chatbot/, scripts/, Live2D 모델 에셋
```

로컬 방송 실행: `start-all.bat` → 5모듈 (8000–8004), Ollama 별도 실행.
