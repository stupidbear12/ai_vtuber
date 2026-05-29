# emeth RVC 보이스 모델 폴더

이 폴더에 RVC 음성 모델 파일을 넣어주세요.

## 필요한 파일
- `emeth.pth`   : 학습된 RVC 모델 (필수)
- `emeth.index` : 피처 인덱스 파일 (선택, 음색 정확도 향상)

## 무료 모델 다운로드 사이트
- https://weights.gg  (한국어 모델 포함)
- https://huggingface.co (rvc-models 검색)
- Discord: AI Hub, AI Cover 관련 서버

## emeth 모델 직접 학습하는 법
1. emeth 캐릭터 목소리 녹음 (최소 10분, 30분 권장)
   - 조용한 환경, 마이크로 깨끗하게 녹음
   - 다양한 감정과 속도로 읽기
2. RVC WebUI 설치: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
3. WebUI에서 학습 탭 → 목소리 파일 업로드 → 학습 시작
4. 학습 완료된 .pth / .index 파일을 이 폴더에 복사

## 모델 없이 테스트하는 법
config_tts.py에서 RVC_ENABLED = False 로 변경하면
Edge TTS 음성(ko-KR-SunHiNeural)이 그대로 출력됩니다.
