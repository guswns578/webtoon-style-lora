# Webtoon Character Studio

캐릭터별 LoRA로 인물을 생성하고, 인물을 분리해 B컷·C컷으로 배치하는 로컬 제작 보조 도구입니다. 제작자가 대화형 AI 없이 버튼만으로 쓰도록 만들었습니다.

- **B컷**: 인물을 패널 높이의 125%로 크게 배치해 상반신 위주로 보이는 컷 (아래가 잘리는 것은 의도)
- **C컷**: 패널 높이의 33.6%로 작게 배치한 원거리 전신 컷. 얼굴 내부선(눈·코·입)을 생략하고 작은 잔선을 정리

C컷의 생략과 정리는 **모델이 아니라 후처리**가 합니다. 생성은 캐릭터 원본 1장이고, 같은 원본을 B와 C로 각각 합성합니다.

## 구조

```
app.py           Gradio UI (캐릭터 선택, 큐 등록, 검수, 마스크 수정, ZIP 저장)
engine.py        SQLite 직렬 큐, Forge API 호출, 인물 분리, B/C 합성, 얼굴 생략 마스크
forge_control.py Forge Neo 실행·연결 점검 (로컬 주소만 허용)
ui_helpers.py    레이아웃 미리보기, 마스크 편집 값 변환
verify_models.py 모델 파일 크기·SHA256 검사
presets.json     캐릭터별 LoRA·프롬프트·추천 포즈(OpenPose 좌표)·시드
tests/           자동 테스트
```

## 동작 흐름

1. 캐릭터 여러 명을 고르면 하나씩 큐에 들어가 차례대로 생성됩니다. 캐릭터당 후보 1~4개.
2. 추천 포즈, 공통 포즈, 또는 3D·일반 이미지에서 추출한 OpenPose를 씁니다. 3D 렌더는 골격만 추출해 txt2img의 조건으로 쓰며, 원본 픽셀을 이어받는 img2img가 아닙니다. 그래서 3D의 의상 디테일이나 손가락, 무기 접촉은 보존되지 않습니다.
3. 생성 후 얼굴을 검출해 C컷 생략 영역을 자동으로 만듭니다. 실패하면 안내를 띄우고 직접 지정하게 합니다.
4. 인물 영역(파랑)과 생략 영역(주황)을 브러시로 고치면 GPU 재생성 없이 다시 합성합니다. 수정 전 원본은 따로 보존됩니다.
5. 사용자가 검수 승인한 결과를 ZIP으로 저장합니다. 승인은 자동 품질 판정이 아니라 사람이 확인했다는 표시입니다.

## 실행

필요: Windows, NVIDIA GPU, Python 3.11, 별도 설치된 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic)

```bat
install.bat
verify_models.bat
run.bat
```

`models/`(베이스 모델, OpenPose ControlNet, 캐릭터 LoRA 8개)와 `assets/face_detector.onnx`, `assets/foreground.png`는 저장소에 포함하지 않습니다. 얼굴 검출 모델은 [deepghs/anime_face_detection](https://huggingface.co/deepghs/anime_face_detection) (MIT)의 `face_detect_v1.4_n`을 씁니다.

## 테스트

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트 12개가 아래 5개 영역을 검사합니다. 테스트 픽스처(`examples/`의 생성 이미지와 마스크)는 원작 캐릭터가 포함돼 저장소에 넣지 않았습니다.

1. 큐 작업 취소, 무작위 시드 기록, 중단 후 복구 상태
2. B/C 크기와 투명 출력, 마스크 값 보존, 수정 이력, 승인 상태, ZIP 내용
3. 자동 분리가 실패한 원본에 수동 마스크를 적용해 복구
4. 얼굴 검출 실패 안내와 추천 포즈 생성
5. 외부 서버 주소 입력 거부

## 검증 환경

Windows, Python 3.11, Gradio 6.20, Forge Neo 2.27, PyTorch 2.11 + cu130, RTX 5070 12GB. 다른 GPU나 Forge 버전에서 같은 픽셀 결과는 보장하지 않습니다.
