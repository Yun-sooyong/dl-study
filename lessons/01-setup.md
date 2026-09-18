# 실습 환경 만들기 — Colab과 내 컴퓨터

> ⏱ 30분 · 설치 안내 · 선수 지식: 없음

**목표:** 이 코스의 모든 코드를 돌릴 수 있는 환경을 두 가지 방법으로 준비합니다. **Colab**(설치 없음, 무료 GPU)과 **내 컴퓨터**(설치 필요, 제한 없음). 처음에는 Colab으로 충분하고, 5부에서 내 프로젝트를 할 때 로컬 환경이 필요해집니다.

## 방법 1: Google Colab (권장 시작점)

브라우저만 있으면 됩니다. 각 레슨의 **▶ Colab에서 실행하기** 버튼이 이 코스의 노트북을 Colab에서 엽니다.

1. 구글 계정으로 로그인합니다.
2. 레슨에 "GPU 필요"라고 적혀 있으면 메뉴 **런타임 → 런타임 유형 변경 → T4 GPU**를 고릅니다.
3. 셀을 위에서부터 `Shift+Enter`로 실행합니다. 첫 셀에서 라이브러리를 내려받느라 1~2분 걸릴 수 있습니다.
4. 코드를 고쳐 실험한 것을 남기려면 **파일 → Drive에 사본 저장**. 원본 노트북은 읽기 전용입니다.

알아 둘 것:

- **세션은 사라집니다.** 90분 정도 손을 안 대거나 12시간이 지나면 런타임이 초기화되어 변수·다운로드·학습 결과가 모두 없어집니다. 오래 걸리는 학습은 중간 결과를 `torch.save`로 Drive에 저장하세요(`from google.colab import drive; drive.mount("/content/drive")`).
- **GPU 사용량 제한.** 무료 계정은 하루 몇 시간 정도의 GPU만 줍니다. 1·2부는 CPU로 충분하니 GPU는 3·4부에 아끼세요. 다 쓰면 **런타임 → 런타임 연결 해제 및 삭제**로 반납합니다.
- **GPU가 잡혔는지 확인:** 아래 셀이 `cuda`를 출력해야 합니다.

```python
import torch, sys
print("Python", sys.version.split()[0], "| PyTorch", torch.__version__)
print("device:", "cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"| 메모리 {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")
```

**코드 읽기**

- `torch.cuda.is_available()` — CUDA(NVIDIA GPU 계산 플랫폼)를 쓸 수 있는지. `False`면 GPU 런타임을 고르지 않았거나 드라이버가 없는 것입니다. 이 값이 이 코스 모든 코드의 `device` 변수를 결정합니다.
- `torch.cuda.get_device_properties(0).total_memory` — GPU 메모리 총량. T4는 약 16GB(표시로는 15GB). [내 프로젝트 시작하기](#50-your-project)의 메모리 계산과 비교할 기준값입니다.

## 방법 2: 내 컴퓨터에 설치하기

Colab보다 번거롭지만, 시간 제한이 없고 내 파일을 바로 쓸 수 있으며 코드를 편집기에서 다룰 수 있습니다. GPU가 없어도 1·2부와 3·4부의 작은 모델은 CPU로 돌아갑니다(느릴 뿐).

### 1) Python 설치

[python.org](https://www.python.org/downloads/)에서 **3.10~3.12** 버전을 설치합니다. Windows에서는 설치 화면의 **"Add python.exe to PATH"** 체크를 잊지 마세요. 터미널에서 확인:

```bash
python --version
```

### 2) 가상환경 만들기

프로젝트마다 라이브러리를 따로 설치하는 격리된 공간입니다. 왜 필요한가: 프로젝트 A는 PyTorch 2.4, B는 2.6이 필요할 때 시스템 전체에 하나만 깔면 충돌합니다. 가상환경은 이 문제를 폴더 단위로 해결합니다.

```bash
python -m venv .venv
```

활성화 (터미널을 새로 열 때마다):

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

프롬프트 앞에 `(.venv)`가 붙으면 성공입니다.

### 3) PyTorch 설치

[pytorch.org](https://pytorch.org/get-started/locally/)의 선택 표에서 내 OS·GPU에 맞는 명령을 복사합니다. GPU가 없거나 Mac이면:

```bash
pip install torch torchvision
```

NVIDIA GPU가 있는 Windows/Linux라면 CUDA 버전이 포함된 명령(예: `--index-url https://download.pytorch.org/whl/cu124`)을 써야 GPU가 잡힙니다. 최신 NVIDIA 드라이버가 설치되어 있어야 합니다.

### 4) 나머지 라이브러리

```bash
pip install transformers peft datasets accelerate scikit-learn pandas matplotlib jupyter
```

### 5) 확인

```bash
python -c "import torch, transformers, sklearn; print(torch.__version__, torch.cuda.is_available())"
```

`True`가 나오면 GPU까지 준비된 것입니다. `False`여도 CPU로 코스를 진행할 수 있습니다.

### 6) 노트북 실행

이 코스의 노트북은 저장소에 있습니다.

```bash
git clone https://github.com/Yun-sooyong/dl-study.git
cd dl-study
jupyter notebook notebooks
```

브라우저가 열리면 레슨 노트북을 골라 실행합니다. Jupyter 대신 **VS Code**(Python·Jupyter 확장 설치)에서 `.ipynb` 파일을 열어도 됩니다. 코드를 파일로 다루기 시작하면 VS Code 쪽이 편합니다.

## 자주 만나는 문제

| 증상 | 원인과 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | 가상환경이 활성화되지 않았거나 다른 Python에 설치됨. 프롬프트의 `(.venv)`를 확인하고 `python -m pip install ...`로 설치 |
| `torch.cuda.is_available()`가 `False` | CPU 전용 PyTorch를 설치했거나 드라이버가 없음. pytorch.org에서 CUDA 포함 명령으로 재설치 |
| `CUDA out of memory` | GPU 메모리 부족. 배치 크기를 절반으로, 그래도 안 되면 모델을 작은 것으로 ([내 프로젝트 시작하기](#50-your-project)) |
| Colab이 느리거나 `Runtime disconnected` | 세션 만료. 다시 연결하고 처음부터 실행. 긴 학습은 중간 저장 |
| 다운로드가 매우 느림 | Hugging Face 허브 속도 문제. 다시 시도하거나 `HF_TOKEN`을 설정하면 빨라지는 경우가 있음 |
| Windows에서 경로 오류 | 역슬래시 경로 문제. 경로를 `r"C:\..."` 또는 `"C:/..."`로 쓰기 |
| 한글 그래프 라벨이 □□로 나옴 | 한글 폰트가 없음. 라벨을 영어로 쓰거나 폰트를 설치. 이 코스의 그래프는 영어 라벨을 씁니다 |

## 어디에 무엇이 저장되나

- **내려받은 모델·데이터:** Hugging Face 캐시 폴더(`~/.cache/huggingface`). 같은 모델은 한 번만 받습니다. 수십 GB가 쌓일 수 있으니 디스크가 부족하면 이 폴더를 정리하세요.
- **torchvision 데이터셋:** 코드의 `"data"` 폴더(현재 작업 폴더 아래).
- **학습 결과:** 코드에서 `torch.save`나 `save_pretrained`로 지정한 경로. Colab에서는 세션이 끝나면 사라지므로 Drive에 저장합니다.

## 핵심 정리

- 시작은 Colab. 설치 없이 GPU까지 쓸 수 있고, 세션이 초기화된다는 점만 기억하면 됩니다.
- 내 컴퓨터에서는 **가상환경 → PyTorch(OS/GPU에 맞게) → 나머지 라이브러리** 순서로 설치합니다.
- `torch.cuda.is_available()`로 GPU를 확인하는 것이 모든 실습의 첫 줄입니다.
- 오류의 대부분은 "가상환경 미활성화", "CPU용 PyTorch", "메모리 부족" 셋 중 하나입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. Colab에서 두 시간 걸리는 학습을 돌릴 때 무엇을 대비해야 하나요?</summary>

세션이 끊길 수 있으므로 일정 스텝마다 가중치를 Google Drive에 저장(체크포인트)하고, 끊기면 그 지점부터 다시 시작하도록 코드를 짭니다.

</details>

<details><summary>Q2. 가상환경을 쓰는 이유는?</summary>

프로젝트마다 필요한 라이브러리 버전이 달라 충돌하기 때문입니다. 가상환경은 프로젝트 폴더 단위로 독립된 설치 공간을 만듭니다.

</details>

<details><summary>Q3. 내 노트북에 GPU가 없습니다. 이 코스를 할 수 있나요?</summary>

네. 1·2부와 대부분의 3·4부 코드는 CPU에서 돌아갑니다(시간이 더 걸릴 뿐). GPU가 꼭 필요한 레슨은 Colab을 쓰면 됩니다.

</details>
