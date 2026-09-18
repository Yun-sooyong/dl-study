# 모델 구조 바꾸기 — CNN, 저장과 불러오기

> ⏱ 50분 · CPU로 충분 (GPU면 더 빠름)

**목표:** 같은 데이터·같은 학습 루프에서 **모델만 갈아끼워** 봅니다. "모델을 수정한다"는 감각을 익히는 레슨입니다.

## 준비 (첫 신경망 레슨과 동일)

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
tf = transforms.ToTensor()
train_dl = DataLoader(datasets.MNIST("data", train=True, download=True, transform=tf), batch_size=64, shuffle=True)
test_dl = DataLoader(datasets.MNIST("data", train=False, download=True, transform=tf), batch_size=256)

def evaluate(model, dl):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(dl.dataset)

def train(model, epochs=2, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1}  loss {loss.item():.4f}  acc {evaluate(model, test_dl):.4f}")
```

## 왜 CNN인가

MLP는 이미지를 784개의 숫자로 **펴버려서** "옆 픽셀끼리 가깝다"는 정보를 버립니다. 합성곱(convolution)은 작은 필터(예: 3×3)를 이미지 전체에 밀면서 적용합니다. 같은 필터를 모든 위치에서 재사용하므로 **파라미터는 적고, 위치가 달라져도 같은 패턴을 찾습니다.**

## 모델을 설계할 때는 shape을 따라가세요

```python
x = torch.randn(1, 1, 28, 28)                       # 가짜 이미지 1장
x = nn.Conv2d(1, 16, kernel_size=3, padding=1)(x);  print(x.shape)  # [1,16,28,28] 채널 1→16
x = nn.MaxPool2d(2)(x);                             print(x.shape)  # [1,16,14,14] 가로세로 절반
x = nn.Conv2d(16, 32, kernel_size=3, padding=1)(x); print(x.shape)  # [1,32,14,14]
x = nn.MaxPool2d(2)(x);                             print(x.shape)  # [1,32,7,7]
print(nn.Flatten()(x).shape)                                        # [1,1568] ← 마지막 Linear의 입력 크기
```

이렇게 가짜 입력을 한 층씩 통과시켜 shape을 확인하는 것이 모델을 고칠 때 가장 확실한 방법입니다.

**코드 읽기**

- `torch.randn(1, 1, 28, 28)` — 내용은 상관없는 가짜 이미지 1장. 모델 설계 단계에서는 진짜 데이터가 필요 없습니다. 모양만 맞으면 됩니다.
- `nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)` — 3×3 필터 **16개**를 이미지 위에 밀면서 적용합니다. 필터 하나가 "출력 채널 하나"를 만들므로 채널이 1 → 16이 됩니다. 각 필터는 학습되면서 가장자리, 곡선, 점 같은 저마다의 패턴 탐지기가 됩니다. 파라미터 수는 `16 × 1 × 3 × 3 + 16 = 160`개뿐입니다. 같은 필터를 784개 위치 모두에 재사용하기 때문입니다.
- `padding=1` — 이미지 가장자리에 0을 한 겹 둘러서 3×3 필터를 적용해도 크기가 28×28로 **유지**되게 합니다. 패딩이 없으면 층을 지날 때마다 2픽셀씩 줄어들어 계산이 번거롭습니다.
- `nn.MaxPool2d(2)` — 2×2 칸마다 최댓값 하나만 남겨 가로세로를 절반으로. 왜: ① 계산량을 1/4로 줄이고, ② "이 근처에 이 패턴이 있다"만 남겨 위치가 조금 어긋나도 같은 특징으로 인식하게 합니다(위치 불변성). 2단계를 거쳐 28 → 14 → 7이 됩니다.
- 두 번째 `Conv2d(16, 32, ...)` — 입력 채널 16(앞 층의 출력), 필터 32개. 앞 층이 찾은 단순 패턴(선)들을 조합해 더 복잡한 패턴(모서리, 고리)을 찾습니다. 층이 깊어질수록 "채널은 늘리고 크기는 줄이는" 것이 CNN의 전형적인 설계입니다.
- `nn.Flatten()` 뒤의 1568 — `32 × 7 × 7`. 마지막 `Linear`의 입력 크기는 이렇게 **계산해서** 넣어야 하고, 층을 하나라도 바꾸면 이 숫자도 바뀝니다. 그래서 가짜 입력을 통과시켜 확인하는 방법이 중요합니다.

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.classifier(self.features(x))

cnn = CNN().to(device)
print("파라미터 수:", sum(p.numel() for p in cnn.parameters()))   # MLP(약 10만)보다 훨씬 적음
train(cnn)                                                        # 그런데 정확도는 더 높음 (98~99%)
```

**코드 읽기**

- `self.features`와 `self.classifier`로 **두 덩어리**로 나눈 이유: 앞부분은 "이미지에서 특징 뽑기", 뒷부분은 "특징으로 분류하기"라는 역할이 다릅니다. 이렇게 나눠 두면 아래에서 `features`만 얼리거나, [전이학습](#25-transfer-learning)에서 `classifier`만 갈아 끼우는 일이 한 줄로 됩니다. 실제 모델들(ResNet의 `fc`, VLM의 `vision_model`/`text_model`)도 같은 이유로 부품을 나눠 이름을 붙입니다.
- `nn.Sequential(nn.Conv2d, nn.ReLU, nn.MaxPool2d, ...)` — "합성곱 → 활성화 → 풀링"이 CNN의 기본 블록입니다. ReLU가 Conv 뒤에 오는 이유는 MLP와 같습니다(비선형성이 없으면 Conv를 아무리 쌓아도 하나의 Conv와 같음).
- `nn.Linear(32 * 7 * 7, 10)` — 위에서 확인한 1568을 곱셈 식으로 적어 두면, 나중에 채널이나 크기를 바꿀 때 어디를 고쳐야 하는지 보입니다.
- 왜 `train`, `evaluate` 함수를 **그대로** 쓰나: 학습 루프는 모델 구조를 전혀 모릅니다. `model(x)`가 `[배치, 10]`을 돌려주기만 하면 됩니다. 모델을 갈아 끼우는 실험이 쉬운 이유이고, 이 레슨의 제목이 "모델 구조 바꾸기"인 이유입니다.
- 파라미터 20,490개(MLP는 101,770개)로 정확도가 더 높습니다. **파라미터 수가 아니라 문제에 맞는 구조**가 성능을 결정한다는 첫 번째 사례입니다.
## 저장하고 불러오기

학습된 모델 = **구조(코드) + 가중치(`state_dict`)**. 파일에는 가중치만 저장하고, 불러올 때는 같은 구조의 모델을 만든 뒤 가중치를 끼웁니다. LLM의 `.safetensors` 파일도 본질은 이것과 같습니다.

```python
torch.save(cnn.state_dict(), "cnn.pt")

cnn2 = CNN().to(device)                          # 새 모델 (랜덤 가중치)
print("불러오기 전:", evaluate(cnn2, test_dl))
cnn2.load_state_dict(torch.load("cnn.pt"))
print("불러온 후:", evaluate(cnn2, test_dl))

for name, p in cnn.state_dict().items():         # state_dict는 그냥 '이름 → 텐서' 사전
    print(f"{name:25s} {tuple(p.shape)}")
```

**코드 읽기**

- `cnn.state_dict()` — 모델의 모든 파라미터(와 BatchNorm의 통계 같은 버퍼)를 `이름 → 텐서` 딕셔너리로 꺼냅니다. 이름은 `features.0.weight`처럼 "속성 이름.Sequential 안의 번호.weight"로 자동 생성됩니다.
- `torch.save(obj, "cnn.pt")` — 파이썬 객체를 파일로 저장. 모델 객체 자체를 저장할 수도 있지만 `state_dict`만 저장하는 것이 관례입니다. 왜: 모델 클래스 코드가 바뀌어도 가중치 파일은 그대로 쓸 수 있고, 파일이 코드에 종속되지 않아 다른 프로젝트로 옮기기 쉽습니다. Hugging Face의 `.safetensors`도 정확히 이 딕셔너리를 저장한 것입니다.
- `cnn2.load_state_dict(torch.load("cnn.pt"))` — 같은 구조의 새 모델에 가중치를 끼워 넣습니다. 이름과 shape이 하나라도 다르면 에러가 납니다. 그것이 안전장치입니다.
- 불러오기 전 정확도 9%(무작위) → 후 97%: **가중치가 곧 모델의 지식**이고, 구조는 그 지식을 담는 그릇이라는 것을 보여줍니다.
## 일부만 학습시키기 (freeze) — 파인튜닝의 기초

`requires_grad=False`로 두면 그 파라미터는 학습되지 않습니다. 뒤에서 배울 LoRA 파인튜닝과 VLM 학습이 전부 이 아이디어 위에 서 있습니다.

```python
for p in cnn2.features.parameters():
    p.requires_grad = False                      # 특징 추출부는 얼림
cnn2.classifier[1].reset_parameters()            # 분류기만 초기화해서 다시 학습
print("학습되는 파라미터:", sum(p.numel() for p in cnn2.parameters() if p.requires_grad))
train(cnn2, epochs=1)
```

**코드 읽기**

- `p.requires_grad = False` — 이 파라미터에 대해서는 기울기를 계산하지 말라는 뜻. `backward()`가 이들을 건너뛰고, 옵티마이저도 `.grad`가 없으므로 갱신하지 않습니다. 메모리와 시간이 줄고, 이미 잘 학습된 부분이 망가지지 않습니다.
- `cnn2.features.parameters()` — 위에서 부품을 나눠 둔 덕분에 특징 추출부만 골라 얼릴 수 있습니다.
- `cnn2.classifier[1].reset_parameters()` — `classifier`는 `Sequential(Flatten, Linear)`이므로 `[1]`이 `Linear`. `reset_parameters()`는 그 층의 가중치를 초기 난수로 되돌립니다. "몸통은 그대로, 머리만 새로"라는 전이학습의 최소 형태를 흉내 낸 것입니다.
- 학습되는 파라미터 15,690개 = `Linear(1568, 10)`의 `1568 × 10 + 10`. 나머지 4,800개(Conv 두 층)는 얼어 있습니다. 1 에폭만에 97.9%가 나오는 것은 얼린 특징이 이미 좋기 때문입니다. 이 실험이 [전이학습](#25-transfer-learning)과 [LLM 파인튜닝](#34-llm-finetune)의 예고편입니다.
## 핵심 정리

- 학습 루프는 그대로 두고 **모델 클래스만 바꾸면** 다른 구조를 실험할 수 있습니다.
- 모델을 고칠 때는 가짜 입력을 한 층씩 통과시키며 shape을 확인하세요.
- CNN은 필터를 모든 위치에서 재사용해 적은 파라미터로 이미지를 더 잘 다룹니다.
- 학습된 모델 = 구조(코드) + 가중치(`state_dict`). 파일에는 가중치를 저장합니다.
- `requires_grad=False`로 일부를 얼리고 나머지만 학습할 수 있습니다. 이것이 파인튜닝의 기초입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. CNN이 MLP보다 파라미터가 훨씬 적은데도 정확도가 높은 이유는?</summary>

3×3 필터 하나를 이미지의 모든 위치에 재사용하기 때문에 파라미터가 적고, '가까운 픽셀끼리 관련 있다', '패턴은 어디에 있든 같은 패턴이다'라는 이미지의 성질을 구조에 미리 담고 있기 때문입니다.

</details>

<details><summary>Q2. <code>load_state_dict</code>가 에러를 내는 가장 흔한 원인은?</summary>

저장할 때와 불러올 때의 모델 구조(층 이름이나 shape)가 다른 경우입니다. 가중치 파일은 '이름 → 텐서' 사전일 뿐이라 구조가 같아야 끼워 넣을 수 있습니다.

</details>

<details><summary>Q3. 특징 추출부를 얼리고 분류기만 학습하면 무엇이 좋은가요?</summary>

학습할 파라미터가 적어 빠르고 메모리를 덜 쓰며, 데이터가 적어도 이미 배운 특징을 망가뜨리지 않습니다. 사전학습 모델을 내 과제에 맞출 때 쓰는 기본 전략입니다.

</details>

## 직접 고쳐보기

1. 데이터를 `datasets.FashionMNIST`로 바꿔보세요 (이름만 바꾸면 됩니다). 옷 사진 분류는 숫자보다 어렵습니다. MLP와 CNN의 차이가 더 벌어지나요?
2. Conv 블록을 하나 더 추가하세요 (`Conv2d(32, 64, ...)`). `Linear`의 입력 크기는 얼마가 되어야 하나요? 위의 shape 추적 방법으로 구하세요.
3. `nn.Dropout(0.3)`을 classifier에, `nn.BatchNorm2d(16)`을 첫 Conv 뒤에 넣어보세요.
4. (도전) [첫 신경망](#21-mlp-mnist) 레슨의 MLP로 FashionMNIST를 학습한 뒤 저장하고, 새 노트북 셀에서 불러와 평가해 보세요.

<details><summary>힌트와 예상 결과 — 먼저 스스로 해 본 뒤 펼치세요</summary>

1. FashionMNIST에서 MLP 약 0.87~0.88, CNN 약 0.90~0.91. 숫자보다 어렵고(옷 종류가 서로 비슷) 격차가 조금 벌어집니다.
2. Conv 블록 3개면 28 → 14 → 7 → 3(MaxPool은 내림)이므로 `Linear(64 * 3 * 3, 10)`. 가짜 입력을 통과시켜 `[1, 576]`을 확인하세요.
3. `nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), ...` 순서(Conv → BN → ReLU). Dropout은 `nn.Flatten(), nn.Dropout(0.3), nn.Linear(...)`. 2 에폭에서는 정확도 차이가 작지만 BN 덕에 초반 loss가 더 빨리 내려갑니다.
4. `torch.save(mlp.state_dict(), "mlp_fashion.pt")` → 새 셀에서 `MLP()`를 다시 만들고 `load_state_dict(torch.load(...))` → `evaluate`. 불러오기 전(약 0.1)과 후(약 0.88)를 비교하세요. 클래스 정의 셀을 먼저 실행해야 합니다.

</details>
