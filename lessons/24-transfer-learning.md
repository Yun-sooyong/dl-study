# 전이학습 — 남이 학습시킨 모델 위에서 시작하기

> ⏱ 60분 · T4 GPU 권장 (GPU 약 3분, CPU는 20분 이상)

**목표:** 사전학습된 모델을 가져와 **내 데이터에 맞게 고쳐서** 학습시킵니다. 데이터가 2000장뿐일 때 "처음부터 학습"과 "전이학습"의 차이를 직접 비교합니다. 뒤에 나올 LLM·VLM 파인튜닝은 전부 이 아이디어의 확장입니다.

## 왜 전이학습인가

ImageNet(120만 장)으로 학습된 모델의 앞쪽 층들은 이미 **모서리, 질감, 눈, 바퀴** 같은 범용 시각 특징을 뽑을 줄 압니다. 내 과제가 달라도 이 "눈"은 그대로 쓸 수 있습니다. 새로 배워야 하는 것은 마지막의 "이 특징들이 모이면 내 클래스 중 무엇인가"뿐입니다.

현실의 딥러닝 프로젝트 대부분은 처음부터 학습하지 않습니다. **좋은 사전학습 모델을 골라 파인튜닝**합니다.

## 데이터: CIFAR-10에서 2000장만

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# 사전학습 모델이 학습될 때 쓰인 것과 같은 크기·정규화를 맞춰 줘야 합니다
tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),   # ImageNet 평균/표준편차
])
train_full = datasets.CIFAR10("data", train=True, download=True, transform=tf)
test_full = datasets.CIFAR10("data", train=False, download=True, transform=tf)
train_dl = DataLoader(Subset(train_full, range(2000)), batch_size=32, shuffle=True)
test_dl = DataLoader(Subset(test_full, range(1000)), batch_size=100)
print(train_full.classes)
```

**코드 읽기**

- `datasets.CIFAR10` — 32×32 컬러 사진 10종(비행기, 자동차, 새, 고양이…) 6만 장. MNIST보다 어렵고, 컬러(3채널)라 사전학습 모델과 형식이 맞습니다.
- `transforms.Resize(224)` — 32×32를 224×224로 키웁니다. 왜: ResNet은 224 크기 이미지로 사전학습되었고, 필터들이 그 크기의 패턴에 맞춰져 있습니다. 32 그대로 넣으면 층을 지날수록 크기가 1×1까지 줄어 동작은 해도 성능이 형편없습니다. **사전학습 모델을 쓸 때는 입력 형식을 그 모델의 학습 조건에 맞추는 것이 첫 번째 규칙**입니다.
- `transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` — 채널(R, G, B)별로 `(픽셀 − 평균) / 표준편차`. 이 여섯 숫자는 ImageNet 전체의 채널별 평균과 표준편차로, 사전학습 때 정확히 이 값으로 정규화했습니다. 다른 값을 쓰면 모델이 보는 색 분포가 학습 때와 달라져 성능이 떨어집니다. 모델 카드에 항상 적혀 있는 값입니다.
- `Subset(train_full, range(2000))` — 일부러 2000장만. 전이학습의 이득은 데이터가 적을수록 극적이므로, 그것을 보이려는 설정입니다.
- `batch_size=32` — 224×224 이미지는 28×28보다 64배 크므로 배치를 줄여 메모리를 아낍니다. 배치 크기는 성능 설정이기 이전에 **메모리 설정**입니다.
## 사전학습 모델 뜯어보기

```python
resnet = models.resnet18(weights="IMAGENET1K_V1")
print([name for name, _ in resnet.named_children()])   # conv1, bn1, ..., layer1~4, avgpool, fc
print(resnet.fc)                                        # Linear(512 → 1000): ImageNet의 1000개 클래스용 머리
```

몸통(`conv1` ~ `layer4`)은 이미지를 512차원 특징 벡터로 바꾸고, 머리(`fc`)가 그것을 1000개 클래스로 분류합니다. 우리 클래스는 10개이므로 **머리를 새것으로 갈아 끼웁니다.** [CNN 레슨](#22-cnn)에서 해 본 "모델 수정"입니다.

```python
def make_model(pretrained, freeze_body):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    if freeze_body:
        for p in model.parameters():
            p.requires_grad = False                 # 몸통을 얼림
    model.fc = nn.Linear(model.fc.in_features, 10)  # 새 머리 (새로 만든 층은 requires_grad=True)
    return model.to(device)
```

**코드 읽기**

- `models.resnet18(weights="IMAGENET1K_V1")` — torchvision이 구조와 함께 ImageNet으로 학습된 **가중치를 내려받아** 끼워 줍니다(약 45MB). `weights=None`이면 구조만 만들고 가중치는 난수입니다. 왜 ResNet-18인가: 잔차 연결(residual)로 유명한 ResNet 계열 중 가장 작아서 빠르고, 전이학습의 기준 모델로 가장 널리 쓰입니다. 18은 층 수입니다.
- `model.named_children()` — 모델의 최상위 부품 이름을 나열합니다. 처음 보는 모델은 이렇게 **구조를 먼저 훑어** 어디가 몸통이고 어디가 머리인지 파악합니다. `print(model)`은 전체를 다 찍어 길지만 더 자세합니다.
- `model.fc` — ResNet의 마지막 `Linear(512, 1000)`. 1000은 ImageNet 클래스 수. 우리 문제는 10개이므로 이 층은 어차피 못 쓰고, **같은 이름의 속성에 새 층을 대입**하면 교체가 끝납니다. 모델마다 이 속성 이름이 다릅니다(`fc`, `classifier`, `head`). `model.fc.in_features`(512)를 읽어 쓰면 숫자를 외울 필요가 없습니다.
- 순서가 중요합니다: **먼저 전부 얼리고, 그다음 새 머리를 만듭니다.** 새로 만든 `nn.Linear`는 기본값이 `requires_grad=True`이므로 머리만 학습 대상으로 남습니다. 순서를 바꾸면 머리까지 얼어서 아무것도 학습되지 않습니다.
- `make_model(pretrained, freeze_body)` 두 개의 스위치 — 세 가지 실험(처음부터 / 얼리기 / 전체 파인튜닝)을 같은 함수로 만들기 위한 설계입니다. 비교 실험에서는 **차이가 나는 부분만 인자로** 빼는 것이 실수를 줄입니다.
## 학습·평가 함수

```python
@torch.no_grad()
def accuracy(model):
    model.eval()
    correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum().item() for x, y in test_dl)
    return correct / len(test_dl.dataset)

def train(model, epochs, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습되는 파라미터: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            loss = loss_fn(model(x.to(device)), y.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"  epoch {epoch+1}  loss {loss.item():.3f}  test acc {accuracy(model):.3f}")
    return accuracy(model)
```

- `params = [p for p in model.parameters() if p.requires_grad]` — 옵티마이저에는 **학습할 파라미터만** 넘깁니다. 얼린 파라미터를 넘겨도 `.grad`가 없어 갱신은 안 되지만, AdamW가 그들을 위한 상태 메모리를 쓸데없이 잡습니다. 이 필터는 LoRA 학습에서도 똑같이 나옵니다.
- `sum(p.numel() for p in params)` 출력 — 얼렸을 때 5,130개(= 512×10 + 10), 아닐 때 1,100만 개. 실험 전에 이 숫자를 찍어 "내가 의도한 부분만 학습되는지" 확인하는 것이 습관이 되어야 합니다.
- `train(model, epochs, lr)`이 `lr`을 인자로 받는 이유 — 아래에서 파인튜닝만 학습률을 다르게 주기 위해서입니다.
## 세 가지 방법 비교

```python
epochs = 3
results = {}

print("① 처음부터 학습 (랜덤 초기화)")
results["from scratch"] = train(make_model(pretrained=False, freeze_body=False), epochs, lr=1e-3)

print("② 특징 추출: 사전학습 몸통은 얼리고 머리만 학습")
results["frozen body"] = train(make_model(pretrained=True, freeze_body=True), epochs, lr=1e-3)

print("③ 파인튜닝: 사전학습 가중치에서 출발해 전체를 작은 학습률로")
results["fine-tune all"] = train(make_model(pretrained=True, freeze_body=False), epochs, lr=1e-4)

for k, v in results.items():
    print(f"{k:15s} {v:.3f}")
```

| 방법 | 학습 대상 | 언제 쓰나 |
|---|---|---|
| 처음부터 학습 | 전부 (랜덤에서 시작) | 데이터가 아주 많고, 맞는 사전학습 모델이 없을 때 |
| 특징 추출 (몸통 얼림) | 머리만 | 데이터가 매우 적을 때, 빠르게 기준선을 세울 때 |
| 파인튜닝 | 전부 (사전학습에서 시작) | 가장 일반적. 보통 가장 성능이 높음 |

같은 구조, 같은 데이터, 같은 에폭인데 **출발점만 다릅니다.** 2000장으로는 "눈"을 처음부터 만들기에 턱없이 부족하지만, 이미 만들어진 눈을 조정하기에는 충분합니다.

## 파인튜닝에서 학습률을 작게 하는 이유

사전학습 가중치는 이미 좋은 위치에 있습니다. 큰 학습률로 크게 움직이면 애써 배운 특징이 망가집니다(**catastrophic forgetting**). 그래서 파인튜닝은 보통 처음부터 학습할 때보다 10배쯤 작은 학습률을 씁니다. 새로 만든 머리에는 큰 학습률을, 몸통에는 작은 학습률을 주기도 합니다.

```python
model = make_model(pretrained=True, freeze_body=False)
body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
optimizer = torch.optim.AdamW([
    {"params": body, "lr": 1e-5},                     # 몸통: 아주 조금씩
    {"params": model.fc.parameters(), "lr": 1e-3},    # 새 머리: 빠르게
])
print([g["lr"] for g in optimizer.param_groups])
```

**코드 읽기**

- `model.named_parameters()` — `(이름, 파라미터)` 쌍을 돌려줍니다. 이름이 `fc.`로 시작하는지로 머리와 몸통을 가릅니다. 이름 기반으로 고르는 이 방식은 [VLM 파인튜닝](#42-vlm-finetune)에서 LoRA를 붙일 층을 정규식으로 고르는 것과 같은 발상입니다.
- `torch.optim.AdamW([{"params": ..., "lr": ...}, {...}])` — 파라미터 묶음(param group)마다 다른 학습률을 주는 문법. 옵티마이저는 그룹별로 설정을 따로 적용합니다. 몸통에는 `1e-5`(거의 안 움직임), 머리에는 `1e-3`(빠르게 학습). 이 방법을 "차등 학습률(discriminative learning rate)"이라고 합니다.
- 왜 이렇게까지 하나: 머리는 난수에서 출발하므로 크게 움직여야 하고, 몸통은 이미 좋은 위치라 조금만 움직여야 합니다. 하나의 학습률로 둘을 동시에 만족시키기 어렵습니다. 데이터가 아주 적을 때 특히 효과적입니다.
## 핵심 정리

- 전이학습 = 사전학습 모델의 가중치에서 출발해 내 과제에 맞게 조정하는 것.
- 절차: 모델 불러오기 → **머리 교체** → (선택) 몸통 얼리기 → 작은 학습률로 학습.
- 입력 전처리(크기, 정규화)는 사전학습 때와 같게 맞춥니다.
- 데이터가 적을수록 전이학습의 이득이 큽니다.
- LLM 파인튜닝도 같은 구조입니다: 사전학습된 몸통, 작은 학습률, 일부만 학습(LoRA).

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. CIFAR-10에는 ImageNet에 없는 클래스도 있는데 왜 전이학습이 통하나요?</summary>

앞쪽 층이 배운 것은 특정 클래스가 아니라 모서리·질감·형태 같은 범용 특징이기 때문입니다. 클래스에 특화된 부분은 머리이고, 그것은 새로 학습합니다.

</details>

<details><summary>Q2. 몸통을 얼리면 학습이 빠른 이유 두 가지는?</summary>

① 학습할 파라미터가 머리(5천여 개)뿐이라 옵티마이저 계산이 거의 없고, ② 몸통에 대해서는 기울기를 계산·저장할 필요가 없어 backward가 가볍습니다. 몸통 출력을 미리 계산해 저장해 두면 더 빨라집니다.

</details>

<details><summary>Q3. 파인튜닝에서 학습률을 크게(1e-2) 주면 어떤 일이 생길까요?</summary>

사전학습으로 얻은 좋은 가중치가 첫 몇 스텝 만에 크게 흐트러져, 사실상 나쁜 초기값에서 처음부터 학습하는 것과 비슷해집니다.

</details>

## 직접 고쳐보기

1. 학습 데이터를 2000장에서 200장, 20000장으로 바꿔 세 방법을 다시 비교하세요. 데이터가 늘수록 "처음부터 학습"과의 격차가 어떻게 변하나요?
2. ③의 학습률을 `1e-2`로 올려 보세요. Q3의 예상이 맞나요?
3. `layer4`만 풀고 나머지는 얼려 보세요: 얼린 뒤 `for p in model.layer4.parameters(): p.requires_grad = True`.
4. `models.resnet50(weights="IMAGENET1K_V2")`나 `models.efficientnet_b0(weights="IMAGENET1K_V1")`로 바꿔 보세요. 머리의 이름이 모델마다 다릅니다(`fc`, `classifier`). `print(model)`로 확인하세요.
5. (도전) [학습 잘 시키는 법](#23-training-recipes)의 데이터 증강과 스케줄러를 ③에 적용해 정확도를 더 올려 보세요.
