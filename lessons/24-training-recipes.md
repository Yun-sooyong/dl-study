# 학습 잘 시키는 법 — loss 곡선, 정규화, 스케줄러, 디버깅

> ⏱ 70분 · GPU 권장 (CPU로도 5~10분이면 실행됨)

**목표:** 모델을 "돌리는 것"과 "잘 학습시키는 것"은 다릅니다. loss 곡선을 읽고, 과적합을 다루고, 학습이 안 될 때 원인을 찾는 **실전 기술**을 익힙니다. LLM 파인튜닝에서도 똑같이 쓰는 기술입니다.

## 준비: 일부러 데이터를 적게 쓴다

과적합을 눈으로 보기 위해 FashionMNIST(옷 사진 10종류) 중 **2000장만** 학습에 씁니다. 그리고 [과적합과 검증](#11-ml-overfitting) 레슨에서 배운 대로 검증(validation) 세트를 따로 둡니다.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tf = transforms.ToTensor()
full = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
train_ds, val_ds = Subset(full, range(2000)), Subset(full, range(50000, 55000))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=512)

def make_cnn(dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(128, 10),
    ).to(device)
```

**코드 읽기**

- `Subset(full, range(2000))` — 데이터셋의 일부만 쓰는 래퍼. 왜 2000장만 쓰나: 6만 장 전부 쓰면 이 작은 CNN은 과적합이 거의 안 일어나서 배울 게 없습니다. **일부러 데이터를 부족하게** 만들어 문제를 재현하는 것입니다. `range(50000, 55000)`처럼 학습 데이터와 겹치지 않는 구간에서 검증 세트를 떼어 냅니다.
- 검증(validation)과 테스트(test)를 구분 — 여기서는 `FashionMNIST(train=True)`의 뒷부분을 검증용으로 쓰고, `train=False`인 공식 테스트 세트는 아예 건드리지 않습니다. 실습에서 하이퍼파라미터를 고른 뒤 최종 점수를 재고 싶다면 그때 테스트 세트를 씁니다.
- `make_cnn(dropout=0.0)` — 모델을 **함수로** 만드는 이유: 실험마다 새 모델(새 난수 가중치)이 필요하고, dropout 비율만 다른 모델을 여러 개 찍어내야 하기 때문입니다. `dropout=0.0`이면 `nn.Dropout(0)`은 아무것도 하지 않으므로 baseline과 같습니다.
- `nn.Dropout(p)` — 학습 중 각 뉴런 출력을 확률 `p`로 0으로 만듭니다(`model.train()`일 때만). 평가 때는 꺼지고 대신 출력을 `1-p`배로 맞춰 줍니다. 여기서는 `Flatten` 뒤와 은닉층 뒤, 즉 파라미터가 가장 많은 `Linear` 층 앞에 넣었습니다. Conv 층에는 보통 dropout 대신 BatchNorm을 씁니다(아래 정규화 표 참고).
## 기록을 남기는 학습 루프

지금까지는 loss를 출력만 했습니다. 이제 에폭마다 **train과 validation의 loss·정확도를 모두 기록**합니다. 이 기록이 학습을 진단하는 청진기입니다.

```python
loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def run_eval(model, dl):
    model.eval()
    loss = correct = n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += loss_fn(out, y).item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return loss / n, correct / n

def fit(model, train_dl, epochs, optimizer, scheduler=None):
    hist = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        total = n = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if scheduler: scheduler.step()
            total += loss.item() * len(y); n += len(y)
        val_loss, val_acc = run_eval(model, val_dl)
        hist["train_loss"].append(total / n); hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc); hist["lr"].append(optimizer.param_groups[0]["lr"])
        if val_acc > best_acc:   # 검증 성능이 가장 좋았던 순간의 가중치를 보관 (best checkpoint)
            best_acc, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:2d}  train {total/n:.3f}  val {val_loss:.3f}  val_acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    return hist

def plot(hists):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hists.items():
        line, = axes[0].plot(h["train_loss"], "--"); axes[0].plot(h["val_loss"], color=line.get_color(), label=name)
        axes[1].plot(h["val_acc"], label=name)
    axes[0].set_title("loss (dashed: train, solid: val)"); axes[1].set_title("val accuracy")
    axes[0].set_xlabel("epoch"); axes[1].set_xlabel("epoch"); axes[0].legend(); axes[1].legend(); plt.show()
```

**코드 읽기**

- `@torch.no_grad()` — 함수 전체를 `with torch.no_grad():`로 감싸는 데코레이터. 평가 함수처럼 통째로 기울기가 필요 없을 때 이렇게 씁니다.
- `run_eval`이 loss까지 재는 이유 — 정확도는 "맞았나 틀렸나"만 보므로 둔합니다. 손실은 "얼마나 확신을 갖고 틀렸나"까지 담고 있어서, 정확도가 아직 오르고 있는데도 val loss가 먼저 오르기 시작하는 **과적합의 초기 신호**를 잡아냅니다. 그래서 진단 그래프에는 둘 다 필요합니다.
- `loss += loss_fn(out, y).item() * len(y)` — 배치 평균 손실에 배치 크기를 곱해 합을 모으고, 마지막에 전체 개수 `n`으로 나눕니다. 마지막 배치는 크기가 다를 수 있어서, 배치 평균들을 그냥 평균 내면 미세하게 틀립니다.
- `fit(model, train_dl, epochs, optimizer, scheduler=None)` — [첫 신경망](#21-mlp-mnist)의 `train`에 세 가지가 추가되었습니다. ① 에폭마다 4가지 값을 `hist`에 기록, ② `scheduler`가 있으면 매 스텝 `scheduler.step()` 호출, ③ 검증 정확도가 최고일 때의 가중치를 복사해 두었다가 끝나면 되돌리기.
- `{k: v.clone() for k, v in model.state_dict().items()}` — `state_dict()`는 텐서를 복사하지 않고 **참조**를 돌려주므로, 그냥 저장하면 이후 학습에 따라 값이 같이 바뀝니다. `clone()`으로 진짜 사본을 떠야 합니다. 이런 미묘한 버그가 실제로 매우 흔합니다.
- `optimizer.param_groups[0]["lr"]` — 현재 학습률을 읽는 관용구. 스케줄러가 학습률을 바꿨는지 그래프로 확인하려고 기록합니다.
- `plot(hists)` — 여러 실험의 곡선을 **한 그래프에 겹쳐** 그립니다. train은 점선, val은 실선, 실험마다 같은 색. 실험 결과를 따로따로 보면 비교가 안 되므로 처음부터 겹쳐 그리는 함수를 만들어 두는 것입니다.
## 실험 1: 아무 장치 없이 학습 → 과적합 관찰

```python
hists = {}
model = make_cnn()
hists["baseline"] = fit(model, train_dl, 30, torch.optim.Adam(model.parameters(), lr=1e-3))
plot(hists)
```

**loss 곡선 읽는 법**

| 모양 | 진단 | 처방 |
|---|---|---|
| train↓ val↓ 함께 내려감 | 정상. 아직 더 학습할 여지 있음 | 계속 학습 |
| train↓ 인데 val은 **다시 올라감** | 과적합 | 정규화, 증강, 데이터 추가, 일찍 멈추기 |
| 둘 다 높은 곳에서 정체 | 과소적합 또는 학습률 문제 | 모델 키우기, 학습률 조정, 더 오래 학습 |
| loss가 튀거나 NaN | 발산 | 학습률 낮추기 |

위 그래프에서 val loss가 몇 에폭째부터 다시 올라가는지 확인하세요. 그 뒤의 학습은 "외우기"입니다. `fit`이 검증 성능이 가장 좋았던 순간의 가중치를 되돌려 주는 것(**early stopping**의 간단한 형태)도 그 때문입니다.

## 실험 2: 과적합 줄이기 — dropout, weight decay, 데이터 증강

- **Dropout:** 학습 중 뉴런을 무작위로 꺼서 특정 뉴런에 의존하지 못하게 합니다.
- **Weight decay:** 가중치가 커지는 데 벌점 ([과적합과 검증](#11-ml-overfitting)의 Ridge와 같은 원리). `AdamW`의 `weight_decay`.
- **데이터 증강(augmentation):** 이미지를 살짝 이동·뒤집어 "새 데이터"를 만들어 냅니다. 데이터를 늘리는 가장 값싼 방법입니다.

```python
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),          # 좌우 반전 (옷은 뒤집어도 같은 옷)
    transforms.ToTensor(),
])
aug_full = datasets.FashionMNIST("data", train=True, transform=aug)     # 같은 데이터, 매번 다르게 변형되어 나옴
aug_dl = DataLoader(Subset(aug_full, range(2000)), batch_size=64, shuffle=True)

model = make_cnn(dropout=0.5)
hists["dropout+wd+aug"] = fit(model, aug_dl, 30, torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05))
plot(hists)
```

두 곡선을 비교해 보세요.

- baseline의 val loss는 중간부터 **다시 올라가지만**(마지막에 약 0.55), 정규화한 쪽은 낮은 곳(약 0.45)에 머뭅니다.
- train loss는 정규화한 쪽이 훨씬 **높습니다**. 외우기가 어려워졌기 때문이며 정상입니다. 목표는 train loss를 낮추는 것이 아니라 **val 성능**입니다.
- 그런데 val **정확도**는 둘 다 85% 근처로 큰 차이가 없습니다. 정규화는 과적합을 늦추고 학습을 안정시키는 장치이지, **부족한 데이터를 대신해 주지는 못합니다.** 성능을 크게 올리는 것은 더 많은 데이터와 더 좋은 사전학습 모델입니다(아래 실습 2번, 다음 레슨).

> 증강은 과제에 맞아야 합니다. 이 데이터에서는 이미지를 상하좌우로 미는 `RandomCrop`을 추가하면 30 에폭 안에서는 오히려 성능이 떨어집니다. 증강이 강할수록 더 오래 학습해야 하기 때문입니다. "좋다고 알려진 기법"도 내 데이터에서 검증해야 합니다.

**코드 읽기**

- `transforms.Compose([...])` — 변환 여러 개를 순서대로 적용하는 묶음. 증강(무작위 변형)은 `ToTensor()` **앞에**, 즉 이미지 상태에서 합니다.
- `transforms.RandomHorizontalFlip()` — 50% 확률로 좌우 반전. 왜 이것만 남겼나: 옷은 뒤집어도 같은 옷이므로 라벨이 안 바뀝니다(반면 숫자 6/9나 글자에는 쓰면 안 됩니다). 증강을 고를 때의 기준은 "**이 변형을 해도 정답이 그대로인가**"입니다.
- 왜 `datasets.FashionMNIST(..., transform=aug)`를 **새로** 만드나: `transform`은 데이터셋 객체에 붙어 있습니다. 같은 파일을 다른 변환으로 읽는 두 번째 데이터셋 객체를 만든 것이고, 파일은 이미 내려받았으므로 `download=True`가 필요 없습니다. 검증 세트는 증강 없이 그대로 둡니다. 평가 데이터를 변형하면 안 됩니다.
- `torch.optim.AdamW(..., weight_decay=0.05)` — Adam에 weight decay(L2 벌점)를 올바르게 붙인 버전. 그냥 `Adam`에도 `weight_decay` 인자가 있지만 Adam의 적응형 학습률과 섞여 효과가 왜곡됩니다. AdamW는 벌점을 기울기와 분리해서 적용하도록 고친 것이고, 지금은 트랜스포머·LLM 학습의 사실상 표준입니다. 0.05는 흔히 쓰는 범위(0.01~0.1)의 값입니다.
- `make_cnn(dropout=0.5)` — 이번 실험에서는 절반을 끕니다. 0.2~0.5가 보통이고, 데이터가 적을수록 크게 씁니다.

### 정규화 기법 총정리: 무엇을, 왜, 언제

[과적합과 검증](#11-ml-overfitting)에서 고전 머신러닝의 정규화(L1, L2, 모델 크기 제한)를 봤습니다. 딥러닝에는 도구가 더 많습니다. 공통 원리는 하나입니다. **모델이 학습 데이터의 우연한 세부 사항까지 외우기 어렵게 만든다.** 다만 "어디를 어렵게 만드느냐"가 다르고, 그에 따라 쓰는 상황이 갈립니다.

| 기법 | 어떻게 방해하나 | 언제 쓰나 | 코드 |
|---|---|---|---|
| **Weight decay (L2)** | 가중치가 커지면 벌점 → 몇몇 특징에 극단적으로 의존하지 못함 | 거의 항상 기본으로. 트랜스포머·LLM은 0.01~0.1 | `AdamW(weight_decay=0.05)` |
| **Dropout** | 학습 중 뉴런을 무작위로 꺼서 특정 뉴런 조합에 의존하지 못하게 함 | `Linear` 층이 크고 데이터가 적을 때. 트랜스포머의 어텐션·MLP 뒤(0.1). 최신 대형 LLM은 데이터가 충분해 거의 안 씀 | `nn.Dropout(0.5)` |
| **데이터 증강** | 매 에폭 조금씩 다른 입력을 보여 줘 "똑같은 샘플"을 못 외우게 함 | 이미지·음성에서 가장 효과적. 라벨이 안 바뀌는 변형만 | `RandomHorizontalFlip`, `RandomCrop`, 색 변형 |
| **Early stopping** | val 성능이 나빠지기 전에 멈춤. 외울 시간을 안 줌 | 반복 학습하는 모든 모델. 비용이 0이라 항상 | `fit`의 best checkpoint |
| **BatchNorm / LayerNorm** | 층 출력의 분포를 고르게 맞춤. 주목적은 학습 안정화이지만 약한 정규화 효과도 있음 | CNN에는 BatchNorm(Conv 뒤), 트랜스포머에는 LayerNorm. 거의 모든 현대 모델에 기본 포함 | `nn.BatchNorm2d(16)`, `nn.LayerNorm(d)` |
| **Label smoothing** | 정답 확률 1.0 대신 0.9를 목표로 → 지나친 확신을 벌함 | 분류에서 모델이 과하게 확신할 때. 대형 이미지 분류의 표준 | `CrossEntropyLoss(label_smoothing=0.1)` |
| **모델 축소** | 표현력 자체를 줄임 | 데이터가 매우 적고 위 방법으로도 안 될 때 | `hidden=32`, 층 수 줄이기 |
| **더 많은 데이터** | 외울 수 있는 "틈"을 없앰 | 언제나 최선. 가능한지부터 먼저 확인 | — |
| **사전학습 모델 사용** | 이미 좋은 위치에서 출발해 조금만 움직임 | 데이터가 적은 거의 모든 실전 문제 | [전이학습](#25-transfer-learning), LoRA |

고르는 순서의 기준:

1. **데이터를 더 구할 수 있는가?** 있다면 그게 정답입니다. 위 실험에서 본 대로, 정규화는 부족한 데이터를 대신하지 못합니다.
2. **사전학습 모델이 있는가?** 있다면 처음부터 학습하지 않습니다.
3. 그다음 weight decay + early stopping을 기본으로 깔고, 이미지라면 증강, 큰 `Linear` 층이 있다면 dropout을 추가합니다.
4. 정규화는 **train 성능을 일부러 낮추는** 장치이므로, 과소적합(train도 못 맞힘) 상태에서는 오히려 해롭습니다. 곡선을 보고 과적합이 확인된 뒤에 씁니다.
5. 한 번에 하나씩 넣고 val 곡선으로 확인합니다. 두 개를 동시에 넣으면 어느 것이 효과인지 알 수 없습니다.

BatchNorm은 왜 정규화 표에 있으면서 "주목적은 학습 안정화"인가: 배치 단위로 출력의 평균·분산을 맞추면 깊은 층까지 값이 폭발하거나 사라지지 않아 큰 학습률로 빠르게 학습할 수 있습니다. 그 과정에서 배치마다 조금씩 다른 통계가 노이즈로 작용해 정규화 효과가 덤으로 생깁니다. 트랜스포머가 BatchNorm 대신 LayerNorm을 쓰는 이유는, 문장 길이가 제각각이고 배치 크기가 작아서 배치 통계가 불안정하기 때문입니다. LayerNorm은 샘플 하나 안에서만 통계를 내므로 배치와 무관합니다.

## 실험 3: 학습률 스케줄러

처음에는 작게 시작해 빠르게 올리고(warmup), 끝으로 갈수록 줄이는 것이 표준입니다. 큰 학습률로 넓게 탐색한 뒤 작은 학습률로 세밀하게 마무리하는 효과가 있습니다. LLM 학습은 거의 예외 없이 **warmup + cosine decay**를 씁니다.

```python
epochs = 30
model = make_cnn(dropout=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=epochs * len(aug_dl))   # warmup 후 cosine으로 감소
hists["+scheduler"] = fit(model, aug_dl, epochs, optimizer, scheduler)
plot(hists)
plt.plot(hists["+scheduler"]["lr"]); plt.title("learning rate per epoch"); plt.show()
```

**코드 읽기**

- `torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=...)` — 학습률을 작은 값에서 시작해 `max_lr`까지 올렸다가(전체의 약 30%) 코사인 곡선을 따라 거의 0까지 내리는 스케줄. 왜 이것을 골랐나: 설정이 `max_lr` 하나뿐이고, 전체 스텝 수만 알려주면 warmup과 감소를 알아서 배치해 주는 가장 손이 덜 가는 스케줄러입니다. `total_steps`는 "에폭 수 × 에폭당 배치 수"이며 이 값이 틀리면 에러가 납니다.
- 왜 `max_lr=5e-3`으로 기본 학습률(1e-3)보다 크게 잡나: 초반에 작게 시작(warmup)하기 때문에 최고점은 평소보다 높게 가도 안전합니다. 큰 학습률 구간이 넓게 탐색하고, 마지막의 작은 학습률이 정밀하게 마무리합니다.
- `scheduler.step()`을 **매 스텝**(`fit` 안의 배치 루프) 호출 — `OneCycleLR`은 스텝 단위 스케줄러입니다. 에폭 단위 스케줄러(`StepLR` 등)는 에폭 루프 끝에서 부릅니다. 이 차이를 헷갈리면 학습률이 엉뚱하게 움직이므로, 학습률을 기록해서 **그래프로 확인**하는 습관이 중요합니다.
- 다른 선택지: `CosineAnnealingLR`(warmup 없이 코사인 감소만), `LinearLR`+`CosineAnnealingLR`을 `SequentialLR`로 이어 붙인 "warmup + cosine"(LLM 학습의 표준), `ReduceLROnPlateau`(val 손실이 정체되면 학습률을 낮춤). 어느 것이든 "처음엔 크게, 끝엔 작게"가 공통입니다.
## 학습이 안 될 때: 디버깅 체크리스트

신경망은 버그가 있어도 에러 없이 "그냥 성능이 나쁜" 상태로 돌아갑니다. 그래서 의심스러우면 아래 순서로 확인합니다.

**① 초기 loss가 예상값인가?** 10개 클래스를 찍는 모델의 cross entropy는 `ln(10) ≈ 2.30`이어야 합니다.

```python
import math
model = make_cnn()
x, y = next(iter(train_dl))
print("초기 loss:", loss_fn(model(x.to(device)), y.to(device)).item(), " 예상:", math.log(10))
```

**② 배치 하나를 완벽하게 외울 수 있는가?** 못 외운다면 데이터·모델·루프 어딘가에 버그가 있습니다. 가장 강력한 점검법입니다.

```python
x, y = x[:32].to(device), y[:32].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(200):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
print("같은 배치 200번 학습 후 loss:", loss.item(), " 정확도:", (model(x).argmax(1) == y).float().mean().item())   # 0에 가깝고, 1.0이어야 정상
```

**코드 읽기**

- `math.log(10)` — 10개 클래스에 균등하게 확률을 주면 정답 확률이 0.1이고, cross entropy는 `−ln 0.1 = ln 10 ≈ 2.30`. 이 값보다 초기 loss가 훨씬 크면 마지막 층의 출력이 너무 큰 것(초기화 문제), 훨씬 작으면 뭔가 정답을 미리 보고 있다는 뜻(데이터 누출)입니다. 클래스 수가 다르면 `ln(클래스 수)`로 바꿔 계산하세요. [미니 GPT](#31-mini-gpt)에서는 `ln(어휘 크기)`가 됩니다.
- 배치 하나로 200번 학습 — 같은 32장을 반복해서 보여주면 정상적인 모델은 반드시 외웁니다(loss → 0, 정확도 1.0). 못 외운다면 **학습 자체가 안 되는 것**이므로 데이터·모델·손실·옵티마이저 중 어딘가가 잘못된 것이고, 원인을 찾는 범위가 크게 좁혀집니다. 새 모델·새 데이터를 다룰 때 가장 먼저 하는 점검입니다.
- `.float().mean()` — True/False 텐서를 1.0/0.0으로 바꿔 평균 = 정확도. `sum() / len()`과 같지만 짧습니다.
**③ 그래도 안 되면**

- 학습률을 10배씩 바꿔 본다 (`1e-2, 1e-3, 1e-4`). 가장 흔한 원인입니다.
- 입력과 라벨을 직접 출력/시각화해 짝이 맞는지 본다.
- `model.train()` / `model.eval()` 전환, `zero_grad()` 누락을 확인한다.
- 모든 텐서의 shape과 device를 찍어 본다.

## 핵심 정리

- 항상 **train과 validation 곡선을 함께** 기록하고 봅니다. 둘의 격차가 과적합의 크기입니다.
- 과적합 대응: dropout, weight decay, 데이터 증강, early stopping(최고 시점의 가중치 보관).
- 학습률은 고정값보다 **warmup → 감소** 스케줄이 대체로 낫습니다.
- 버그 점검은 "초기 loss 확인 → 배치 하나 외우기"부터.
- 실험은 **한 번에 하나만** 바꾸고, 결과를 같은 그래프에 겹쳐 비교합니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 정규화를 넣었더니 train loss가 오히려 높아졌습니다. 실패한 걸까요?</summary>

아닙니다. 정규화는 학습 데이터를 외우기 어렵게 만드는 장치이므로 train loss가 오르는 것이 정상입니다. 판단 기준은 val loss/정확도입니다.

</details>

<details><summary>Q2. 최종 모델로 "마지막 에폭의 가중치"가 아니라 "val 성능이 가장 좋았던 에폭의 가중치"를 쓰는 이유는?</summary>

과적합이 시작된 뒤의 가중치는 검증 성능이 더 나쁩니다. 다만 이렇게 고른 val 점수는 약간 낙관적이므로, 최종 보고용 성능은 따로 떼어 둔 test 세트로 잽니다.

</details>

<details><summary>Q3. 새로 짠 학습 코드의 정확도가 10%에서 움직이지 않습니다. 가장 먼저 무엇을 해 보겠습니까?</summary>

배치 하나를 외울 수 있는지 확인합니다. 그것도 안 되면 학습률, 라벨과 입력의 짝, `zero_grad`/`step` 호출, 모델 출력 shape을 점검합니다. 외워지는데 전체 학습만 안 된다면 학습률이나 데이터 로딩(셔플, 정규화)을 의심합니다.

</details>

## 직접 고쳐보기

1. baseline에 dropout **만**, weight decay **만**, 증강 **만** 각각 추가해 세 개의 곡선을 비교하세요. 어느 것의 효과가 가장 큰가요?
2. 학습 데이터를 2000장에서 20000장(`range(20000)`)으로 늘리고 baseline을 다시 돌려 보세요(GPU 권장). 어떤 정규화보다 효과가 큽니다. 과적합은 어떻게 달라지나요?
3. `lr=1e-1`과 `lr=1e-5`로 baseline을 돌려 위 진단표의 어느 모양이 나오는지 확인하세요.
4. `OneCycleLR` 대신 `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(aug_dl))`을 써 보세요.
5. (도전) `make_cnn`에 `nn.BatchNorm2d`를 각 Conv 뒤에 넣어 보세요. 수렴 속도가 어떻게 달라지나요?

<details><summary>힌트와 예상 결과 — 먼저 스스로 해 본 뒤 펼치세요</summary>

1. 2000장·30에폭 기준으로 val loss 마지막 값: baseline 약 0.57, dropout만 약 0.45, wd만 약 0.5, flip만 약 0.5. dropout 단독 효과가 가장 큽니다(큰 Linear 층이 과적합의 주범이라서). 정확도는 셋 다 0.85 근처.
2. 20000장이면 baseline의 val loss가 0.3 아래로, 정확도 0.9 이상. 정규화 세 개를 합친 것보다 데이터 10배가 훨씬 큽니다.
3. `1e-1`: loss가 2.3 근처에서 정체하거나 nan(발산, 진단표의 마지막 줄). `1e-5`: 30 에폭 뒤에도 loss 1.0 이상(과소적합처럼 보이는 "학습률 너무 작음").
4. 코사인만 쓰면 warmup 없이 처음부터 큰 학습률로 시작합니다. 이 작은 모델에서는 차이가 작지만, 최고 정확도 도달 에폭이 조금 뒤로 밀리는 것을 볼 수 있습니다.
5. `nn.Conv2d(...), nn.BatchNorm2d(32), nn.ReLU()`. 1~3 에폭에서 val 정확도가 baseline보다 빠르게 오릅니다(예: 1 에폭 0.75 → 0.8). 최종 정확도도 1~2%p 높습니다.

</details>
