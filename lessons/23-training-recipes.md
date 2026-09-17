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
