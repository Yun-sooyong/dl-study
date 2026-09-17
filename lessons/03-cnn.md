# 3. 모델 구조 바꾸기 — CNN, 저장과 불러오기

**목표:** 같은 데이터·같은 학습 루프에서 **모델만 갈아끼워** 봅니다. "모델을 수정한다"는 감각을 익히는 레슨입니다.

## 준비 (레슨 2와 동일)

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

## 일부만 학습시키기 (freeze) — 파인튜닝의 기초

`requires_grad=False`로 두면 그 파라미터는 학습되지 않습니다. 레슨 5~7의 LoRA·VLM 학습이 전부 이 아이디어 위에 서 있습니다.

```python
for p in cnn2.features.parameters():
    p.requires_grad = False                      # 특징 추출부는 얼림
cnn2.classifier[1].reset_parameters()            # 분류기만 초기화해서 다시 학습
print("학습되는 파라미터:", sum(p.numel() for p in cnn2.parameters() if p.requires_grad))
train(cnn2, epochs=1)
```

## 직접 고쳐보기

1. 데이터를 `datasets.FashionMNIST`로 바꿔보세요 (이름만 바꾸면 됩니다). 옷 사진 분류는 숫자보다 어렵습니다. MLP와 CNN의 차이가 더 벌어지나요?
2. Conv 블록을 하나 더 추가하세요 (`Conv2d(32, 64, ...)`). `Linear`의 입력 크기는 얼마가 되어야 하나요? 위의 shape 추적 방법으로 구하세요.
3. `nn.Dropout(0.3)`을 classifier에, `nn.BatchNorm2d(16)`을 첫 Conv 뒤에 넣어보세요.
4. (도전) 레슨 2의 MLP로 FashionMNIST를 학습한 뒤 저장하고, 새 노트북 셀에서 불러와 평가해 보세요.
