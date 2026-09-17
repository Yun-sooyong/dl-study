# 텐서와 자동미분 — 학습이란 무엇인가

> ⏱ 40분 · CPU로 충분 · 선수 지식: 파이썬 기초

**목표:** "모델을 학습시킨다"가 실제로 어떤 계산인지, 가장 작은 예제로 손에 익힙니다.

## 텐서: 숫자가 담긴 다차원 배열

딥러닝의 모든 데이터(이미지, 글, 모델의 가중치)는 텐서입니다.

```python
import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(x.shape)        # torch.Size([2, 3]) — 2행 3열
print(x * 2)          # 원소별 연산
print(x @ x.T)        # 행렬곱: (2,3) @ (3,2) = (2,2)
print(x.mean(dim=0))  # 0번 축(행)을 따라 평균 → 길이 3
```

## 자동미분: PyTorch의 핵심

`requires_grad=True`인 텐서로 계산하면, PyTorch가 계산 과정을 기록해 두었다가 `backward()` 한 번에 미분값을 구해줍니다.

```python
w = torch.tensor(3.0, requires_grad=True)
y = w ** 2 + 2 * w    # y = w² + 2w
y.backward()          # dy/dw 계산
print(w.grad)         # 2w + 2 = 8
```

기울기(gradient)는 "w를 조금 키우면 y가 얼마나 변하나"입니다. 손실의 기울기를 알면, **반대 방향으로 조금 움직여서 손실을 줄일 수 있습니다.** 이것이 학습의 전부입니다.

## 직접 학습시켜 보기: 직선 맞추기

정답이 `y = 2x + 1`인 데이터를 만들고, 모델이 `w=2, b=1`을 스스로 찾아내게 합니다.

```python
torch.manual_seed(0)
X = torch.rand(100, 1) * 10
Y = 2 * X + 1 + torch.randn(100, 1) * 0.5   # 약간의 노이즈

w = torch.zeros(1, requires_grad=True)   # 학습할 파라미터. 0에서 시작
b = torch.zeros(1, requires_grad=True)
lr = 0.01                                # 학습률: 한 번에 얼마나 움직일지

for step in range(1000):
    pred = X * w + b                     # 1) 예측
    loss = ((pred - Y) ** 2).mean()      # 2) 손실 (평균제곱오차)
    loss.backward()                      # 3) 기울기 계산
    with torch.no_grad():                # 4) 파라미터 갱신 (이 계산은 기록하지 않음)
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()                   # 기울기는 누적되므로 매번 0으로 초기화
        b.grad.zero_()
    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

print(f"결과: w={w.item():.3f} (정답 2), b={b.item():.3f} (정답 1)")
```

데이터에 노이즈를 섞었기 때문에 정확히 2와 1이 나오지는 않습니다. 모델은 "주어진 데이터에 가장 잘 맞는" 값을 찾을 뿐입니다.

## 같은 일을 PyTorch 도구로

위의 수작업을 `nn.Linear`(파라미터 묶음), `optim.SGD`(갱신 규칙)가 대신해 줍니다. **구조는 완전히 같습니다.** 앞으로 모든 레슨이 이 모양입니다.

```python
from torch import nn

model = nn.Linear(1, 1)                                  # w와 b를 가진 층
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for step in range(1000):
    loss = loss_fn(model(X), Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model.weight.item(), model.bias.item())
```

## 핵심 정리

- 학습 = **손실을 줄이는 방향으로 파라미터를 조금씩 옮기는 것**. 방향은 기울기가 알려줍니다.
- `loss.backward()`가 모든 파라미터의 기울기를 계산하고, `optimizer.step()`이 파라미터를 갱신합니다.
- 기울기는 누적되므로 매 스텝 `zero_grad()`가 필요합니다.
- 학습률이 너무 크면 발산하고, 너무 작으면 느립니다.
- `예측 → 손실 → backward → step` 네 줄은 이 코스 끝까지 그대로 나옵니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. <code>requires_grad=True</code>는 무엇을 하라는 표시인가요?</summary>

이 텐서가 관여한 계산을 기록해 두었다가, `backward()` 때 이 텐서에 대한 기울기를 구해 `.grad`에 넣으라는 표시입니다. 학습할 파라미터에 붙입니다.

</details>

<details><summary>Q2. 파라미터를 갱신할 때 왜 기울기를 <b>빼나요</b>?</summary>

기울기는 손실이 **커지는** 방향입니다. 손실을 줄이려면 반대 방향으로 가야 하므로 `w -= lr * w.grad`입니다.

</details>

<details><summary>Q3. 직선 모델로 <code>y = 3x² + 1</code> 데이터를 잘 못 맞히는 이유는?</summary>

모델이 표현할 수 있는 함수가 직선뿐이기 때문입니다(과소적합). 더 복잡한 관계를 배우려면 층을 쌓고 비선형 활성화 함수를 넣어야 합니다.

</details>

## 직접 고쳐보기

1. `lr`을 `0.1`로 키워보세요. 무슨 일이 일어나나요? `0.0001`은요? (학습률이 왜 가장 중요한 하이퍼파라미터인지 체감됩니다)
2. `w.grad.zero_()` 두 줄을 지우고 돌려보세요. 왜 망가질까요?
3. 데이터를 `Y = 3 * X**2 + 1`로 바꾸면 직선 모델은 잘 맞출 수 있을까요? loss가 어디서 멈추나요? → 이것이 다음 레슨에서 **층을 쌓고 비선형 함수를 넣는** 이유입니다.
4. `torch.optim.SGD`를 `torch.optim.Adam`으로 바꿔보세요. 수렴 속도가 어떻게 달라지나요?
