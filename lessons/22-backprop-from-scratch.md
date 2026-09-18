# 역전파 직접 구현 — 자동미분 없이 신경망 학습시키기

> ⏱ 90분 · CPU로 충분 · 선수 지식: [필요한 수학](#02-math-minimum), [첫 신경망](#21-mlp-mnist)

**목표:** `loss.backward()`가 안에서 무엇을 하는지 **numpy로 직접 짜서** 확인합니다. 2층 신경망의 순전파와 역전파를 손으로 구현하고, 수치 미분과 PyTorch autograd로 검증한 뒤, 그 코드로 손글씨 분류기를 실제로 학습시킵니다. 이 레슨을 마치면 "딥러닝 프레임워크가 하는 일"에 블랙박스가 남지 않습니다.

## 왜 굳이 직접 짜는가

PyTorch가 다 해 주는데 왜 손으로 하나? 세 가지 이유입니다.

1. **디버깅:** 기울기가 사라지거나 폭발하는 문제, 학습이 안 되는 이유는 역전파의 구조를 알아야 진단할 수 있습니다.
2. **새 구조 설계:** 어텐션, LoRA, 새로운 손실 함수를 만들 때 "기울기가 어디로 어떻게 흐르는지"를 생각해야 합니다.
3. **이해:** 한 번 손으로 짜 본 사람과 아닌 사람은 같은 코드를 봐도 보이는 것이 다릅니다.

이 레슨은 단 한 번만 손으로 하고, 다음 레슨부터는 다시 autograd를 씁니다.

## 데이터

8×8 손글씨 숫자([첫 모델](#10-ml-first-model)에서 쓴 것). 작아서 numpy로도 몇 초면 학습됩니다.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0                                   # 0~16 → 0~1
Y = np.eye(10)[digits.target]                            # 정답을 one-hot으로: 3 → [0,0,0,1,0,0,0,0,0,0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print(X_train.shape, Y_train.shape)                      # (1347, 64) (1347, 10)
```

**코드 읽기**

- `X / 16.0` — 픽셀 값을 0~1로. 입력 크기를 맞추는 것은 [첫 신경망](#21-mlp-mnist)의 `ToTensor()`와 같은 이유입니다.
- `np.eye(10)[target]` — 단위행렬의 `target`번째 행을 꺼내면 one-hot 벡터가 됩니다. cross entropy를 손으로 계산하려면 정답을 이 형태로 두는 것이 편합니다(정답 위치만 1이므로 `Y * log(P)`의 합이 곧 정답 확률의 로그).

## 모델: 2층 신경망을 행렬로

```
입력 x (64)  →  z1 = x @ W1 + b1 (32)  →  h = ReLU(z1)  →  z2 = h @ W2 + b2 (10)  →  p = softmax(z2)  →  loss = -log p[정답]
```

파라미터는 `W1 (64×32), b1 (32), W2 (32×10), b2 (10)` 네 개입니다.

```python
rng = np.random.default_rng(0)
W1 = rng.normal(0, np.sqrt(2 / 64), (64, 32))            # He 초기화: 분산 2/입력수
b1 = np.zeros(32)
W2 = rng.normal(0, np.sqrt(2 / 32), (32, 10))
b2 = np.zeros(10)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def forward(X):
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0)                                # ReLU
    z2 = h @ W2 + b2
    p = softmax(z2)
    return z1, h, z2, p                                  # 역전파에 필요하므로 중간값을 모두 돌려줌

def loss_fn(p, Y):
    return -np.mean(np.sum(Y * np.log(p + 1e-12), axis=1))   # 배치 평균 cross entropy

z1, h, z2, p = forward(X_train[:5])
print("출력 확률 shape:", p.shape, " 각 행의 합:", p.sum(axis=1).round(3))
print("초기 loss:", loss_fn(p, Y_train[:5]).round(3), " 예상 ln(10) =", np.log(10).round(3))
```

**코드 읽기**

- `rng.normal(0, np.sqrt(2 / 64), ...)` — 가중치를 0이 아닌 **작은 난수**로 시작합니다. 0으로 시작하면 모든 뉴런이 똑같이 계산되고 똑같은 기울기를 받아 영원히 똑같은 값에 머뭅니다(대칭 문제). 분산을 `2/입력 수`로 잡는 것은 He 초기화로, ReLU 층을 여러 개 쌓아도 값의 크기가 층마다 커지거나 작아지지 않게 하는 규칙입니다. `nn.Linear`가 기본으로 비슷한 초기화를 합니다.
- `b1 = np.zeros(32)` — 편향은 0으로 시작해도 됩니다(대칭 문제는 가중치에서만 생김).
- `forward`가 중간값(`z1, h, z2, p`)을 **모두** 돌려주는 이유 — 역전파에서 각 단계의 미분을 계산하려면 순전파 때의 값이 필요합니다. PyTorch가 "계산 그래프를 기록한다"는 것은 바로 이 중간값들을 저장해 두는 것입니다. 학습이 추론보다 메모리를 훨씬 많이 쓰는 이유이기도 합니다.
- `softmax`의 `axis=1, keepdims=True` — 배치 `(B, 10)`에서 행(샘플)마다 따로 softmax. `keepdims`는 `(B, 1)` 모양을 유지해 브로드캐스팅이 되게 합니다.
- `np.log(p + 1e-12)` — 확률이 정확히 0이면 log가 −∞가 되므로 아주 작은 값을 더해 막습니다. PyTorch의 `CrossEntropyLoss`는 이런 처리를 내부에서 더 정교하게 합니다.
- 초기 loss ≈ 2.30 = ln 10 — [필요한 수학](#02-math-minimum)에서 배운 "찍는 수준" 확인입니다.

## 역전파: 뒤에서부터 미분을 곱해 나간다

손실에서 출발해 각 파라미터까지 연쇄법칙을 적용합니다. 각 단계의 미분 규칙은 아래 네 가지뿐입니다.

| 순전파 | 역전파 (위에서 받은 기울기 `g`를 아래로 넘김) | 이유 |
|---|---|---|
| `p = softmax(z2)`, `loss = CE(p, Y)` | `dz2 = (p − Y) / B` | softmax + cross entropy를 합치면 미분이 이렇게 단순해짐 (아래 설명) |
| `z2 = h @ W2 + b2` | `dW2 = h.T @ dz2`, `db2 = dz2.sum(0)`, `dh = dz2 @ W2.T` | 행렬곱의 미분: 다른 쪽 피연산자를 전치해서 곱함 |
| `h = ReLU(z1)` | `dz1 = dh * (z1 > 0)` | ReLU는 양수 구간에서 기울기 1, 음수 구간에서 0 |
| `z1 = X @ W1 + b1` | `dW1 = X.T @ dz1`, `db1 = dz1.sum(0)` | 위와 같은 행렬곱 규칙 |

```python
def backward(X, Y, z1, h, z2, p):
    B = X.shape[0]
    dz2 = (p - Y) / B                    # 손실을 z2로 미분. 배치 평균이므로 B로 나눔
    dW2 = h.T @ dz2                      # (32,B) @ (B,10) = (32,10) ← W2와 같은 shape
    db2 = dz2.sum(axis=0)                # (10,)
    dh = dz2 @ W2.T                      # (B,10) @ (10,32) = (B,32) ← h와 같은 shape
    dz1 = dh * (z1 > 0)                  # ReLU의 미분: 켜져 있던 뉴런만 기울기 통과
    dW1 = X.T @ dz1                      # (64,B) @ (B,32) = (64,32)
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2

grads = backward(X_train[:5], Y_train[:5], *forward(X_train[:5]))
print([g.shape for g in grads])         # 파라미터와 같은 shape이어야 함
```

**코드 읽기**

- `dz2 = (p − Y) / B` — softmax와 cross entropy를 따로 미분하면 복잡하지만, 둘을 합친 미분은 "예측 확률 − 정답 one-hot"으로 놀랍도록 단순합니다. 정답 클래스의 확률이 1보다 작으면 그 로짓을 올리고, 나머지 클래스는 확률만큼 내리라는 뜻입니다. PyTorch가 softmax를 손실 함수 안에 넣어 두는 이유가 이 단순함(과 수치 안정성)입니다.
- **행렬곱의 미분 규칙** — `z = h @ W`일 때 `dW = h.T @ dz`, `dh = dz @ W.T`. 외울 필요 없이 **shape을 맞추면** 저절로 나옵니다. `dW`는 `W`와 같은 `(32, 10)`이어야 하고, 그것을 `h (B,32)`와 `dz (B,10)`으로 만들 수 있는 유일한 곱이 `h.T @ dz`입니다. 실무에서 새 층의 역전파를 짤 때도 이렇게 shape으로 검산합니다.
- `dz1 = dh * (z1 > 0)` — `(z1 > 0)`은 True/False 행렬이고 곱하면 1/0으로 작동합니다. 순전파에서 0으로 잘렸던 뉴런은 기울기도 0(그 뉴런을 조금 바꿔도 출력이 안 변하므로). 순전파의 `z1`이 필요한 이유가 여기 있습니다.
- `db = dz.sum(axis=0)` — 편향은 배치의 모든 샘플에 더해졌으므로 기울기는 샘플 방향으로 합칩니다.
- 기울기의 shape이 파라미터와 같은지 `print`로 확인 — 역전파 코드를 짤 때 가장 먼저 하는 검사입니다.

## 검증 1: 수치 미분과 비교

내 역전파가 맞는지 어떻게 아나? [필요한 수학](#02-math-minimum)의 `numeric_grad`를 파라미터 하나하나에 적용해 비교합니다. 느리지만 확실합니다.

```python
def numeric_grad_check(X, Y, param, analytic, n=5, h=1e-5):
    """param의 무작위 원소 n개에 대해 수치 미분과 역전파 결과를 비교"""
    worst = 0
    for _ in range(n):
        idx = tuple(rng.integers(s) for s in param.shape)
        old = param[idx]
        param[idx] = old + h; lp = loss_fn(forward(X)[3], Y)
        param[idx] = old - h; lm = loss_fn(forward(X)[3], Y)
        param[idx] = old
        num = (lp - lm) / (2 * h)
        worst = max(worst, abs(num - analytic[idx]) / (abs(num) + abs(analytic[idx]) + 1e-12))
    return worst

Xb, Yb = X_train[:20], Y_train[:20]
grads = backward(Xb, Yb, *forward(Xb))
for name, param, g in zip(["W1", "b1", "W2", "b2"], [W1, b1, W2, b2], grads):
    print(f"{name}: 최대 상대 오차 {numeric_grad_check(Xb, Yb, param, g):.2e}")   # 1e-6 이하면 정확
```

**코드 읽기**

- 원소 하나를 `+h`, `−h`로 바꿔 손실을 두 번 계산하고 차이를 `2h`로 나눕니다. 파라미터가 2천 개라 전부 검사하면 손실 계산 4천 번이 필요하므로 무작위로 5개씩만 뽑습니다.
- `param[idx] = old`로 반드시 **원래 값으로 되돌리는** 것을 잊으면 검사 자체가 모델을 망가뜨립니다.
- 상대 오차 `|num − analytic| / (|num| + |analytic|)`를 쓰는 이유 — 기울기의 절대 크기가 제각각이라 절대 오차로는 판단이 안 됩니다. 1e-6 근처면 맞고, 1e-2 이상이면 역전파 어딘가가 틀린 것입니다. 틀린 파라미터 이름이 어디부터 틀렸는지 알려줍니다(예: `W1`만 틀렸다면 `dz1` 계산을 의심).

## 검증 2: PyTorch autograd와 비교

```python
import torch
tW1, tb1, tW2, tb2 = (torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2))
tX, tY = torch.tensor(Xb), torch.tensor(Yb)
th = torch.relu(tX @ tW1 + tb1)
tp = torch.softmax(th @ tW2 + tb2, dim=1)
tloss = -(tY * torch.log(tp)).sum(dim=1).mean()
tloss.backward()
for name, tg, g in zip(["W1", "b1", "W2", "b2"], [tW1.grad, tb1.grad, tW2.grad, tb2.grad], grads):
    print(f"{name}: autograd와 최대 차이 {np.abs(tg.numpy() - g).max():.2e}")
```

**코드 읽기**

- 같은 파라미터, 같은 배치로 PyTorch가 계산한 `.grad`와 우리의 `grads`가 소수점 열 자리 이상 일치합니다. **autograd가 하는 일이 우리가 방금 짠 `backward`와 같다**는 것을 눈으로 확인한 것입니다.
- 차이가 있다면 순전파 정의가 미묘하게 다르다는 뜻입니다(예: `1e-12`를 더한 곳).

## 학습: 프레임워크 없이 경사하강법

```python
lr, batch_size = 0.5, 64
for epoch in range(30):
    lr_e = lr * 0.9 ** epoch                           # 에폭마다 학습률을 10%씩 줄임 (가장 단순한 스케줄러)
    perm = rng.permutation(len(X_train))
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        Xb, Yb = X_train[idx], Y_train[idx]
        dW1, db1, dW2, db2 = backward(Xb, Yb, *forward(Xb))
        W1 -= lr_e * dW1; b1 -= lr_e * db1             # 옵티마이저 = 이 두 줄 (SGD)
        W2 -= lr_e * dW2; b2 -= lr_e * db2
    if epoch % 5 == 4:
        acc = (forward(X_test)[3].argmax(1) == Y_test.argmax(1)).mean()
        print(f"epoch {epoch+1:2d}  train loss {loss_fn(forward(X_train)[3], Y_train):.3f}  test acc {acc:.3f}")
```

**코드 읽기**

- 이 루프에 PyTorch는 한 줄도 없습니다. `forward → backward → 파라미터 -= lr × 기울기`. [텐서와 자동미분](#20-tensor-autograd)의 네 줄이 실제로 하는 계산이 전부 드러나 있습니다.
- `rng.permutation`으로 에폭마다 순서를 섞는 것은 `DataLoader(shuffle=True)`에 해당합니다.
- `lr=0.5`가 큰 이유 — 손실을 배치 **평균**으로 정의했고 입력이 0~1이라 기울기가 작습니다. Adam이 아니라 순수 SGD이므로 학습률을 직접 맞춰야 하고, 이 값은 실험으로 찾은 것입니다(실습 1번).
- `lr_e = lr * 0.9 ** epoch` — 학습률 감소. 이것 없이 0.5로 고정하면 손실이 중간중간 튀어 오릅니다(직접 지워서 확인해 보세요). 큰 학습률로 빨리 내려간 뒤 작은 학습률로 마무리하는 것이 [학습 잘 시키는 법](#24-training-recipes)에서 배울 스케줄러의 원형입니다.
- 30 에폭 후 테스트 정확도 약 96%. [첫 모델](#10-ml-first-model)의 로지스틱 회귀(97%)와 비슷한 수준을 **직접 짠 신경망**으로 달성했습니다.

## 여기서 보이는 것들

- **기울기 소실/폭발:** 역전파는 미분의 곱입니다. 층이 20개이고 각 단계의 미분이 0.5씩이면 첫 층의 기울기는 `0.5²⁰ ≈ 1e-6`으로 사라지고, 2씩이면 백만 배로 폭발합니다. He 초기화, ReLU, 잔차 연결, LayerNorm은 모두 이 곱이 1 근처에 머물게 하는 장치입니다.
- **메모리:** 역전파에 `z1, h`가 필요하므로 순전파의 중간값을 전부 저장해야 합니다. 배치와 시퀀스가 길수록, 층이 깊을수록 이 저장량이 커집니다. 학습 시 GPU 메모리 부족의 주범이고, gradient checkpointing은 일부를 버렸다가 다시 계산하는 타협입니다.
- **얼린 파라미터:** `W1`을 학습하지 않기로 했다면 `dW1`을 계산할 필요가 없지만, `dh → dz1`은 여전히 계산해야 그 아래 층으로 기울기가 흐릅니다. [미니 VLM](#41-mini-vlm)에서 "얼린 LLM을 기울기가 통과해 프로젝터에 도달한다"가 바로 이것입니다.
- **LoRA:** `z = h @ (W + B A)`에서 `W`는 얼리고 `A, B`의 기울기만 구합니다. `dB = (h @ A.T).T @ dz` 같은 식으로, 규칙은 위와 같습니다.

## 핵심 정리

- 역전파 = 손실에서 파라미터까지 **각 단계의 미분을 뒤에서부터 곱해 나가는 것.** 규칙은 몇 개 안 됩니다(행렬곱, 활성화 함수, softmax+CE).
- 행렬곱의 미분은 shape을 맞추면 나옵니다: `dW = 입력.T @ dz`, `d입력 = dz @ W.T`.
- 순전파의 중간값을 저장해야 역전파가 가능합니다. 그것이 학습의 메모리 비용입니다.
- 새로 짠 역전파는 반드시 **수치 미분으로 검증**합니다.
- `loss.backward()`는 마법이 아니라 방금 짠 30줄의 일반화입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 가중치를 전부 0으로 초기화하면 어떻게 되나요? 코드로 확인하려면?</summary>

같은 층의 모든 뉴런이 같은 출력, 같은 기울기를 받아 영원히 같은 값을 유지합니다. 뉴런이 32개 있어도 1개짜리와 같습니다. `W1 = np.zeros((64, 32))`로 바꿔 학습시키면 정확도가 로지스틱 회귀 수준에도 못 미치는 것을 볼 수 있습니다.

</details>

<details><summary>Q2. ReLU 대신 sigmoid를 쓰면 역전파의 어느 줄이 바뀌고, 깊은 모델에서 무엇이 문제가 되나요?</summary>

`dz1 = dh * (z1 > 0)`이 `dz1 = dh * h * (1 − h)`로 바뀝니다(sigmoid의 미분). 이 값은 최대 0.25라 층을 지날 때마다 기울기가 최소 1/4씩 줄어들어, 깊은 모델에서는 앞쪽 층의 기울기가 사라집니다.

</details>

<details><summary>Q3. 수치 미분으로 검증하는데 상대 오차가 1e-3으로 나왔습니다. 무엇을 의심하나요?</summary>

역전파 코드의 오류일 가능성이 큽니다. 다만 ReLU의 꺾이는 지점(z1 ≈ 0) 근처 원소는 수치 미분 자체가 불안정하므로, 여러 원소를 검사해 대부분이 1e-6 근처인지 봅니다. 모든 원소가 1e-3 이상이면 코드 오류입니다.

</details>

<details><summary>Q4. 배치 크기를 2배로 하면 `dz2 = (p − Y) / B`의 값은 어떻게 되고, 학습률은 어떻게 조정해야 하나요?</summary>

각 샘플의 기여가 절반이 되지만 샘플이 두 배이므로 기울기의 크기는 비슷합니다(평균이므로). 다만 더 많은 샘플의 평균이라 기울기가 덜 요동치므로 학습률을 조금 키워도 안정적입니다. 실무에서는 배치를 k배 하면 학습률도 k배(또는 √k배) 하는 규칙을 씁니다.

</details>

## 직접 고쳐보기

1. `lr`을 0.05, 0.5, 5로 바꿔 보세요. 어느 것이 발산하나요? Adam이 없는 순수 SGD에서 학습률 감각을 익히는 실험입니다.
2. 은닉층 크기를 32 → 128로, 층을 3개로 늘려 보세요. `forward`와 `backward`에 각각 무엇을 추가해야 하나요? 수치 미분 검사를 통과시키세요.
3. ReLU를 `tanh`로 바꿔 보세요. 미분은 `1 − tanh²`입니다. `numeric_grad_check`가 통과하는지 확인하세요.
4. 학습 루프에 **모멘텀**을 넣어 보세요: `v = 0.9 * v + dW; W -= lr * v`. 수렴이 빨라지나요? 이것이 Adam의 절반입니다(나머지 절반은 기울기 크기로 나누는 것).
5. L2 정규화를 손실에 더해 보세요: `loss += 0.001 * (W1**2).sum()`. 역전파에서는 `dW1 += 2 * 0.001 * W1`이 됩니다. 이것이 weight decay의 정체입니다.
6. (도전) 이 코드를 [미니 GPT](#31-mini-gpt)의 어텐션에 적용하려면 `softmax(QKᵀ/√d) V`의 역전파가 필요합니다. `dV = attᵀ @ dout`부터 시작해 나머지를 유도해 보세요. Karpathy의 micrograd/nanoGPT 강의가 좋은 안내서입니다.
