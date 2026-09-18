# 과적합과 검증 — 모델을 믿어도 되는지 아는 법

> ⏱ 45분 · CPU로 충분

**목표:** 머신러닝에서 가장 중요한 개념인 **과적합**을 눈으로 보고, 검증 데이터와 교차검증으로 모델을 올바르게 고르는 법을 배웁니다. 나중에 LLM을 파인튜닝할 때도 매번 마주치는 문제입니다.

## 실험: 곡선 맞추기

진짜 관계는 부드러운 사인 곡선인데, 우리는 노이즈가 섞인 점 20개만 관찰했다고 합시다.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2 * np.pi * x)
x_train = np.sort(rng.uniform(0, 1, 20));  y_train = true_f(x_train) + rng.normal(0, 0.25, 20)
x_test = np.sort(rng.uniform(0, 1, 200));  y_test = true_f(x_test) + rng.normal(0, 0.25, 200)

grid = np.linspace(0, 1, 300)
plt.plot(grid, true_f(grid), "g--", label="true function (hidden from the model)")
plt.scatter(x_train, y_train, label="training data (20 points)"); plt.legend(); plt.show()
```

**코드 읽기**

- 왜 진짜 데이터가 아니라 **만든 데이터**인가: 진짜 관계(`true_f`)를 우리가 알고 있어야 "모델이 노이즈를 외웠는지, 진짜 관계를 배웠는지"를 판정할 수 있습니다. 실제 데이터에서는 진짜 관계를 모르기 때문에 이런 실험이 불가능합니다. 개념을 배울 때는 정답을 아는 장난감 데이터가 가장 좋은 교구입니다.
- `np.random.default_rng(0)` — 난수 생성기를 시드 0으로 만듭니다. 이후의 `rng.uniform`(균등분포), `rng.normal`(정규분포) 호출이 매번 같은 값을 내므로 실험이 재현됩니다.
- `true_f = lambda x: np.sin(2 * np.pi * x)` — 0~1 구간에서 한 번 출렁이는 사인 곡선. 직선으로는 못 맞히고, 너무 복잡하지도 않은 "적당한" 목표입니다.
- `rng.normal(0, 0.25, 20)` — 평균 0, 표준편차 0.25인 노이즈. 현실의 측정값에는 항상 잡음이 섞여 있다는 것을 흉내 냅니다. 과적합이란 바로 이 잡음까지 외우는 것입니다.
- `x_test`를 200개나 만드는 이유: 평가는 점이 많을수록 정확합니다. 학습 데이터는 일부러 20개로 적게 두어 과적합이 쉽게 일어나게 합니다.
- `np.linspace(0, 1, 300)` — 곡선을 매끈하게 그리기 위한 촘촘한 x 값. 학습·평가와는 무관한 그림용입니다.
## 모델의 복잡도를 바꿔 가며 맞춰 보기

다항식의 차수가 모델의 "복잡도"입니다. 1차는 직선, 15차는 매우 구불구불할 수 있는 곡선입니다.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, degree in zip(axes, [1, 4, 15]):
    m = poly_model(degree).fit(x_train[:, None], y_train)
    tr = mean_squared_error(y_train, m.predict(x_train[:, None]))
    te = mean_squared_error(y_test, m.predict(x_test[:, None]))
    ax.plot(grid, true_f(grid), "g--"); ax.scatter(x_train, y_train)
    ax.plot(grid, m.predict(grid[:, None]), "r"); ax.set_ylim(-2, 2)
    ax.set_title(f"degree {degree}  |  train MSE {tr:.3f}  test MSE {te:.3f}")
plt.show()
```

- **1차 (과소적합):** 너무 단순해서 학습 데이터조차 못 맞힙니다. train·test 오차 모두 큼.
- **4차 (적절):** 진짜 관계를 잘 따라갑니다.
- **15차 (과적합):** 학습 데이터의 **노이즈까지 외워** 점들을 거의 다 지나가지만(train 오차 최소), 새 데이터에서는 엉망입니다.

**코드 읽기**

- `PolynomialFeatures(degree)` — 입력 x 하나를 `x, x², x³, …, x^degree` 여러 개의 특징으로 **늘려 줍니다.** 왜 이렇게 하나: 선형 회귀는 직선밖에 못 긋지만, 입력에 x²·x³을 특징으로 추가하면 "그 특징들의 선형 결합"은 곡선이 됩니다. 즉 모델은 그대로 두고 **입력을 바꿔서** 표현력을 키우는 방법입니다. 차수가 모델 복잡도를 조절하는 손잡이가 됩니다.
- `LinearRegression()` — 특징들의 가중합으로 y를 맞추는 가장 기본 회귀 모델. 정답이 정해진 계산(최소제곱법)으로 한 번에 풀리므로 반복 횟수 설정이 없습니다.
- `make_pipeline(PolynomialFeatures(degree), LinearRegression())` — 특징 생성과 회귀를 하나로 묶어, `poly_model(4)`처럼 차수만 바꿔 같은 모양의 모델을 찍어낼 수 있게 합니다. 이렇게 **모델을 만드는 함수**를 두면 비교 실험 코드가 짧아집니다.
- `x_train[:, None]` — scikit-learn은 입력이 2차원 `(샘플 수, 특징 수)`여야 합니다. 1차원 배열 `(20,)`을 `(20, 1)`로 바꾸는 numpy 표기입니다(`reshape(-1, 1)`과 같음). 초보자가 가장 자주 만나는 오류 중 하나가 이 모양 문제입니다.
- `mean_squared_error(정답, 예측)` — MSE. `model.score`가 주는 R² 대신 쓴 이유는, 오차가 "얼마나 커지는지"를 절대 크기로 보고 싶기 때문입니다.
- `ax.set_ylim(-2, 2)` — 15차 곡선은 데이터 밖에서 수십, 수백까지 튀어 오릅니다. 축을 고정하지 않으면 그 때문에 그림이 납작해져 아무것도 안 보입니다.

## 과적합의 서명: train은 내려가는데 test는 올라간다

```python
degrees = range(1, 16)
tr_err, te_err = [], []
for d in degrees:
    m = poly_model(d).fit(x_train[:, None], y_train)
    tr_err.append(mean_squared_error(y_train, m.predict(x_train[:, None])))
    te_err.append(mean_squared_error(y_test, m.predict(x_test[:, None])))

plt.plot(degrees, tr_err, "o-", label="train error"); plt.plot(degrees, te_err, "o-", label="test error")
plt.yscale("log"); plt.xlabel("model complexity (degree)"); plt.legend(); plt.show()
```

이 그래프의 모양을 기억해 두세요. 가로축을 "학습 스텝 수"로 바꾸면 딥러닝의 loss 곡선에서 똑같은 모양을 보게 됩니다.

**코드 읽기**

- 차수 1~15를 **반복문으로** 돌려 오차를 리스트에 모으는 것이 이 코드의 전부입니다. 실험은 이렇게 "설정 하나를 바꿔 가며 같은 측정을 반복하고 그래프로 겹쳐 보기"입니다. 앞으로 모든 레슨의 실습이 이 패턴입니다.
- `plt.yscale("log")` — train 오차는 0.01, test 오차는 1 이상으로 100배 차이가 납니다. 보통 축으로는 작은 값이 바닥에 붙어 안 보이므로 로그 축을 씁니다. 손실 그래프는 로그 축으로 보는 습관을 들이면 좋습니다.

## 과적합을 줄이는 세 가지 방법

**① 데이터를 늘린다** — 가장 확실한 방법입니다.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n in zip(axes, [20, 100, 1000]):
    xs = rng.uniform(0, 1, n); ys = true_f(xs) + rng.normal(0, 0.25, n)
    m = poly_model(15).fit(xs[:, None], ys)
    ax.plot(grid, true_f(grid), "g--"); ax.scatter(xs, ys, s=5, alpha=.5)
    ax.plot(grid, m.predict(grid[:, None]), "r"); ax.set_ylim(-2, 2); ax.set_title(f"degree 15, {n} points")
plt.show()
```

**② 모델을 단순하게 한다** — 위에서 본 것처럼 차수를 낮춥니다.

**③ 정규화(regularization)** — 복잡한 모델을 쓰되 **가중치가 커지는 것에 벌점**을 줍니다. 구불구불한 곡선은 큰 가중치가 필요하므로 곡선이 부드러워집니다. 딥러닝의 weight decay가 바로 이것입니다.

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, alpha in zip(axes, [0, 0.01, 100]):
    m = make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=alpha)).fit(x_train[:, None], y_train)
    te = mean_squared_error(y_test, m.predict(x_test[:, None]))
    ax.plot(grid, true_f(grid), "g--"); ax.scatter(x_train, y_train)
    ax.plot(grid, m.predict(grid[:, None]), "r"); ax.set_ylim(-2, 2); ax.set_title(f"degree 15 + penalty {alpha}  |  test MSE {te:.3f}")
plt.show()
```

벌점이 없으면(0) 과적합, 적당하면(0.01) 같은 15차 모델인데도 잘 맞고, 너무 세면(100) 곡선이 납작해져 과소적합입니다. 이렇게 사람이 정해야 하는 값을 **하이퍼파라미터**라고 합니다.

**코드 읽기**

- `Ridge(alpha=...)` — 선형 회귀에 **L2 벌점**을 붙인 모델입니다. 보통의 회귀는 "오차 합"만 줄이지만, Ridge는 `오차 합 + alpha × (가중치²의 합)`을 줄입니다. 가중치를 크게 쓰면 벌점을 받으므로, 꼭 필요한 만큼만 큰 가중치를 쓰게 됩니다. `alpha`가 벌점의 세기입니다.
- 왜 `StandardScaler`를 사이에 넣었나: x¹⁵은 x보다 값의 범위가 극단적으로 다릅니다. 벌점은 "가중치의 크기"에 걸리는데, 특징의 크기가 제각각이면 어떤 특징은 부당하게 세게, 어떤 특징은 약하게 벌을 받습니다. 스케일을 맞춰야 벌점이 공평해집니다. **벌점을 쓰는 모델에는 스케일링이 거의 필수**입니다.
- `alpha=0`이면 벌점이 없으므로 보통의 선형 회귀와 같아지고, 15차의 과적합이 그대로 나옵니다. 그래서 첫 번째 그림이 "기준"이 됩니다.

### 정규화의 종류: 언제 무엇을 쓰나

"정규화(regularization)"는 **모델이 지나치게 복잡해지는 것을 억제하는 모든 장치**의 총칭입니다. 고전 머신러닝에서 자주 쓰는 것들입니다. (딥러닝에서 쓰는 dropout·데이터 증강 등은 [학습 잘 시키는 법](#24-training-recipes)에서 따로 정리합니다.)

| 이름 | 무엇을 하나 | 언제 쓰나 |
|---|---|---|
| **L2 (Ridge, weight decay)** | 가중치²의 합에 벌점. 모든 가중치를 고르게 작게 만듦 | 기본 선택. 특징이 많고 서로 상관이 있을 때 안정적 |
| **L1 (Lasso)** | \|가중치\|의 합에 벌점. 쓸모없는 특징의 가중치를 **정확히 0**으로 만듦 | 특징이 수백 개인데 실제로 중요한 건 몇 개뿐일 때(특징 선택 효과) |
| **Elastic Net** | L1 + L2 | 둘의 장점이 모두 필요할 때 |
| **모델 크기 제한** | 차수, 트리 깊이, 층 수 등 복잡도 자체를 줄임 | 데이터가 적을 때 가장 직접적인 방법 |
| **조기 종료** | 검증 오차가 올라가기 시작하면 학습을 멈춤 | 반복 학습하는 모든 모델(부스팅, 신경망) |
| **데이터 추가** | 진짜 데이터든 증강이든 샘플 수를 늘림 | 가능하다면 언제나. 효과가 가장 확실 |

왜 벌점이 과적합을 막는가: 구불구불한 곡선은 큰 양수와 큰 음수 가중치가 서로 상쇄하며 만들어집니다. 가중치의 크기에 비용을 물리면 그런 "무리한" 곡선이 손해가 되어, 모델은 데이터를 대체로 설명하는 부드러운 곡선을 택합니다. 딥러닝의 `weight_decay`가 정확히 L2 벌점입니다.

## 하이퍼파라미터는 무엇으로 고르나: 검증 데이터

차수나 벌점 세기를 **test 오차가 가장 낮은 것**으로 고르면 될까요? 안 됩니다. 그 순간 test 데이터가 모델 선택에 쓰였으므로, 더 이상 "처음 보는 데이터"가 아닙니다. 그래서 데이터를 셋으로 나눕니다.

| 이름 | 용도 |
|---|---|
| train | 모델 학습 |
| validation (검증) | 하이퍼파라미터·모델 선택, 학습을 언제 멈출지 결정 |
| test | 모든 결정이 끝난 뒤 **딱 한 번** 최종 성적 확인 |

데이터가 적을 때는 train을 K등분해서 돌아가며 검증용으로 쓰는 **교차검증(cross-validation)**을 합니다.

```python
from sklearn.model_selection import cross_val_score

cv_err = [-cross_val_score(poly_model(d), x_train[:, None], y_train, cv=5, scoring="neg_mean_squared_error").mean() for d in degrees]
best = degrees[int(np.argmin(cv_err))]
print("교차검증이 고른 차수:", best)      # test 데이터를 전혀 보지 않고 골랐습니다

final = poly_model(best).fit(x_train[:, None], y_train)
print("최종 test 오차:", mean_squared_error(y_test, final.predict(x_test[:, None])))
```

**코드 읽기**

- `cross_val_score(model, X, y, cv=5, scoring=...)` — 학습 데이터를 5조각으로 나눠, 4조각으로 학습하고 1조각으로 평가하기를 조각을 바꿔 가며 5번 반복한 점수 5개를 돌려줍니다. 직접 짜면 20줄이 넘는 반복문을 한 줄로 대신합니다.
- `scoring="neg_mean_squared_error"` — scikit-learn은 "점수는 클수록 좋다"는 약속을 지키려고 오차에 **마이너스**를 붙여 돌려줍니다. 그래서 앞에 `-`를 붙여 다시 오차로 바꾸고 `.mean()`으로 5개를 평균합니다.
- `np.argmin(cv_err)` — 오차가 가장 작은 인덱스. `degrees`가 1부터 시작하는 `range`라서 인덱스를 다시 차수로 바꿔 줍니다.
- 마지막에 `final`을 **전체 학습 데이터로 다시** 학습시키는 이유: 교차검증은 차수를 고르기 위한 것이고, 그 과정의 모델들은 데이터의 80%만 봤습니다. 차수가 정해졌으면 데이터를 전부 써서 최종 모델을 만듭니다.
## 핵심 정리

- **과적합:** 학습 데이터의 노이즈까지 외운 상태. train 오차는 낮은데 test 오차가 높습니다.
- **과소적합:** 모델이 너무 단순해 train 오차부터 높습니다.
- 해법은 더 많은 데이터, 더 단순한 모델, 정규화입니다.
- 모델·하이퍼파라미터 선택은 **검증 데이터**로, 최종 성적은 **테스트 데이터**로 한 번만 확인합니다.
- "train은 내려가는데 validation이 올라가기 시작하는 지점"이 멈출 때입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. train 오차 0.01, test 오차 0.9인 모델과 train 오차 0.5, test 오차 0.55인 모델. 각각 어떤 상태이고 무엇을 해야 하나요?</summary>

앞은 과적합입니다(격차가 큼): 데이터 추가, 정규화, 모델 축소. 뒤는 과소적합에 가깝습니다(둘 다 높고 격차는 작음): 모델을 키우거나 더 오래 학습하거나 더 좋은 특징을 넣습니다.

</details>

<details><summary>Q2. test 데이터로 하이퍼파라미터를 여러 번 조정한 뒤 얻은 test 성능은 왜 믿을 수 없나요?</summary>

여러 설정 중 우연히 그 test 세트에 잘 맞는 것을 고르게 되어, test 성능이 실제보다 낙관적으로 나옵니다. 간접적으로 test 데이터에 과적합한 것입니다.

</details>

<details><summary>Q3. 데이터를 1000개로 늘리자 15차 모델도 잘 동작했습니다. 왜일까요?</summary>

점이 촘촘해지면 곡선이 노이즈를 따라 제멋대로 구불거릴 "틈"이 없어집니다. 모든 점을 그럭저럭 지나려면 진짜 관계에 가까워질 수밖에 없습니다. 대규모 모델이 대규모 데이터와 함께 쓰이는 이유입니다.

</details>

## 직접 고쳐보기

1. 노이즈 크기 `0.25`를 `0.05`와 `0.6`으로 바꿔 보세요. 최적 차수가 어떻게 변하나요?
2. 학습 데이터를 20개에서 10개로 줄이면 과적합이 시작되는 차수가 어떻게 달라지나요?
3. `Ridge`의 `alpha`를 교차검증으로 골라 보세요. (`for alpha in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]`)
4. (도전) `sklearn.model_selection.GridSearchCV`로 차수와 alpha를 동시에 탐색해 보세요.
