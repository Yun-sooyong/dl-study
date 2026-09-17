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
