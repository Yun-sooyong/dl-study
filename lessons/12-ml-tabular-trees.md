# 표 데이터와 트리 모델 — 회귀, 랜덤 포레스트, 부스팅

> ⏱ 50분 · CPU로 충분

**목표:** 엑셀 같은 **표 데이터**로 숫자를 예측(회귀)합니다. 기준선(baseline)을 세우고, 결정 트리 → 랜덤 포레스트 → 그래디언트 부스팅으로 모델을 키워 가며 비교하는 실험 습관을 익힙니다.

> 실무에서 표 데이터는 지금도 딥러닝보다 **트리 기반 모델**이 더 잘 맞는 경우가 많습니다. 딥러닝이 압도적인 분야는 이미지·텍스트·음성처럼 "날것의" 데이터입니다.

## 데이터: 캘리포니아 집값

한 행이 한 지역, 열이 그 지역의 특징입니다. 맞혀야 할 값은 집값 중앙값(단위: 10만 달러)입니다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame                     # pandas DataFrame: 표 데이터를 다루는 표준 도구
print(df.shape)
df.head()
```

```python
print(df.describe().T[["mean", "min", "max"]])   # 열마다 범위가 제각각입니다
df.hist(bins=40, figsize=(12, 7)); plt.tight_layout(); plt.show()
```

모델을 만들기 전에 **데이터를 먼저 들여다보는 것**이 가장 가성비 좋은 작업입니다. 이상한 값(예: 방 개수 평균이 140개인 지역)이나 잘린 값(집값이 5.0에서 뭉쳐 있음)이 보이나요?

```python
from sklearn.model_selection import train_test_split

X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 기준선부터 세운다

"무조건 평균값으로 답하기"보다 못한 모델은 의미가 없습니다. 가장 멍청한 방법의 점수를 먼저 알아 둬야 내 모델이 얼마나 좋은지 말할 수 있습니다.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

results = {}
def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {"train MAE": mean_absolute_error(y_train, model.predict(X_train)),
                     "test MAE": mean_absolute_error(y_test, pred), "test R2": r2_score(y_test, pred)}
    print(f"{name:22s} train MAE {results[name]['train MAE']:.3f} | test MAE {results[name]['test MAE']:.3f} | R2 {results[name]['test R2']:.3f}")
    return model

evaluate("평균으로 찍기", DummyRegressor())
linear = evaluate("선형 회귀", LinearRegression())
```

- **MAE (평균 절대 오차):** 평균적으로 몇 (10만) 달러 틀리는지. 해석이 쉽습니다.
- **R²:** 1이면 완벽, 0이면 평균으로 찍는 것과 같음.

선형 회귀는 `집값 = w1×소득 + w2×집 나이 + … + b`입니다. 학습된 가중치를 보면 모델의 "생각"을 읽을 수 있습니다.

```python
print(pd.Series(linear.coef_, index=X.columns).round(3))
```

## 결정 트리: 스무고개로 예측하기

"소득이 5 이상인가? → 위도가 37.9 이하인가? → …" 질문을 따라 내려가 도착한 칸의 평균값으로 답합니다. 직선으로 표현할 수 없는 관계(위치에 따른 집값 등)를 잡아냅니다.

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree

small = DecisionTreeRegressor(max_depth=2, random_state=0).fit(X_train, y_train)
plt.figure(figsize=(13, 5)); plot_tree(small, feature_names=list(X.columns), filled=True, fontsize=9); plt.show()
```

깊이를 제한하지 않으면 트리는 학습 데이터를 **완벽하게 외울 때까지** 가지를 칩니다. 앞 레슨에서 본 과적합입니다.

```python
evaluate("트리 (깊이 제한 없음)", DecisionTreeRegressor(random_state=0))   # train MAE가 0에 가깝습니다
evaluate("트리 (깊이 8)", DecisionTreeRegressor(max_depth=8, random_state=0))
```

## 앙상블: 약한 모델 여럿이 강한 모델 하나를 이긴다

- **랜덤 포레스트:** 데이터와 특징을 무작위로 달리해 트리를 수백 그루 만들고 **평균**을 냅니다. 개별 트리의 과적합이 서로 상쇄됩니다.
- **그래디언트 부스팅:** 트리를 하나씩 차례로 추가하되, 새 트리는 **지금까지의 오차**를 맞히도록 학습합니다. 표 데이터 대회의 단골 우승 모델입니다(XGBoost, LightGBM이 이 계열).

```python
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

forest = evaluate("랜덤 포레스트", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0))
boost = evaluate("그래디언트 부스팅", HistGradientBoostingRegressor(random_state=0))
pd.DataFrame(results).T.round(3)
```

## 모델은 무엇을 보고 판단했을까

특징 하나의 값을 무작위로 섞었을 때 성능이 얼마나 떨어지는지 봅니다(permutation importance). 많이 떨어질수록 중요한 특징입니다.

```python
from sklearn.inspection import permutation_importance

imp = permutation_importance(boost, X_test, y_test, n_repeats=5, random_state=0)
pd.Series(imp.importances_mean, index=X.columns).sort_values().plot.barh(); plt.xlabel("drop in R2 when shuffled"); plt.show()

plt.scatter(y_test, boost.predict(X_test), s=3, alpha=.3); plt.plot([0, 5], [0, 5], "r")
plt.xlabel("true"); plt.ylabel("predicted"); plt.show()    # 대각선에 가까울수록 정확
```

## 핵심 정리

- 표 데이터는 pandas로 **먼저 들여다보고**, 기준선(평균으로 찍기, 선형 모델)부터 세웁니다.
- 회귀의 평가 지표는 MAE, R² 등. 분류의 정확도에 해당합니다.
- 결정 트리는 제한 없이 키우면 학습 데이터를 외웁니다(과적합).
- 여러 트리를 평균(랜덤 포레스트)하거나 차례로 오차를 보정(부스팅)하면 훨씬 강해집니다.
- 표 데이터 → 트리 앙상블, 이미지·텍스트 → 딥러닝이 기본 선택입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 깊이 제한 없는 트리의 train MAE가 거의 0인데 test MAE는 선형 회귀와 비슷합니다. 무슨 뜻인가요?</summary>

학습 데이터의 각 행을 사실상 통째로 외운 과적합 상태입니다. train 성능은 모델 실력을 말해 주지 않습니다.

</details>

<details><summary>Q2. 랜덤 포레스트에서 트리마다 데이터와 특징을 무작위로 다르게 주는 이유는?</summary>

모든 트리가 똑같으면 평균을 내도 그대로입니다. 서로 **다르게 틀리는** 트리들을 만들어야 평균을 냈을 때 오차가 상쇄됩니다.

</details>

<details><summary>Q3. 새 프로젝트에서 복잡한 모델부터 만들지 않고 기준선부터 세우는 이유는?</summary>

① 복잡한 모델의 점수가 좋은 것인지 판단할 기준이 생기고, ② 데이터 로딩·분할·평가 코드의 버그를 단순한 상황에서 먼저 잡을 수 있으며, ③ 단순한 모델로 충분한 경우도 많기 때문입니다.

</details>

## 직접 고쳐보기

1. `max_depth`를 2, 4, 8, 16, None으로 바꿔 train/test MAE를 표로 만들어 보세요. 어디서부터 과적합인가요?
2. `RandomForestRegressor`의 `n_estimators`를 1, 10, 100, 300으로 바꿔 보세요. 성능은 어디서 포화되나요?
3. 위도·경도 열을 빼고(`X.drop(columns=["Latitude", "Longitude"])`) 학습하면 얼마나 나빠지나요? 선형 회귀와 부스팅 중 어느 쪽이 더 크게 나빠지나요? 왜일까요?
4. 새 특징을 만들어 넣어 보세요(예: `df["AveRooms"] / df["AveOccup"]`). 이런 작업을 특징 공학(feature engineering)이라고 합니다. 딥러닝은 이 작업을 모델이 스스로 하게 만든 것입니다.
5. (도전) `sklearn.datasets.load_breast_cancer`(분류 문제)에 같은 모델들의 `Classifier` 버전을 적용해 비교표를 만들어 보세요.
