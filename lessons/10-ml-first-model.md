# 머신러닝이란 — 첫 모델 만들기

> ⏱ 40분 · CPU로 충분 · 선수 지식: 파이썬 기초

**목표:** "데이터로 모델을 학습시킨다"는 것이 무엇인지, scikit-learn으로 10줄짜리 모델을 만들며 전체 흐름을 익힙니다. 이 흐름은 LLM 학습까지 똑같이 이어집니다.

## 프로그래밍과 머신러닝의 차이

- **프로그래밍:** 사람이 규칙을 짠다. `if 꽃잎 길이 > 2.5: ...`
- **머신러닝:** 사람은 **예시(데이터)**를 주고, 규칙은 컴퓨터가 찾는다.

손글씨 숫자를 구분하는 규칙을 `if`문으로 짤 수 있을까요? 거의 불가능합니다. 하지만 "이 그림은 3, 저 그림은 7"이라는 예시는 얼마든지 줄 수 있습니다. 이것이 머신러닝이 필요한 이유입니다.

머신러닝의 모든 작업은 다음 흐름을 따릅니다.

```
데이터 준비 → 학습용/평가용으로 나누기 → 모델 학습(fit) → 처음 보는 데이터로 평가 → 개선
```

## 데이터 살펴보기

8×8 픽셀의 손글씨 숫자 1797장입니다. 이미지 한 장은 숫자 64개짜리 벡터, 정답(라벨)은 0~9입니다.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape, y.shape)      # (1797, 64) 입력 , (1797,) 정답
print(X[0].reshape(8, 8))    # 이미지 = 숫자들의 배열
print("정답:", y[0])

fig, axes = plt.subplots(1, 8, figsize=(10, 2))
for ax, img, label in zip(axes, digits.images, y):
    ax.imshow(img, cmap="gray_r"); ax.set_title(label); ax.axis("off")
plt.show()
```

- **X (입력, feature):** 모델이 보는 것. 샘플 수 × 특징 수의 표.
- **y (정답, label):** 모델이 맞혀야 하는 것.

## 가장 중요한 습관: 데이터를 나눈다

모델을 학습에 쓴 데이터로 평가하면, 시험 문제를 미리 보여주고 시험을 치르는 것과 같습니다. 반드시 **일부를 떼어 놓고** 학습이 끝난 뒤 그것으로 평가합니다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(len(X_train), "장으로 학습 /", len(X_test), "장으로 평가")
```

## 첫 모델: k-최근접 이웃 (kNN)

가장 직관적인 모델입니다. 새 그림이 들어오면 **학습 데이터 중 가장 비슷한 k장**을 찾아 다수결로 답합니다.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)               # 학습
pred = model.predict(X_test)              # 예측
print("예측:", pred[:10])
print("정답:", y_test[:10])
print("정확도:", model.score(X_test, y_test))
```

scikit-learn의 모든 모델은 `fit`(학습) → `predict`(예측) → `score`(평가)라는 같은 사용법을 가집니다. 그래서 모델을 바꿔 끼우기가 쉽습니다.

## 두 번째 모델: 로지스틱 회귀

kNN은 데이터를 통째로 기억할 뿐 "학습되는 숫자"가 없습니다. 로지스틱 회귀는 다릅니다. 픽셀마다 **가중치**를 두고, `점수 = 픽셀값 × 가중치의 합`으로 각 숫자의 점수를 계산합니다. 학습이란 **정답의 점수가 높아지도록 가중치를 조정하는 과정**입니다. 이것은 층이 하나뿐인 신경망과 같으며, 2부에서 직접 구현합니다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))   # 입력 크기를 고르게 맞춘 뒤 학습
clf.fit(X_train, y_train)
print("정확도:", clf.score(X_test, y_test))

W = clf[-1].coef_                  # 학습된 가중치: [10개 숫자, 64개 픽셀]
fig, axes = plt.subplots(1, 10, figsize=(12, 1.8))
for d, ax in enumerate(axes):
    ax.imshow(W[d].reshape(8, 8), cmap="bwr"); ax.set_title(d); ax.axis("off")
plt.show()   # 빨강: 이 픽셀이 칠해져 있으면 그 숫자일 가능성 ↑, 파랑: ↓
```

가중치를 그려 보면 모델이 "0은 가운데가 비어 있다"를 스스로 찾아낸 것이 보입니다.

## 정확도만 보면 안 되는 이유

정확도는 "맞힌 비율" 하나로 뭉뚱그린 숫자입니다. **무엇을 무엇으로 틀리는지**는 혼동 행렬이 알려줍니다.

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

ConfusionMatrixDisplay.from_predictions(y_test, clf.predict(X_test))
plt.show()
print(classification_report(y_test, clf.predict(X_test)))
```

- **정밀도(precision):** 모델이 "8"이라고 한 것 중 진짜 8의 비율
- **재현율(recall):** 진짜 8 중에서 모델이 찾아낸 비율

암 진단 모델이라면? 환자 100명 중 1명만 암일 때 "전부 정상"이라고 답해도 정확도는 99%입니다. 그러나 재현율은 0%입니다. **문제에 맞는 지표를 고르는 것**은 모델을 고르는 것만큼 중요합니다.

## 머신러닝 문제의 종류

| 종류 | 정답의 형태 | 예 |
|---|---|---|
| 분류 (classification) | 정해진 범주 중 하나 | 스팸 여부, 숫자 인식, **LLM의 다음 토큰 예측** |
| 회귀 (regression) | 연속된 숫자 | 집값, 기온 예측 |
| 비지도 학습 (unsupervised) | 정답 없음 | 고객 군집화, 차원 축소 |
| 자기지도 학습 (self-supervised) | 데이터 자체에서 정답을 만듦 | LLM 사전학습 (다음 단어가 곧 정답) |

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 비지도 학습 맛보기: 정답 없이 64차원을 2차원으로 줄여 그려 보기
X2 = PCA(n_components=2).fit_transform(X)
groups = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(X)   # 정답을 보지 않고 10개 무리로 나눔
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(X2[:, 0], X2[:, 1], c=y, cmap="tab10", s=8); axes[0].set_title("colored by true digit")
axes[1].scatter(X2[:, 0], X2[:, 1], c=groups, cmap="tab10", s=8); axes[1].set_title("clusters found by KMeans (no labels)")
plt.show()
```

## 핵심 정리

- 머신러닝은 **규칙 대신 예시**를 주고 컴퓨터가 규칙(파라미터)을 찾게 하는 것입니다.
- 흐름은 언제나 `데이터 → 분할 → 학습(fit) → 처음 보는 데이터로 평가`입니다.
- 학습에 쓴 데이터로 평가하면 안 됩니다. 테스트 데이터는 마지막까지 떼어 둡니다.
- 정확도 하나만 믿지 말고 혼동 행렬, 정밀도, 재현율로 **어떻게 틀리는지** 보세요.
- 로지스틱 회귀의 "가중치 × 입력의 합"은 신경망의 가장 작은 단위입니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. kNN에서 k=1일 때 <b>학습 데이터</b>에 대한 정확도는 얼마일까요? 그것이 좋은 모델이라는 뜻일까요?</summary>

100%입니다. 자기 자신이 가장 가까운 이웃이기 때문입니다. 하지만 이것은 외운 것일 뿐이며, 모델의 실력은 테스트 데이터로만 알 수 있습니다.

</details>

<details><summary>Q2. 사기 거래가 0.1%인 데이터에서 정확도 99.9%인 모델은 좋은 모델일까요?</summary>

알 수 없습니다. "전부 정상"이라고만 답해도 99.9%가 나옵니다. 이런 불균형 데이터에서는 사기 거래에 대한 재현율과 정밀도를 봐야 합니다.

</details>

<details><summary>Q3. LLM의 다음 토큰 예측은 위 표의 어느 종류에 해당하나요?</summary>

모델이 푸는 문제의 형태는 **분류**(수만 개 토큰 중 하나 고르기)이고, 정답을 사람이 붙이지 않고 텍스트 자체에서 얻는다는 점에서 **자기지도 학습**입니다.

</details>

## 직접 고쳐보기

1. `n_neighbors`를 1, 5, 15, 50으로 바꿔 테스트 정확도를 비교해 보세요.
2. `test_size=0.9`로 바꿔(학습 데이터를 10%만 사용) 두 모델을 다시 학습시켜 보세요. 데이터 양이 성능에 미치는 영향이 보입니다.
3. `random_state`를 바꾸면 정확도가 조금씩 달라집니다. 왜일까요? 그렇다면 "모델 A가 0.5% 더 좋다"는 결론은 언제 믿을 수 있을까요?
4. 혼동 행렬에서 가장 많이 헷갈리는 숫자 쌍을 찾고, 그 틀린 이미지들을 `plt.imshow`로 직접 그려 보세요. 사람이 봐도 헷갈리나요?
5. (도전) `from sklearn.svm import SVC`나 `from sklearn.ensemble import RandomForestClassifier`로 모델만 바꿔 보세요. 나머지 코드는 그대로입니다.
