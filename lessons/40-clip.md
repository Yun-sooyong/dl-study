# CLIP — 이미지와 글을 같은 공간에 놓기

> ⏱ 60분 · T4 GPU 권장 (CPU로도 실행 가능)

**목표:** VLM의 "눈"으로 가장 널리 쓰이는 CLIP을 다뤄 봅니다. 학습 없이 이미지를 분류하고(제로샷), 글로 사진을 검색하고, CLIP을 학습시킨 **대조 손실**을 직접 구현합니다.

## CLIP의 아이디어

인터넷에는 (이미지, 그 설명글) 쌍이 수억 개 있습니다. CLIP은 **이미지 인코더**와 **텍스트 인코더**를 하나씩 두고, 짝이 맞는 이미지와 글은 **벡터가 가깝게**, 아닌 것은 멀게 되도록 학습했습니다.

```
"풀밭을 달리는 개"  → [텍스트 인코더] → ●  ← 가깝게 →  ●  ← [이미지 인코더] ←  🐕 사진
"도시의 야경"       → [텍스트 인코더] → ○        멀게
```

그 결과 이미지와 글이 **같은 벡터 공간**에 놓입니다. [임베딩과 RAG](#35-embeddings-rag)에서 문장끼리 했던 일을 이미지와 문장 사이에서 하는 것입니다.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
name = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(name).to(device).eval()
processor = CLIPProcessor.from_pretrained(name)          # 이미지 전처리 + 토크나이저

ds = load_dataset("jxie/flickr8k", split="test[:200]")   # 사진 200장 + 영어 설명
print(ds[0]["caption_0"])
ds[0]["image"]
```

```python
@torch.no_grad()
def embed_images(images):
    inputs = processor(images=[im.convert("RGB") for im in images], return_tensors="pt").to(device)
    out = clip.vision_model(pixel_values=inputs.pixel_values).pooler_output    # 이미지 인코더(ViT)의 요약 벡터
    return F.normalize(clip.visual_projection(out), dim=-1)                    # 공용 공간(512차원)으로 투영

@torch.no_grad()
def embed_texts(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    out = clip.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).pooler_output
    return F.normalize(clip.text_projection(out), dim=-1)

print(embed_images([ds[0]["image"]]).shape, embed_texts(["a dog"]).shape)     # 둘 다 [1, 512]
```

**코드 읽기**

- `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` — 이미지 인코더(ViT-B/32)와 텍스트 인코더를 **둘 다** 담은 모델. "B/32"는 Base 크기, 32×32 픽셀 패치라는 뜻. 가장 작고 빠른 CLIP이라 실습에 적합하며, [미니 VLM](#41-mini-vlm)에서 이 모델의 이미지 인코더 부분만 다시 씁니다.
- `CLIPProcessor` — 이미지 전처리기(224로 리사이즈, CLIP 전용 평균·표준편차로 정규화)와 토크나이저를 묶은 것. `processor(images=...)`와 `processor(text=...)`로 각각 씁니다. [전이학습](#25-transfer-learning)의 `transforms.Normalize`를 모델 카드에 맞게 미리 설정해 둔 셈입니다.
- `load_dataset("jxie/flickr8k", split="test[:200]")` — Hugging Face `datasets`로 사진+캡션 데이터를 받습니다. `split="test[:200]"`처럼 문자열로 일부만 지정할 수 있습니다. 200장이면 검색 실험에 충분하고 임베딩도 몇 초면 끝납니다.
- `clip.vision_model(pixel_values=...).pooler_output` — 이미지 인코더의 요약 벡터(768차원, ViT의 CLS 토큰). `clip.visual_projection`이 이것을 텍스트와 공유하는 512차원 공간으로 투영합니다. 텍스트 쪽도 `text_model` → `text_projection`으로 대칭입니다. 두 인코더의 출력 차원이 다르므로(768 vs 512) 투영층이 **같은 공간으로 맞추는** 역할을 합니다.
- 왜 `get_image_features` 같은 한 줄짜리 편의 함수 대신 두 단계로 부르나: 안에서 어떤 부품이 어떤 순서로 쓰이는지 보이기 위해서입니다. 편의 함수는 정확히 이 두 줄을 감싼 것입니다.
- `im.convert("RGB")` — Flickr 사진 중 일부는 흑백(1채널)이라 그대로 넣으면 채널 수가 안 맞아 에러가 납니다. 이미지 데이터를 다룰 때 습관처럼 붙이는 변환입니다.
- `F.normalize(..., dim=-1)` — 길이 1로 맞춰 내적 = 코사인 유사도. [임베딩과 RAG](#35-embeddings-rag)와 같은 이유입니다.
## 제로샷 분류: 학습 없이 분류기 만들기

[전이학습](#25-transfer-learning)에서는 새 클래스를 분류하려면 머리를 새로 학습해야 했습니다. CLIP은 **클래스 이름을 글로 써 주기만** 하면 됩니다. 이미지 벡터와 가장 가까운 문장이 답입니다.

```python
labels = ["dog", "person", "bicycle", "car", "cat", "horse"]
text_vecs = embed_texts([f"a photo of a {l}" for l in labels])

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, row in zip(axes, ds.select(range(5))):
    sims = embed_images([row["image"]]) @ text_vecs.T                 # 코사인 유사도 [1, 6]
    probs = (100 * sims).softmax(dim=-1)[0]                           # 100은 CLIP이 학습한 온도(temperature) 값
    best = probs.argmax().item()
    ax.imshow(row["image"]); ax.axis("off"); ax.set_title(f"{labels[best]} ({probs[best].item():.0%})")
plt.show()
```

`labels`를 아무 단어로나 바꿔도 **재학습 없이** 즉시 새 분류기가 됩니다. 이것을 제로샷(zero-shot)이라고 합니다.

**코드 읽기**

- `[f"a photo of a {l}" for l in labels]` — 클래스 이름을 문장 틀에 넣습니다. CLIP은 "단어"가 아니라 "설명 문장"과 이미지를 짝지어 학습했으므로, 문장 형태가 학습 분포에 가깝습니다(스스로 점검 Q1).
- `text_vecs`를 루프 **밖에서** 한 번만 계산 — 클래스 문장은 이미지가 바뀌어도 같으므로 미리 계산합니다. 분류할 이미지가 만 장이어도 텍스트 인코더는 6번만 돌면 됩니다.
- `embed_images([...]) @ text_vecs.T` — 이미지 1장과 클래스 6개의 유사도 `[1, 6]`. 분류기의 "로짓"에 해당합니다.
- `(100 * sims).softmax(dim=-1)` — 코사인 유사도는 −1~1 범위라 그대로 softmax하면 확률이 거의 균등합니다. CLIP은 학습 때 유사도에 100 근처의 값(`logit_scale`)을 곱했으므로, 같은 값을 곱해야 학습 때와 같은 확신 수준의 확률이 나옵니다. [LLM 다루기](#33-llm-inference)의 temperature와 같은 역할입니다(100 = 1/0.01).
- 이 코드에는 `fit`도 `backward`도 없습니다. **분류기를 학습시키지 않았는데 분류가 됩니다.** [전이학습](#25-transfer-learning)에서 머리를 새로 학습해야 했던 것과 대조해 보세요.

## 글로 사진 찾기

```python
image_vecs = torch.cat([embed_images(ds[i:i + 50]["image"]) for i in range(0, len(ds), 50)])   # 200장을 미리 임베딩

def search(query, k=4):
    scores = (embed_texts([query]) @ image_vecs.T)[0]
    top = scores.topk(k)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4), squeeze=False); fig.suptitle(query)
    for ax, s, i in zip(axes[0], top.values.tolist(), top.indices.tolist()):
        ax.imshow(ds[i]["image"]); ax.axis("off"); ax.set_title(f"{s:.3f}")
    plt.show()
    return top.indices.tolist()

search("a dog jumping to catch something")
search("children playing in water")
search("a person riding a bike")
```

사진에 태그를 단 적이 없는데도 문장으로 검색이 됩니다. 스마트폰 사진 앱의 검색 기능이 이런 방식입니다.

## CLIP은 어떻게 학습되었나: 대조 손실

배치에 (이미지, 글) 쌍이 N개 있으면, N×N 유사도 표를 만듭니다. **대각선(진짜 짝)의 점수는 높게, 나머지는 낮게** 만드는 것이 목표입니다. 각 행을 "N개 글 중 내 짝 고르기"라는 **분류 문제**로 보면, 손실은 우리가 계속 써 온 cross entropy입니다.

```python
def contrastive_loss(img_vecs, txt_vecs, temperature=100.0):
    logits = temperature * img_vecs @ txt_vecs.T           # [N, N] 유사도 표
    target = torch.arange(len(logits), device=logits.device)   # i번째 이미지의 정답은 i번째 글
    loss_i = F.cross_entropy(logits, target)               # 이미지 → 맞는 글 고르기
    loss_t = F.cross_entropy(logits.T, target)             # 글 → 맞는 이미지 고르기
    return (loss_i + loss_t) / 2, logits

batch = ds.select(range(8))
img_v, txt_v = embed_images(batch["image"]), embed_texts(list(batch["caption_0"]))
loss, logits = contrastive_loss(img_v, txt_v)
print("올바른 짝일 때 loss:", loss.item())
print("글 순서를 섞었을 때 loss:", contrastive_loss(img_v, txt_v.roll(1, 0))[0].item())

plt.imshow((logits / 100).cpu(), cmap="viridis"); plt.colorbar(); plt.xlabel("caption"); plt.ylabel("image")
plt.title("similarity matrix (diagonal = true pairs)"); plt.show()
```

**코드 읽기**

- `logits = temperature * img_vecs @ txt_vecs.T` — 이미지 N개 × 글 N개의 유사도 표 `[N, N]`. 행 i, 열 j는 "i번째 이미지와 j번째 글". 짝이 맞는 것은 대각선(i = j)입니다.
- `target = torch.arange(N)` — i번째 행의 정답은 i번째 열. 즉 정답 라벨이 `[0, 1, 2, ..., N-1]`입니다. 라벨을 사람이 붙일 필요가 없다는 것이 이 한 줄에 담겨 있습니다.
- `F.cross_entropy(logits, target)` — 각 이미지에 대해 "N개 글 중 내 짝 고르기" N-way 분류. [첫 신경망](#21-mlp-mnist)의 손실 함수를 그대로 씁니다. 클래스가 숫자 10개가 아니라 **같은 배치의 다른 글들**일 뿐입니다.
- `F.cross_entropy(logits.T, target)` — 표를 뒤집어 "각 글에 대해 N개 이미지 중 내 짝 고르기". 양방향으로 계산해 평균합니다. 이미지→글, 글→이미지 검색이 모두 되는 이유입니다.
- `txt_v.roll(1, 0)` — 글 벡터의 순서를 한 칸 밀어 짝을 일부러 어긋나게 만든 대조군. 손실이 0.2에서 14로 뛰는 것으로 "손실이 짝 맞춤을 재고 있다"를 확인합니다. 손실 함수를 새로 짜면 이렇게 **정답일 때와 틀렸을 때의 값**을 비교해 검증하세요.
- 이 코드는 `backward()`가 없으므로 학습이 아니라 **측정**입니다. 여기에 옵티마이저와 루프를 붙이면 CLIP 학습 코드가 됩니다. 실제로 그렇게 학습하려면 배치가 수만 개여야 하고(스스로 점검 Q2) GPU 수백 개가 필요하므로 실습에서는 손실만 확인합니다.

대각선이 밝게 나옵니다. CLIP은 이 손실로 4억 쌍을 학습했습니다. 라벨을 사람이 붙인 것이 아니라 **인터넷에 원래 있던 짝**을 이용했다는 점에서 LLM 사전학습과 같은 자기지도 학습입니다.

## CLIP의 한계와 VLM으로 가는 길

```python
tests = ["a photo of two dogs", "a photo of three dogs", "a dog on the left of a person", "a person on the left of a dog"]
row = ds[search("two dogs", k=1)[0]]
sims = (embed_images([row["image"]]) @ embed_texts(tests).T)[0]
for t, s in zip(tests, sims.tolist()):
    print(f"{s:.3f}  {t}")
```

네 문장의 점수가 거의 같습니다. 개가 두 마리인지 세 마리인지, 누가 왼쪽인지 구분하지 못하고, 사진에 사람이 없어도 "dog … person" 문장의 점수가 높게 나오기도 합니다. "개"라는 단어가 들어 있는 것만으로 점수 대부분이 결정되기 때문입니다.

CLIP은 이미지 전체를 **벡터 하나**로 요약하기 때문에 "무엇이 있는가"는 잘 알지만 개수, 위치 관계, 글자 읽기에는 약하고, 무엇보다 **문장을 생성하지 못합니다.** 그래서 다음 단계가 나옵니다: CLIP의 이미지 인코더가 뽑은 **패치별 벡터들**을 LLM에 직접 넣어 주는 것. 그것이 [미니 VLM](#41-mini-vlm)입니다.

## 핵심 정리

- CLIP = 이미지 인코더 + 텍스트 인코더. 짝이 맞는 이미지와 글이 가까워지도록 **대조 학습**되었습니다.
- 이미지와 글이 같은 공간에 있으므로 제로샷 분류, 글↔이미지 검색이 **추가 학습 없이** 됩니다.
- 대조 손실은 "배치 안에서 내 짝 고르기" cross entropy입니다.
- 벡터 하나로 요약하는 방식이라 세밀한 이해(개수, 위치)와 생성은 못합니다.
- CLIP(또는 후속작 SigLIP)의 이미지 인코더는 대부분의 VLM에서 "눈"으로 재사용됩니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 제로샷 분류에서 클래스 이름을 그냥 "dog"이 아니라 "a photo of a dog"으로 넣는 이유는?</summary>

CLIP이 학습한 글은 단어 하나가 아니라 "~의 사진" 같은 설명 문장이었습니다. 학습 때와 비슷한 형태로 넣어 주면 성능이 올라갑니다. 일종의 프롬프트 엔지니어링입니다.

</details>

<details><summary>Q2. 대조 학습에서 배치 크기가 클수록 유리한 이유는?</summary>

배치 안의 다른 샘플들이 "오답 보기" 역할을 합니다. 배치가 크면 오답 보기가 많아져 문제가 어려워지고, 더 세밀한 구분을 배웁니다. CLIP은 배치 크기 32,768로 학습했습니다.

</details>

<details><summary>Q3. CLIP의 이미지 인코더를 VLM에 쓸 때, 왜 요약 벡터 하나가 아니라 패치별 벡터들을 쓰나요?</summary>

벡터 하나에는 "어디에 무엇이 있는지" 같은 공간 정보가 거의 남지 않습니다. 패치별 벡터(예: 49개, 576개)를 모두 LLM에 넘기면 LLM이 어텐션으로 필요한 부분을 골라 볼 수 있습니다.

</details>

## 직접 고쳐보기

1. `labels`를 바꿔 나만의 분류기를 만들어 보세요(예: `["indoor", "outdoor"]`, `["happy", "sad"]`, `["daytime", "night"]`).
2. `"a photo of a {l}"`을 그냥 `"{l}"`로 바꾸면 결과가 달라지나요?
3. 한국어 질의(`"눈밭에서 뛰는 개"`)로 검색해 보세요. 잘 안 됩니다. 왜일까요? (힌트: 이 CLIP의 학습 데이터 언어)
4. 직접 찍은 사진 여러 장을 업로드해 "내 사진 검색기"를 만들어 보세요.
5. (도전) 200장 전체에 대해 "이미지 → 자기 캡션 찾기" 정확도(top-1)를 계산해 보세요. `logits.argmax(1) == arange` 의 평균입니다.
