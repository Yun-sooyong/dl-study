# 나만의 미니 GPT — LLM을 밑바닥부터

> ⏱ 90분 · T4 GPU 권장 (학습 3~5분, CPU는 수십 분)

**목표:** ChatGPT와 **같은 구조**(디코더 트랜스포머)의 언어모델을 약 100줄로 직접 만들고 학습시킵니다. 크기만 수만 배 작을 뿐, 원리는 동일합니다.

## LLM이 하는 일은 단 하나: 다음 토큰 맞히기

"안녕하세" → "요". 텍스트를 토큰 번호의 나열로 바꾸고, 앞부분을 보고 바로 다음 토큰을 분류하는 문제입니다. [첫 신경망](#21-mlp-mnist) 레슨의 숫자 분류와 같은 `CrossEntropyLoss`를 씁니다. 클래스가 10개가 아니라 어휘 크기만큼일 뿐입니다.

## 데이터와 토크나이저

가장 단순한 **글자 단위** 토크나이저를 씁니다. (진짜 LLM은 [토크나이저](#30-tokenizer) 레슨에서 만든 BPE를 쓰지만 역할은 같습니다: 글 ↔ 정수열. 여기서는 모델 자체에 집중하려고 가장 단순한 방식을 씁니다)

```python
import torch, urllib.request
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")
print(len(text), "글자\n", text[:200])

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(chars[i] for i in ids)
print(vocab_size, encode("Hello"), decode(encode("Hello")))

data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
```

## 배치 만들기: 입력과 정답은 한 칸 차이

```python
block_size = 128   # 한 번에 보는 문맥 길이 (context length)
batch_size = 64

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # x를 한 칸 민 것이 정답
    return x.to(device), y.to(device)

x, y = get_batch("train")
print(repr(decode(x[0, :20].tolist())), "→", repr(decode(y[0, :20].tolist())))
```

## 핵심: 셀프 어텐션

각 토큰이 **앞의 토큰들 중 누구를 얼마나 참고할지** 스스로 정하는 장치입니다.

- 각 토큰이 Query(내가 찾는 것), Key(내가 가진 것), Value(내가 줄 정보) 세 벡터를 만듭니다.
- `Q · K`가 크면 "관련 있다" → softmax로 가중치를 만들어 Value들을 가중평균합니다.
- **미래 토큰은 보면 안 되므로**(정답 유출) 마스크로 가립니다. 이것이 GPT가 "causal" LM인 이유입니다.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)   # Q, K, V를 한 번에 계산
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                           # 배치, 토큰 수, 임베딩 차원
        q, k, v = self.qkv(x).split(C, dim=2)
        # 헤드 여러 개로 쪼갬: [B,T,C] → [B,헤드,T,C/헤드]
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)          # [B,헤드,T,T] 토큰 간 관련도
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # 미래 가리기
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)              # 헤드 다시 합침
        return self.proj(out)
```

## 트랜스포머 블록과 GPT 전체

블록 = 어텐션(토큰끼리 정보 교환) + MLP(각 토큰이 혼자 생각). `x + ...` 형태의 **잔차 연결**이 깊은 모델의 학습을 가능하게 합니다.

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, n_embd=128, n_head=4, n_layer=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # 토큰 번호 → 벡터
        self.pos_emb = nn.Embedding(block_size, n_embd)   # 위치 번호 → 벡터 (어텐션은 순서를 모르므로)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)         # 벡터 → 다음 토큰 점수

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))     # [B,T,vocab]
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])                   # 문맥 길이만큼만 봄
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)   # 마지막 위치의 예측만 사용
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)   # 확률대로 하나 뽑아 이어붙임
        return idx

model = GPT().to(device)
print("파라미터 수:", sum(p.numel() for p in model.parameters()))
```

## 학습 전: 아무 말이나 뱉습니다

```python
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 200)[0].tolist()))
print("초기 loss 예상값:", torch.log(torch.tensor(float(vocab_size))).item())   # 완전히 찍을 때의 loss = ln(어휘 크기)
```

## 학습

```python
@torch.no_grad()
def estimate_loss(iters=20):
    model.eval()
    out = {s: sum(model(*get_batch(s))[1].item() for _ in range(iters)) / iters for s in ("train", "val")}
    model.train()
    return out

max_steps = 2000
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for step in range(max_steps + 1):
    if step % 250 == 0:
        l = estimate_loss()
        print(f"step {step:5d}  train {l['train']:.3f}  val {l['val']:.3f}")
    _, loss = model(*get_batch("train"))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 학습 후: 셰익스피어 "풍"의 글이 나옵니다

```python
print(decode(model.generate(start, 500, temperature=0.8)[0].tolist()))
```

단어 철자, 대사 형식(`이름:` 뒤에 줄바꿈)을 모델이 **스스로** 익혔습니다. 누구도 규칙을 알려주지 않았고, "다음 글자 맞히기"만 시켰을 뿐입니다. 진짜 LLM은 이것을 수천억 파라미터·수조 토큰 규모로 키운 것입니다.

## 핵심 정리

- LLM의 학습 과제는 **다음 토큰 예측** 하나뿐이고, 손실은 분류와 같은 cross entropy입니다.
- 입력을 한 칸 민 것이 정답입니다. 한 문장에서 위치마다 학습 신호가 나옵니다.
- 셀프 어텐션: 각 토큰이 Q·K로 관련도를 구해 앞 토큰들의 V를 가중평균합니다. causal mask로 미래를 가립니다.
- 트랜스포머 블록 = 어텐션 + MLP, 각각 잔차 연결과 LayerNorm.
- 생성은 '하나 뽑아 붙이고 다시 예측'의 반복이며, temperature로 무작위성을 조절합니다.

## 스스로 점검

답을 머릿속으로 먼저 말해 본 뒤 펼쳐 보세요.

<details><summary>Q1. 학습 전 loss가 약 4.17(= ln 65)로 나오는 이유는?</summary>

무작위 모델은 65개 글자에 거의 균등한 확률(1/65)을 줍니다. cross entropy는 −ln(정답 확률)이므로 −ln(1/65) = ln 65 ≈ 4.17입니다. 초기 loss가 이 값과 크게 다르면 버그를 의심하세요.

</details>

<details><summary>Q2. causal mask를 빼면 loss는 매우 낮아지는데 생성이 엉망인 이유는?</summary>

학습 때 각 위치가 정답인 '다음 글자'를 직접 볼 수 있게 되어 베끼기만 배우기 때문입니다. 생성할 때는 미래 토큰이 존재하지 않으므로 배운 것이 쓸모없어집니다.

</details>

<details><summary>Q3. 위치 임베딩이 필요한 이유는?</summary>

어텐션은 토큰들의 집합에 대한 가중평균이라 순서 정보가 없습니다. 위치 임베딩이 없으면 'dog bites man'과 'man bites dog'을 구분할 수 없습니다.

</details>

<details><summary>Q4. train loss는 계속 내려가는데 val loss가 다시 올라간다면?</summary>

과적합입니다. 모델이 학습 텍스트를 외우기 시작한 것입니다. 데이터를 늘리거나, 모델을 줄이거나, dropout을 넣거나, val loss가 가장 낮았던 시점에서 멈춥니다.

</details>

## 직접 고쳐보기

1. `temperature`를 `0.3`과 `1.5`로 바꿔 생성해 보세요. 무엇이 달라지나요?
2. `GPT(n_embd=256, n_head=8, n_layer=6)`으로 키워서 학습시키세요. val loss가 더 내려가나요? train과 val loss의 **격차**는 어떻게 되나요? (과적합)
3. 마스크를 적용하는 `masked_fill` 줄을 주석 처리하고 학습시켜 보세요. loss는 엄청 낮아지는데 생성 결과는 엉망입니다. 왜일까요?
4. `pos_emb`를 빼고(`x = self.tok_emb(idx)`) 학습시켜 보세요.
5. **내 데이터로 학습하기:** `text`를 원하는 한국어 텍스트(소설, 가사, 내 블로그 글 — 최소 수십만 글자 권장)로 바꿔보세요. 코드는 한 줄도 고칠 필요가 없습니다.
6. (도전) 어텐션 계산 네 줄을 `F.scaled_dot_product_attention(q, k, v, is_causal=True)` 한 줄로 바꿔보세요. 실무에서는 이 최적화된 구현(FlashAttention)을 씁니다.
