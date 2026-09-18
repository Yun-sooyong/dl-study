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

**코드 읽기**

- 왜 셰익스피어인가: 100만 글자짜리 영어 텍스트 파일 하나로, 글자 종류가 65개뿐이라 어휘가 작고, 대사 형식이 뚜렷해서 모델이 배운 것을 눈으로 확인하기 좋습니다. Karpathy의 nanoGPT 강의에서 쓰는 바로 그 파일입니다. 한국어 텍스트로 바꿔도 코드는 그대로입니다(실습 5번).
- `urllib.request.urlopen(url).read().decode("utf-8")` — 표준 라이브러리로 파일 내려받기. 별도 패키지가 필요 없습니다.
- `sorted(set(text))` — 텍스트에 나오는 글자의 집합을 정렬한 것이 어휘입니다. `set`으로 중복을 없애고 `sorted`로 순서를 고정해야 실행할 때마다 같은 번호가 붙습니다.
- `stoi`(string to index) / `encode` / `decode` — 글자 ↔ 번호 변환. 이 세 줄이 [토크나이저](#30-tokenizer) 레슨의 BPE를 가장 단순하게 대체한 것입니다. `decode(encode(s)) == s`인지 확인하는 것도 같습니다.
- `torch.tensor(encode(text))` — 100만 글자 전체를 정수 텐서 하나로. `dtype`은 자동으로 `int64`(long)가 되는데, 임베딩 층은 정수 인덱스를 받으므로 이것이 맞습니다.
- `data[:n], data[n:]` — 앞 90%가 학습, 뒤 10%가 검증. 무작위로 섞지 않고 **앞뒤로** 자르는 이유: 텍스트는 이어져 있어서 무작위로 섞으면 검증 문장의 앞뒤가 학습 데이터에 들어가 "본 적 있는" 것이 됩니다(누출).

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

**코드 읽기**

- `block_size = 128` — 모델이 한 번에 보는 글자 수(컨텍스트 길이). 이보다 앞의 글자는 모델이 볼 수 없습니다. 크게 잡을수록 긴 맥락을 배우지만 어텐션 계산이 길이의 제곱으로 늘어납니다. 실제 LLM의 "컨텍스트 128K"가 이 값입니다.
- `torch.randint(len(d) - block_size - 1, (batch_size,))` — 텍스트에서 무작위 시작 위치 64개를 뽑습니다. `DataLoader`를 쓰지 않는 이유: 텍스트 하나에서 무작위 구간을 잘라 내는 것이라 이 함수 하나가 더 단순합니다. `- block_size - 1`은 끝에서 넘치지 않게 하는 여유입니다.
- `x = d[i:i+block_size]`, `y = d[i+1:i+block_size+1]` — **정답은 입력을 한 칸 민 것**입니다. 위치 t의 입력 `x[t]`에 대한 정답이 `y[t] = x[t+1]`이므로, 문장 하나에서 128개의 "다음 글자 맞히기" 문제가 나옵니다. 따로 라벨을 만들 필요가 없는 자기지도 학습의 핵심입니다.
- `torch.stack([...])` — 길이 128짜리 텐서 64개를 쌓아 `[64, 128]` 하나로. 길이가 모두 같으므로 패딩이 필요 없습니다.
- 마지막 출력에서 확인할 것: x의 두 번째 글자부터가 y의 첫 글자와 같아야 합니다.

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

**코드 읽기** — 이 클래스가 GPT의 심장입니다. 줄마다 shape을 따라가 보세요.

- `nn.Linear(n_embd, 3 * n_embd)` — Q, K, V를 만드는 세 개의 `Linear`를 하나로 합친 것. 출력 `[B, T, 3C]`를 `split(C, dim=2)`로 셋으로 자릅니다. 세 번 따로 계산하는 것보다 행렬곱 한 번이 빠르기 때문에 실제 구현들도 이렇게 합니다.
- 왜 Q, K, V가 필요한가: 각 토큰이 "나는 무엇을 찾는가(Q)", "나는 무엇을 갖고 있는가(K)", "나를 참고하면 무엇을 얻는가(V)"를 각각 다른 `Linear`로 만듭니다. 세 역할을 분리해야 "관련도를 계산하는 정보"와 "실제로 전달할 정보"를 따로 배울 수 있습니다. 하나의 벡터로 다 하면 표현력이 떨어집니다.
- `register_buffer("mask", torch.tril(...))` — 하삼각(대각선 아래만 1) 행렬. `register_buffer`는 "파라미터는 아니지만 모델과 함께 저장·이동해야 하는 텐서"를 등록합니다. 그냥 `self.mask = ...`로 두면 `.to(device)`가 옮겨 주지 않아 GPU에서 에러가 납니다.
- `.view(B, T, n_head, C // n_head).transpose(1, 2)` — 임베딩 C차원을 헤드 수만큼 쪼개서 `[B, 헤드, T, 헤드당 차원]`으로 배치합니다. 왜 여러 헤드인가: 헤드마다 다른 관계(어떤 헤드는 바로 앞 글자, 어떤 헤드는 문장 첫 단어)를 볼 수 있어서, 하나의 큰 어텐션보다 표현력이 좋습니다. 헤드마다 파라미터가 늘지는 않습니다. 같은 C를 나눠 쓸 뿐입니다.
- `q @ k.transpose(-2, -1)` — `[B, h, T, d] @ [B, h, d, T] = [B, h, T, T]`. 모든 토큰 쌍의 관련도 표입니다. `/ sqrt(d)`로 나누는 이유: d가 크면 내적 값이 커져 softmax가 한 곳에 몰리고(거의 one-hot) 기울기가 사라집니다. 논문의 "scaled" dot-product attention이 이것입니다.
- `masked_fill(mask == 0, -inf)` — 미래 위치의 관련도를 −∞로. softmax를 지나면 정확히 0이 되어 미래 토큰의 V가 전혀 섞이지 않습니다. 0으로 채우면 안 되는 이유는 softmax(0)이 0이 아니기 때문입니다.
- `F.softmax(att, dim=-1)` — 각 토큰(행)마다 앞 토큰들에 대한 가중치의 합이 1이 되게. 이 가중치가 "누구를 얼마나 참고할지"입니다.
- `att @ v` — 가중치로 V들을 가중평균. 결과 `[B, h, T, d]`를 `transpose`·`reshape`으로 다시 `[B, T, C]`로 합칩니다. `reshape` 앞에 `transpose`가 있어 메모리가 연속이 아니므로 `view` 대신 `reshape`을 씁니다.
- `self.proj` — 헤드들을 합친 결과를 한 번 더 섞는 `Linear`. 헤드끼리 정보를 교환하는 유일한 지점입니다.

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

**코드 읽기 — Block**

- `x = x + self.attn(self.ln1(x))` — **잔차 연결(residual)**. 어텐션의 출력을 입력에 **더합니다.** 왜: 층을 수십 개 쌓으면 기울기가 앞쪽까지 전달되지 않아 학습이 안 되는데, 덧셈 경로가 있으면 기울기가 그 경로로 곧장 흘러갑니다. 또 각 층은 "입력을 바꾸는 법"이 아니라 "입력에 무엇을 더할지"만 배우면 되어 학습이 쉽습니다. ResNet이 CNN에 도입한 것을 트랜스포머가 그대로 가져왔습니다.
- `nn.LayerNorm(n_embd)` — 토큰 벡터 하나의 값들을 평균 0, 분산 1로 맞춥니다. 깊은 모델에서 값의 크기가 층마다 커지거나 작아지는 것을 막습니다. 어텐션·MLP **앞에** 적용하는 것(pre-norm)이 GPT-2 이후 표준이며, 학습이 더 안정적입니다.
- `nn.Sequential(Linear(C, 4C), GELU(), Linear(4C, C))` — 블록 안의 MLP. 4배로 넓혔다가 다시 줄입니다. 어텐션이 "토큰끼리 정보 교환"이라면 MLP는 "각 토큰이 받은 정보를 혼자 처리"하는 단계입니다. 4배는 원 논문의 설정이며 대부분의 LLM이 그대로 씁니다. GELU는 ReLU의 부드러운 버전으로 트랜스포머에서 관례적으로 씁니다.

**코드 읽기 — GPT**

- `nn.Embedding(vocab_size, n_embd)` — 토큰 번호 → 벡터인 조회표. 내부는 `[65, 128]` 행렬이고, 번호 i의 행을 꺼내 줍니다. 학습되는 파라미터라서 모델이 "비슷하게 쓰이는 글자는 비슷한 벡터"가 되도록 스스로 정합니다.
- `nn.Embedding(block_size, n_embd)` — 위치 0~127 각각의 벡터. 어텐션은 토큰의 **집합**만 보고 순서를 모르므로, "이 토큰이 몇 번째인가"를 벡터에 더해 알려줍니다. 실습 4번에서 이것을 빼면 무슨 일이 생기는지 봅니다. 최근 LLM은 RoPE라는 다른 방식을 쓰지만 목적은 같습니다.
- `torch.arange(T, device=idx.device)` — `[0, 1, ..., T-1]`. `device`를 입력과 맞추는 것을 잊으면 GPU에서 에러가 납니다.
- `nn.Sequential(*[Block(...) for _ in range(n_layer)])` — 같은 블록을 `n_layer`개 쌓습니다. `*`는 리스트를 풀어 인자로 넘기는 문법. GPT-2 small은 12개, 큰 LLM은 80개 이상을 쌓습니다.
- `self.head = nn.Linear(n_embd, vocab_size)` — 마지막 벡터를 어휘 크기의 로짓으로. [첫 신경망](#21-mlp-mnist)의 `Linear(hidden, 10)`과 역할이 같고, 클래스가 65개일 뿐입니다.
- `F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))` — 손실은 `[샘플, 클래스]`와 `[샘플]`을 받으므로 배치와 시퀀스를 하나로 폅니다. 64×128 = 8192개의 분류 문제를 한 번에 채점하는 셈입니다.
- `targets=None`일 때 loss를 안 구하는 이유 — 생성할 때는 정답이 없기 때문에 같은 `forward`를 두 용도로 씁니다.
- `generate`의 `@torch.no_grad()` — 생성은 학습이 아니므로 그래프를 기록하지 않습니다.
- `idx[:, -block_size:]` — 생성이 길어져 컨텍스트를 넘으면 마지막 128개만 봅니다. 위치 임베딩이 128개뿐이라 그 이상은 넣을 수 없습니다.
- `logits[:, -1] / temperature` — 마지막 위치의 로짓만 씁니다(다음 글자 예측). temperature로 나누면 T<1일 때 분포가 뾰족해지고(확실한 것만), T>1일 때 평평해집니다(모험).
- `torch.multinomial(probs, 1)` — 확률에 따라 하나를 **뽑습니다**. `argmax`(항상 1등)로 바꾸면 결과가 매번 같고 반복이 심해집니다. 샘플링 방법은 [LLM 다루기](#32-llm-inference)에서 자세히 다룹니다.
- 파라미터 약 80만 개. GPT-2 small(1.2억)의 1/150, 큰 LLM의 1/100만입니다. 그러나 클래스 구조는 실제 LLM 코드와 거의 일대일로 대응합니다. [LLM 파인튜닝](#33-llm-finetune)에서 `print(model)`을 하면 확인할 수 있습니다.

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

**코드 읽기**

- `torch.log(torch.tensor(float(vocab_size)))` — 학습 전 loss의 예상값 `ln 65 ≈ 4.17`. [학습 잘 시키는 법](#23-training-recipes)의 "초기 loss 확인"과 같습니다. 실제 값이 이 근처면 초기화와 손실 계산이 정상입니다.
- `estimate_loss(iters=20)` — 배치 하나의 loss는 우연히 쉽거나 어려운 구간이 걸려 요동칩니다. 20개 배치를 평균해서 안정된 값을 봅니다. train과 val을 **같은 방법으로** 재야 둘의 차이가 과적합의 크기가 됩니다.
- `model.eval()` → 측정 → `model.train()` — 지금 모델에는 dropout이 없어 차이가 없지만, 습관으로 넣어 둡니다.
- `torch.optim.AdamW(lr=3e-4)` — LLM 학습의 표준 옵티마이저. `3e-4`는 트랜스포머에서 "일단 이걸로 시작"하는 관례적 값입니다(Karpathy가 농담으로 "Adam의 최적 학습률"이라고 부른 숫자). 모델이 커질수록 더 작게 잡습니다.
- 에폭이라는 개념이 없는 이유 — `get_batch`가 무작위 위치를 뽑으므로 "데이터 한 바퀴"가 정의되지 않습니다. 대신 `max_steps`만큼 스텝을 돕니다. 실제 LLM 사전학습도 스텝(또는 토큰 수) 기준으로 돌립니다.
- 250스텝마다 평가 — 매 스텝 평가하면 학습보다 평가에 시간이 더 듭니다. 곡선의 모양을 볼 수 있을 정도의 간격이면 충분합니다.

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
