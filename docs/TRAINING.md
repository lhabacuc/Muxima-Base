# Muxima-Base — Preparação de Dados e Treino

Este guia explica como preparar o corpus, treinar o tokenizer, criar o dataset e treinar o Transformer com o CLI `muxima`.

## 1. Estrutura do Projeto

```
Muxima-Base/
├── config/
│   ├── model.json         # hiperparâmetros do Transformer
│   └── tokenizer.json     # vocab + merges BPE (gerado)
├── data/
│   ├── corpus.txt         # corpus bruto (uma frase por linha)
│   └── train.txt          # opcional: split treino
├── src/
│   ├── core/              # Tensor (Mtx::Matrix), Variable, Autograd
│   ├── io/                # Json, Config, Checkpoint
│   ├── nn/                # Linear(Variable), Embedding, LayerNorm, LMHead
│   ├── transformer/       # MultiHeadAttention causal, TransformerBlock, Transformer
│   ├── training/          # Dataset, CrossEntropy, SGD, ParameterList
│   └── tokenizer/         # Tokenizer BPE, BPETrainer
├── third_party/
│   ├── nlohmann/json.hpp  # https://github.com/nlohmann/json
│   └── matrix-hpp/matrix.hpp # https://github.com/gc1905/matrix-hpp
├── checkpoints/
└── muxima                 # binário CLI (make)
```

> **Libs externas**: `nlohmann/json` (header-only, vendorizado em `third_party/nlohmann/`) e `matrix-hpp` (header-only, `matmul`/`transpose` em `src/core/math.cpp`). `armadillo` (`https://arma.sourceforge.net`) pode substituir `matrix-hpp` com `-larmadillo` quando `libarmadillo-dev` disponível.

## 2. Requisitos

```bash
c++ --version # clang 12+ ou g++ 9+ com -std=c++11
make --version
```

O projeto compila com:

```makefile
CXX = c++
CXXFLAGS = -Wall -Wextra -Werror -std=c++11
INCLUDES = -I./third_party
```

## 3. Preparar o Corpus

O `Dataset` espera um ficheiro de texto com **uma frase por linha**. O `Tokenizer` faz `split_text` por espaços e `BPE` por bytes.

```bash
cat data/corpus.txt
# ola mundo
# ola muxima
# ola angola
# eu sou angolano
# muxima e inteligencia artificial
```

Para um treino real (português):

```bash
# 1) Junta o teu corpus (UTF-8, uma frase por linha)
cat data/raw/*.txt > data/corpus.txt

# 2) Normaliza (opcional): minúsculas, remover duplicados, filtrar linhas vazias
python3 -c "
import pathlib
lines = pathlib.Path('data/corpus.txt').read_text(encoding='utf-8').splitlines()
lines = [l.strip().lower() for l in lines if l.strip()]
lines = sorted(set(lines))
pathlib.Path('data/corpus.txt').write_text('\n'.join(lines), encoding='utf-8')
"
wc -l data/corpus.txt
```

**Tamanho recomendado**: comece com 10k-100k linhas para teste; para 16k-32k vocab, use 1M+ linhas.

**UTF-8**: a versão atual do `BPETrainer` é byte-level mas `Json::save` usa `error_handler_t::replace` para bytes 0x80-0xFF. Para produção com `ã/ç/é`, use BPE com contagem por frequência de palavras e tratamento UTF-8 correto (próximo passo do roadmap).

## 4. Treinar o Tokenizer (BPE)

```bash
# Compila apenas o CLI
make

# Treina BPE a partir do corpus
./muxima tokenizer train --corpus data/corpus.txt --output config/tokenizer.json --vocab-size 300
# ou via Make
make test_bpe  # usa data/corpus.txt -> config/tokenizer.json (vocab 300)

cat config/tokenizer.json | head -n 40
# {
#   "type": "bpe",
#   "vocab": { "<unk>":0, "<pad>":1, ... "ab":18 },
#   "merges": [["a","b"], ["a","m"], ...],
#   "special_tokens": {"unk":0, "pad":1, "bos":2, "eos":3}
# }

# Testa encode/decode
./muxima tokenizer encode --text "ola mundo"
# [tokenizer encode] "ola mundo" -> 261 32 293
# [decode] ola mundo

make test_tokenizer  # BPE trainer test passed
```

**Vocabulário**: `256` bytes + `4` especiais + `merges` = `vocab_size`. Para português real use `16000` ou `32000`:

```bash
./muxima tokenizer train --corpus data/corpus.txt --output config/tokenizer.json --vocab-size 16000
```

## 5. Configuração do Modelo

`config/model.json`:

```json
{
    "model": {
        "name": "Muxima-Base",
        "vocab_size": 32000,
        "max_length": 1024,
        "embedding_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "hidden_dim": 2048
    },
    "training": {
        "learning_rate": 0.0003,
        "batch_size": 8,
        "epochs": 3
    },
    "tokenizer": {
        "type": "bpe",
        "vocab_size": 32000
    }
}
```

O `vocab_size` do modelo **deve** coincidir com o do tokenizer. Após `tokenizer train --vocab-size 32000`, atualize `model.vocab_size`.

Teste:

```bash
make test_json  # JSON tests passed (carrega config/model.json)
```

## 6. Dataset — Next-Token Prediction

`src/training/dataset.*` converte `corpus.txt` → `Tokenizer` → `[token IDs]` → `input/target`:

```cpp
Dataset dataset("data/corpus.txt", "config/tokenizer.json", 8);
// tokens = [10,20,30,40,50] + seq_len 4
// input  [10,20,30,40]
// target [20,30,40,50]
```

Cada `TrainingSample` tem `input` e `target` com `sequence_length` tokens, onde `target[i] = input[i+1]`.

Teste:

```bash
make test_dataset  # Dataset tests passed
```

## 7. Treino

O `Trainer` faz:

```
get_sample() -> tokens -> Transformer -> logits -> CrossEntropy -> loss -> backward() -> SGD::step + zero_grad
```

Via CLI (stub atual, 2 steps por época):

```bash
./muxima train --config config/model.json --corpus data/corpus.txt --epochs 1
# [train] config=config/model.json corpus=data/corpus.txt epochs=1
# [train] model: vocab=32000 dim=512 layers=8 ... dataset tokens=21 samples=13
# [train] optimizer lr=0.0003 params=2
# [train] epoch 1/1
#   step 0 sample input[0]=261
#   step 1 sample input[0]=32
# [train] checkpoint saved to checkpoints/last.json

ls -lh checkpoints/
# last.json  { "model_name":"Muxima-Base", "step":1, "epoch":1, "loss":0.5, "parameters":[] }

# Treino completo (quando Trainer estiver ligado ao AutogradGraph):
# make t  # 23 tests incluindo Transformer causal, CrossEntropy backward, Autograd
```

**Checkpoint**: JSON protótipo com `model_name, step, epoch, loss, parameters[]` (`name, shape, data`). Para modelos reais (16M+ floats ~64MB) use formato binário `weights.bin` + `metadata.json`.

```bash
make test_io  # Checkpoint tests passed
```

## 8. Inferência e Chat

```bash
# Greedy decode de um prompt
./muxima infer --prompt "ola mundo" --config config/model.json
# [infer] tokens: 261 32 293
# [infer] decoded: ola mundo

# Chat interativo
./muxima chat
# Muxima chat (type 'exit' to quit)
# Tu: ola, quem es?
# Muxima: ola, quem es? (echo stub)

# Compactar checkpoint JSON -> binário (stub atual copia JSON)
./muxima compact --input checkpoints/last.json --output /tmp/compact.json
```

Futuro com `temperature/top-p/top-k`:

```bash
./muxima infer --prompt "Olá, como estás?" --temperature 0.8 --top-p 0.9 --max-tokens 50
```

## 9. Fluxo Completo

```
Texto "Olá, como estás?"
  ↓ Tokenizer (BPE) -> [124,82,19...]
  ↓ Dataset (input/target)
  ↓ Transformer (Embedding + PosEnc + N*Block{Norm,CausalAttn,FFN} + FinalNorm + LMHead)
  ↓ logits [seq_len, vocab]
  ↓ CrossEntropy -> loss -> CrossEntropy::backward -> dL/dlogits = (p - y)/N
  ↓ AutogradGraph backward -> dL/dW
  ↓ SGD (lr=0.0003) -> W -= lr * grad
  ↓ ParameterList
  ↓ Checkpoint
  ↓ infer/chat (autoregressivo)
```

## 10. Comandos Make

```bash
make              # compila apenas src/main.cpp -> ./muxima (CLI)
make t            # todos os 23 tests (tensor, math, transformer, etc)
make test         # 12 tests rápidos
make test_bpe     # treina BPE 300 vocabs
make test_tokenizer
make test_dataset
make test_json
make test_io
make clean        # rm -f test_* muxima checkpoint_test.json
rm -f config/tokenizer.json && make test_bpe  # regenera tokenizer
```

## 11. Próximos Passos

- [ ] BPE UTF-8 byte-level + frequência por palavra para corpus grande
- [ ] `Variable` em `Embedding/LayerNorm/Transformer` + `ParameterList` completo
- [ ] `Trainer` com `AutogradGraph` integrado e `batch_size`
- [ ] Checkpoint binário `weights.bin` (evitar JSON para 16M floats)
- [ ] Sampling `temperature/top-k/top-p` no `infer`/`chat`
```

