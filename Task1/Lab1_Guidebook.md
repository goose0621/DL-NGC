# Lab 1 Guidebook
## MNIST Classifier + Weights & Biases Monitoring
### 実験1ガイドブック：環境構築から対照実験まで

---

## References / 参考文献

Before starting, read and watch the following materials. They are the foundation this guidebook is built on.  
作業を始める前に、以下の資料を読んで視聴すること。本ガイドブックはこれらを前提として書かれている。

| # | Resource / 資料 | Purpose / 目的 |
|---|----------------|---------------|
| 1 | [PyTorch — Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | PyTorch training loop fundamentals / PyTorch訓練ループの基礎 |
| 2 | [Weights & Biases — Quickstart](https://docs.wandb.ai/quickstart) | W&B logging setup / W&Bのログ設定 |
| 3 | [3Blue1Brown — But what is a Neural Network? (Chapter 1)](https://www.youtube.com/watch?v=aircAruvnKk) | Visual intuition for neural networks / ニューラルネットワークの視覚的直感 |
| 4 | [3Blue1Brown — Gradient descent, how neural networks learn (Chapter 2)](https://www.youtube.com/watch?v=IHZwWFHWa-w) | Understanding gradient descent / 勾配降下法の理解 |
| 5 | [3Blue1Brown — Neural Networks full playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Complete series / シリーズ全体 |

> **Recommended order / 推奨順序:** Watch 3Blue1Brown chapters 1–2 first to build intuition, then work through the PyTorch 60-min Blitz, then follow this guidebook.  
> まず3Blue1Brownの第1・2章を視聴して直感を養い、次にPyTorch 60分ブリッツを進め、その後このガイドブックに従う。

---

## Table of Contents / 目次

1. [Environment Setup / 環境構築](#1-environment-setup--環境構築)
2. [Dataset: MNIST Download & Visualization / データセット](#2-dataset-mnist-download--visualization--データセット)
3. [Building LeNet-5 by Hand / LeNet-5の手書き実装](#3-building-lenet-5-by-hand--lenet-5の手書き実装)
4. [Training Loop / 訓練ループ](#4-training-loop--訓練ループ)
5. [Weights & Biases Integration / W&B接続](#5-weights--biases-integration--wb接続)
6. [Controlled Experiments / 対照実験](#6-controlled-experiments--対照実験)
7. [Goal Checklist / 目標チェックリスト](#7-goal-checklist--目標チェックリスト)

---

## 1. Environment Setup / 環境構築

### 1.1 Installing PyTorch / PyTorchのインストール

Confirm Python >= 3.9, then install according to your hardware.  
Python >= 3.9 を確認してから、ハードウェアに合わせてインストールする。

```bash
# CPU only
pip install torch torchvision

# GPU (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Verify / 確認:** Run `torch.cuda.is_available()` — should return `True` in a GPU environment.

### 1.2 Installing Weights & Biases

```bash
pip install wandb python-dotenv
```

After installation, create an account at [wandb.ai](https://wandb.ai) and obtain your API Key from Settings.  
インストール後、wandb.ai でアカウントを作成し、SettingsページでAPIキーを取得する。

## 2. Dataset: MNIST Download & Visualization / データセット

### 2.1 Understanding the Data Pipeline / データパイプラインの理解

PyTorch data loading has two layers:  
PyTorchのデータ読み込みは2層構造になっている。

- **Dataset** — reads individual samples / 個別サンプルを読み込む
- **DataLoader** — handles batching, shuffling, and multiprocessing / バッチ化・シャッフル・マルチプロセスを担当する

MNIST is a grayscale image dataset: 28×28 pixels, 10 classes (digits 0–9), 60,000 training images and 10,000 test images.  
MNISTはグレースケール画像データセット：28×28ピクセル、10クラス（数字0〜9）、訓練60,000枚・テスト10,000枚。

### 2.2 What Transforms Do / Transformの役割

| Transform | Function / 機能 | Notes / 備考 |
|-----------|----------------|--------------|
| `ToTensor()` | PIL image → [0,1] Tensor | Also reshapes HxW → CxHxW |
| `Normalize((0.1307,),(0.3081,))` | Z-score normalization | Mean and std computed from MNIST training set |

> **Important / 重要:** `Normalize` here is a static data preprocessing step. It is completely different from `BatchNorm` inside the model. The former uses fixed precomputed values; the latter has learnable parameters updated during training.  
> ここでの `Normalize` は静的なデータ前処理である。モデル内部の `BatchNorm` とは全く別物。前者は固定値、後者は訓練中に更新される学習可能パラメータを持つ。

### 2.3 Visualization / 可視化

After creating the DataLoader, take one batch and display several images with their labels to confirm correct loading.  
DataLoader作成後、1バッチ取り出して画像とラベルを表示し、正しく読み込まれているか確認する。

```python
images, labels = next(iter(train_loader))
print(images.shape)  # torch.Size([64, 1, 28, 28])
```

---

## 3. Building LeNet-5 by Hand / LeNet-5の手書き実装

### 3.1 Network Architecture / ネットワーク構造

This experiment uses 28×28 input (no resize) with 3×3 convolution kernels.  
本実験では28×28入力（リサイズなし）と3×3畳み込みカーネルを使用する。

**Always compute output sizes manually before writing code — this prevents flatten dimension errors.**  
**コードを書く前に必ず各層の出力サイズを手計算する。flatten次元エラーを防ぐため。**

| Layer / 層 | Operation / 操作 | Output Shape / 出力形状 | Notes / 備考 |
|-----------|-----------------|----------------------|-------------|
| Input | — | 1×28×28 | — |
| Conv1 | Conv2d(1, 10, 3) | 10×26×26 | 28−3+1=26 |
| BN1 | BatchNorm2d(10) | 10×26×26 | learnable / 学習可能 |
| Activation | ReLU | 10×26×26 | — |
| Pool1 | MaxPool2d(2,2) | 10×13×13 | — |
| Conv2 | Conv2d(10, 20, 3) | 20×11×11 | 13−3+1=11 |
| BN2 | BatchNorm2d(20) | 20×11×11 | learnable / 学習可能 |
| Activation | ReLU | 20×11×11 | — |
| Pool2 | MaxPool2d(2,2) | 20×5×5 | floor(11/2)=5 |
| Flatten | flatten | 500 | 20×5×5=500 |
| FC1 | Linear(500, 144) | 144 | — |
| FC2 | Linear(144, 72) | 72 | — |
| FC3 | Linear(72, 10) | 10 | logits output |

> **Key point / 重要:** The flatten dimension = channels × height × width. Recompute this number every time you change convolution parameters.  
> flatten次元 = チャンネル数 × 高さ × 幅。畳み込みパラメータを変更するたびに必ず再計算する。

### 3.2 Correct BatchNorm Usage / BatchNormの正しい書き方

BatchNorm **must** be defined in `__init__`, not created on-the-fly inside `forward`. Creating it in `forward` instantiates a new untrained layer every call, so its statistics are never learned.  
BatchNormは `__init__` 内で定義しなければならない。`forward` 内で毎回生成すると、未訓練の新しい層が都度作られ、統計量が一切学習されない。

Use `use_batchnorm` as a constructor parameter to toggle it on/off:

```python
self.bn1 = nn.BatchNorm2d(10) if use_batchnorm else nn.Identity()
self.bn2 = nn.BatchNorm2d(20) if use_batchnorm else nn.Identity()
```

`nn.Identity()` is a pass-through layer that does nothing — no need to change `forward` logic.  
`nn.Identity()` は何もしない直通層。`forward` のロジックを変えずにBNのオン・オフができる。

**Correct order in `forward` / `forward` 内の正しい順序:**

```
conv → bn → activation → pool
```

```python
c1 = self.conv1(x)
c1 = self.bn1(c1)      # before activation / 活性化の前
c1 = F.relu(c1)
s2 = F.max_pool2d(c1, 2)
```

### 3.3 Activation Function Choice / 活性化関数の選択

- **Convolutional layers / 畳み込み層:** Use `F.relu`. Sigmoid causes vanishing gradients in deeper networks and is not recommended.  
  Sigmoidは深層ネットワークで勾配消失を引き起こすため非推奨。
- **Output layer FC3:** No activation function. Output raw logits directly — `CrossEntropyLoss` handles the rest internally.  
  活性化関数なし。生のlogitsをそのまま出力し、`CrossEntropyLoss` が内部で処理する。

### 3.4 Layers That Need `__init__` vs. Functional Calls

| Needs `__init__` / 要定義 | Use as function / 関数として使用 |
|--------------------------|-------------------------------|
| Conv2d — has weights | F.relu — no parameters |
| Linear — has weights | F.max_pool2d — no parameters |
| BatchNorm — has learnable params + statistics | torch.flatten — no parameters |

---

## 4. Training Loop / 訓練ループ

### 4.1 Loss Function & Optimizer / 損失関数とオプティマイザ

**Always use `CrossEntropyLoss` for classification — not `MSELoss`.**  
**分類タスクには必ず `CrossEntropyLoss` を使う。`MSELoss` は不可。**

`MSELoss` expects two tensors of the same shape. Since labels are integer indices with shape `[batch]`, while outputs are `[batch, 10]`, the shapes are incompatible and will raise a `RuntimeError`.  
`MSELoss` は同形状のテンソルを期待するが、labelsは `[batch]` の整数インデックスで、outputsは `[batch, 10]` のため形状が合わずエラーになる。

| Optimizer / オプティマイザ | Characteristics / 特徴 | Best for / 用途 |
|--------------------------|----------------------|----------------|
| SGD + momentum=0.9 | Global learning rate, inertia, better generalization | Production models after tuning |
| Adam | Adaptive per-parameter lr, fast convergence, less sensitive to lr choice | Early-stage experimentation |

**SGD parameters / SGDのパラメータ:**
- `net.parameters()` — passes all learnable parameters to the optimizer
- `lr` — step size; gradient gives direction, lr controls how far to step
- `momentum=0.9` — 90% historical velocity + 10% current gradient; reduces oscillation

### 4.2 GPU Support / GPU対応

Three places require `.to(device)` / 3箇所に `.to(device)` が必要:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Model / モデル
net = Net().to(device)

# 2. Training data / 訓練データ
inputs, labels = inputs.to(device), labels.to(device)

# 3. Validation data / 検証データ
images, labels = images.to(device), labels.to(device)
```

### 4.3 Validation Loop / 検証ループ

Two essential steps before evaluating:  
評価前に2つの必須手順がある。

1. `net.eval()` — switches BatchNorm to inference mode (uses running statistics, not batch statistics)  
   BatchNormを推論モードに切り替える（バッチ統計ではなく実行統計を使用）
2. `torch.no_grad()` — disables gradient computation to save memory and speed up evaluation  
   勾配計算を無効にしてメモリ節約と高速化

**Accuracy formula / 精度の計算式:**

```python
_, predicted = torch.max(outputs, 1)
correct += (predicted == labels).sum().item()
accuracy = correct / total   # use / not //
```

> **Structure note / 構造上の注意:** Wrap the validation logic in a standalone function. Call it once per epoch at the end of the epoch loop — **not** inside the batch loop. Calling it per batch reruns the entire test set every batch, which is extremely slow.  
> 検証ロジックは独立した関数にまとめる。epochループの末尾で1回だけ呼び出す。バッチループ内で呼び出すと毎バッチごとにテストセット全体を走らせることになり非常に遅い。

---

## 5. Weights & Biases Integration / W&B接続

### 5.1 Three Core APIs / 3つのコアAPI

| API | When to call / 呼び出しタイミング | Key parameters / 主なパラメータ |
|-----|--------------------------------|-------------------------------|
| `wandb.init()` | Once before training starts / 訓練開始前に1回 | `project`, `name`, `config` |
| `wandb.log({})` | After each epoch ends / 各epoch終了後 | dict — keys become curve names |
| `wandb.finish()` | Once after all training ends / 全訓練終了後に1回 | — |

```python
wandb.init(
    project="mnist-lenet5",
    name="lr=1e-3_bn=True",
    config={"lr": 1e-3, "use_batchnorm": True, "epochs": 20}
)

# At the end of each epoch / 各epochの末尾で
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_acc": val_acc
})

wandb.finish()
```

### 5.2 config Dictionary / config辞書の注意

The `config` dict records hyperparameters. W&B displays them in the dashboard for easy comparison across runs.  
`config` 辞書はハイパーパラメータを記録する。W&Bはダッシュボードに表示して実験間の比較を容易にする。

> **Syntax note / 構文上の注意:** Keys must be strings in quotes. `{lr: 1e-3}` uses a variable as a key (Python treats `lr` as a variable name). Use `{"lr": 1e-3}` instead.  
> キーは必ず文字列（引用符付き）にする。`{lr: 1e-3}` は変数名として解釈される。`{"lr": 1e-3}` と書くこと。

---

## 6. Controlled Experiments / 対照実験

### 6.1 Learning Rate Comparison / 学習率比較（lr = 1e-1 / 1e-3 / 1e-5）

| Learning Rate / 学習率 | Training Behavior / 訓練挙動 | Reason / 理由 |
|----------------------|----------------------------|--------------|
| 1e-1 (too large / 大きすぎ) | Loss stays high and nearly flat; accuracy low and oscillating | Step too large — jumps past optimum, cannot converge / ステップ大きすぎて最適点を飛び越え収束不可 |
| 1e-3 (appropriate / 適切) | Good accuracy from early epochs; converges quickly | Efficient path to low-loss region / 低lossの領域へ効率的に到達 |
| 1e-5 (too small / 小さすぎ) | Accuracy climbs slowly from near zero; smooth curve but slow | Tiny steps — can converge given enough epochs, but inefficient / 小さすぎるステップ — epochが十分あれば収束するが非効率 |

> **Common misconception / よくある誤解:** The 1e-5 curve looks smooth and has a clear arc from bad to good — which can feel more satisfying visually. But this is only because the learning process has been stretched out over many epochs. Given the same number of epochs, 1e-3 achieves far higher accuracy. The goal of learning rate selection is fast and stable convergence, not a visually pleasing curve.  
> 1e-5の曲線は滑らかで「悪い状態から良い状態への過程」が見えて視覚的に満足感がある。しかしこれは学習過程が多くのepochにわたって引き伸ばされているだけ。同じepoch数では1e-3の方が圧倒的に高い精度を達成する。学習率選択の目標は速くて安定した収束であり、曲線の見た目の美しさではない。

**Distinguishing large vs. small lr when both look "stuck" / 両者が「停滞して見える」場合の見分け方:**

- Too small: accuracy shows a slow but consistent upward trend / 小さすぎ：精度にゆっくりとした上昇傾向がある
- Too large: accuracy is flat or erratic with no improvement trend / 大きすぎ：精度が横ばいまたは不規則で改善傾向なし

### 6.2 BatchNorm Ablation / BatchNorm除去実験

| Configuration / 設定 | Initial Behavior / 初期挙動 | Stability / 安定性 |
|--------------------|--------------------------|------------------|
| With BN / BNあり | Lower starting loss; better accuracy from epoch 1 | Stable; inter-layer input distributions are controlled |
| Without BN / BNなし | Higher starting loss; more epochs needed before improving | Potential gradient vanishing or explosion |

BatchNorm normalizes each layer's input to mean 0, variance 1 at the start of training, stabilizing gradient flow and giving the model a better starting point. This is why models with BN show better initial accuracy.  
BatchNormは訓練開始時に各層の入力を平均0・分散1に正規化し、勾配の流れを安定させてモデルにより良い出発点を与える。これがBNありのモデルが初期から精度が高い理由。

### 6.3 Organizing Experiments / 実験の整理

Wrap each experiment in a function accepting `lr` and `use_batchnorm` as parameters.  
各実験を `lr` と `use_batchnorm` を引数に取る関数にまとめる。

**Four experiment configurations / 4つの実験構成:**

| Run | lr | use_batchnorm | Purpose |
|-----|----|--------------|---------|
| 1 | 1e-3 | True | Baseline / ベースライン |
| 2 | 1e-1 | True | Large lr / 大きい学習率 |
| 3 | 1e-5 | True | Small lr / 小さい学習率 |
| 4 | 1e-3 | False | No BN ablation / BN除去 |

**Reset model between runs / 実験間のモデルリセット:**

```python
# Reinstantiate to reset all weights — no kernel restart needed
# 重みをリセットするため再インスタンス化する。カーネル再起動は不要
net = Net(use_batchnorm=True).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
```

All four runs should be logged to the same W&B `project`. Give each run a descriptive `name`. W&B will automatically generate comparison curves in the dashboard.  
4つの実験はすべて同じW&B `project` にログする。各実験に説明的な `name` をつける。W&Bはダッシュボードに自動的に比較曲線を生成する。

---

## 7. Goal Checklist / 目標チェックリスト

| Goal / 目標 | How to Verify / 確認方法 | Pass Condition / 合格条件 |
|------------|------------------------|--------------------------|
| test accuracy > 98% | Validation loop output / 検証ループの出力 | `correct / total >= 0.98` |
| W&B dashboard has curves | Log in to wandb.ai | `train_loss`, `val_loss`, `val_acc` all present |
| Can describe 3 lr behaviors | Compare with Section 6.1 | Understands convergence speed and stability differences |
| Can describe no-BN behavior | Compare initial curve values | Understands BN's role in early training stability |

---

