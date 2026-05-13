# Lab 2 Guidebook
## Message Passing in Graph Neural Networks — Concept Experiment
### 実験2ガイドブック：グラフニューラルネットワークのメッセージパッシング概念実験

---

## References / 参考文献

| # | Resource / 資料 | Purpose / 目的 |
|---|----------------|---------------|
| 1 | [PyTorch Geometric — Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) | PyG の基本的な使い方 / Basic PyG usage |
| 2 | [CS224W — Graph Neural Networks (Stanford)](https://web.stanford.edu/class/cs224w/) | GNN の理論的基礎 / Theoretical foundation of GNNs |
| 3 | [Distill.pub — A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) | 視覚的な GNN 入門 / Visual GNN introduction |
| 4 | [scikit-learn — Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) | 正規化手法の詳細比較 / Normalization methods comparison |

> **Recommended reading / 推奨読み物:** Read the Distill.pub article before starting this lab.  
> 実験を始める前に Distill.pub の記事を読むこと。

---

## Task Overview / タスク概要

In this lab there is **no training loop and no dataset**. The goal is purely conceptual: to thoroughly understand what happens inside a single GNN layer before writing any real model.  
この実験には**訓練ループもデータセットもない**。目標は純粋に概念的なもの：実際のモデルを書く前に、GNN の1層の中で何が起きているかを完全に理解すること。

**The output of this lab is not a model — it is understanding.** After completing it, you should be able to explain what GCNConv is doing in Lab 3.  
**この実験の産物はモデルではなく、理解である。** 完成後は Lab 3 の GCNConv が何をしているかを説明できるようになること。

---

## Core Questions / コア問題

1. **Before and after one round of message passing, what does a node "know"?**  
   1回のメッセージパッシングの前後で、ノードは何を「知っている」か？

2. **Why does the adjacency matrix appear in the GCN update equation?**  
   GCN の更新式に隣接行列が現れるのはなぜか？

3. **What does "1 layer = 1 hop" mean concretely?**  
   「1層 = 1ホップ」とは具体的に何を意味するか？

4. **What capability does the model gain when learnable weights W are added?**  
   学習可能な重み W を加えると、モデルはどんな能力を得るか？

---

## Table of Contents / 目次

1. [Installing PyTorch Geometric / PyTorch Geometricのインストール](#1-installing-pytorch-geometric--pytorch-geometricのインストール)
2. [The Core Idea: Message Passing / コアアイデア：メッセージパッシング](#2-the-core-idea-message-passing--コアアイデアメッセージパッシング)
3. [Building the Graph by Hand / 手動グラフ構築](#3-building-the-graph-by-hand--手動グラフ構築)
4. [Precise Concept Definitions / 概念の正確な定義](#4-precise-concept-definitions--概念の正確な定義)
5. [Step-by-Step Message Passing / ステップごとのメッセージパッシング](#5-step-by-step-message-passing--ステップごとのメッセージパッシング)
6. [Visualization / 可視化](#6-visualization--可視化)
7. [Adding Learnable Weights: A Simple GNN / 学習可能な重みの追加：シンプルなGNN](#7-adding-learnable-weights-a-simple-gnn--学習可能な重みの追加シンプルなgnn)
8. [Verification with PyG / PyGによる検証](#8-verification-with-pyg--pygによる検証)
9. [Reflection Questions / 振り返り問題](#9-reflection-questions--振り返り問題)
10. [Self-Check Before Lab 3 / Lab 3前の自己確認](#10-self-check-before-lab-3--lab-3前の自己確認)

---

## 1. Installing PyTorch Geometric / PyTorch Geometricのインストール

```bash
python -c "import torch; print(torch.__version__)"
pip install torch_geometric
```

```python
import torch_geometric
print(torch_geometric.__version__)
```

> **Note / 注意:** `torch_scatter` and `torch_sparse` are optional — install only if you encounter errors.  
> `torch_scatter` と `torch_sparse` はオプション——エラーが出た場合のみインストールする。

---

## 2. The Core Idea: Message Passing / コアアイデア：メッセージパッシング

Every GNN layer does three things. Memorize this framework — everything that follows is a variation of it.  
すべての GNN 層は3つのことを行う。このフレームワークを記憶すること。以下のすべての内容はその変形である。

```
For each node v / 各ノード v に対して:
  1. MESSAGE:    each neighbor u sends its features to v / 各隣接ノード u が自身の特徴を v に送る
  2. AGGREGATE:  v collects all incoming messages (sum / mean / max) / v はすべての受信メッセージを収集する
  3. UPDATE:     v combines its own features with the aggregated result / v は自身の特徴と集約結果を合わせる
```

```python
for each node v:
    messages = [features[u] for u in neighbors(v)]
    aggregated = mean(messages)
    features[v] = update(features[v], aggregated)
```

GCN, GAT, GraphSAGE, and GIN are all variations of this framework.  
GCN、GAT、GraphSAGE、GIN はすべてこのフレームワークの変形。

---

## 3. Building the Graph by Hand / 手動グラフ構築

### The Scenario / シナリオ

- **Nodes / ノード:** 6 researchers, features: `[publications, h-index, years_active]` / 6人の研究者、特徴量：`[論文数, h指数, 活動年数]`
- **Edges / エッジ:** co-authorship (undirected) / 共著関係（無向）

### Why Normalize? / なぜ正規化するか？

Raw features have different scales — large-valued features dominate computation without normalization.  
生の特徴量はスケールが異なる——正規化なしでは値の大きい特徴が計算を支配する。

| Node / ノード | Role / 役割 | Publications / 論文数 | H-index / h指数 | Years / 活動年数 |
|-------------|------------|----------------------|----------------|----------------|
| 0 | Senior Professor / 資深教授 | 161 | 36 | 25 |
| 1 | Assoc. Professor / 副教授 | 102 | 24 | 15 |
| 2 | Asst. Professor / 助教 | 64 | 17 | 11 |
| 3 | PhD Year 3 / 博士3年 | 24 | 9 | 6 |
| 4 | PhD Year 1 / 博士1年 | 15 | 5 | 3 |
| 5 | Visiting Researcher / 客員研究員 | 83 | 20 | 13 |

Normalization ranges / 正規化範囲: publications [5, 200], h-index [1, 40], years [1, 25]

**Normalization methods / 正規化手法の比較:**

| Method / 手法 | Formula / 公式 | Output / 出力範囲 | When to use / 使用場面 |
|--------------|--------------|-----------------|----------------------|
| Min-Max | `(x - min) / (max - min)` | [0, 1] | Fixed range needed — used here / 固定範囲が必要、本実験で使用 |
| Z-score | `(x - mean) / std` | mean=0, std=1 | Approximately normal data / 正規分布に近いデータ |
| Robust | `(x - median) / IQR` | No fixed range | Data with outliers / 外れ値を含むデータ |

Min-Max example for publications (min=5, max=200) / 論文数の例（min=5, max=200）:

```
Node 0: (161 - 5) / (200 - 5) = 156/195 ≈ 0.80
Node 4: ( 15 - 5) / (200 - 5) =  10/195 ≈ 0.05
Node 1: (102 - 5) / (200 - 5) =  97/195 ≈ 0.50
```

### Node Feature Matrix / ノード特徴行列

```python
import torch
import torch.nn.functional as F

X = torch.tensor([
    [0.8,  0.9,  1.0],  # Node 0: Senior Professor / 資深教授
    [0.5,  0.6,  0.6],  # Node 1: Assoc. Professor / 副教授
    [0.3,  0.4,  0.4],  # Node 2: Asst. Professor / 助教
    [0.1,  0.2,  0.2],  # Node 3: PhD Year 3 / 博士3年
    [0.05, 0.1,  0.1],  # Node 4: PhD Year 1 / 博士1年
    [0.4,  0.5,  0.5],  # Node 5: Visiting Researcher / 客員研究員
], dtype=torch.float)
```

### Adjacency Matrix / 隣接行列

```python
A = torch.tensor([
    [0, 1, 1, 0, 0, 1],  # Node 0 collaborates with 1, 2, 5
    [1, 0, 1, 1, 0, 0],  # Node 1 collaborates with 0, 2, 3
    [1, 1, 0, 1, 1, 0],  # Node 2 collaborates with 0, 1, 3, 4
    [0, 1, 1, 0, 1, 0],  # Node 3 collaborates with 1, 2, 4
    [0, 0, 1, 1, 0, 0],  # Node 4 collaborates with 2, 3
    [1, 0, 0, 0, 0, 0],  # Node 5 collaborates with 0 only
], dtype=torch.float)
```

> **Before continuing / 続きの前に:** Draw this graph on paper. Label nodes 0-5 and draw edges according to the adjacency matrix.  
> 続きに進む前に：このグラフを紙に描くこと。ノード0〜5にラベルを付け、隣接行列に従ってエッジを描く。

---

## 4. Precise Concept Definitions / 概念の正確な定義

Before hand-computing, clarify the meaning of each term. Refer to the graph you just drew.  
手計算に入る前に、各用語の意味を明確にする。先ほど描いたグラフを参照しながら理解すること。

**Neighbor / 隣接ノード**  
A node directly connected by an edge. Determined by graph structure, not feature values. Node 2's neighbors are 0, 1, 3, 4 — regardless of their feature values.  
エッジで直接繋がっているノード。グラフ構造によって決まり、特徴量の値とは無関係。ノード2の隣接ノードは0、1、3、4。

**Neighbor features / 隣接ノードの特徴量**  
The vector currently stored by a neighbor. In layer 1 this is the original input. In layer 2 this is the vector updated by layer 1 — the neighbor has already absorbed its own neighbors' information and is no longer the original feature.  
隣接ノードが現在保持しているベクトル。第1層では元の入力特徴量。第2層では第1層で更新済みのベクトル——隣接ノードはすでに自身の隣接情報を融合しており、元の特徴量ではない。

**Message / メッセージ**  
What a neighbor decides to send. In GCN: its feature vector as-is. In GAT: transformed or weighted by relationship strength.  
隣接ノードが送ることを決定した内容。GCN では特徴ベクトルをそのまま。GAT では関係の強さで重み付けされる。

**Aggregate / 集約**  
Merging multiple messages into one fixed-size vector. Direct concatenation is impossible since neighbor counts differ. Common methods:  
複数のメッセージを一つの固定サイズベクトルにまとめる。隣接ノード数が異なるため直接結合は不可。主な方法：
- Mean / 平均 (GCN): normalizes by neighbor count / 隣接ノード数で正規化
- Sum / 和 (GIN): preserves neighbor count information / 隣接ノード数の情報を保持
- Max / 最大値 (GraphSAGE variant): retains most prominent features / 最も顕著な特徴を保持

**Update / 更新**  
Combining aggregated result with own features to produce new features. This is where message passing truly changes the node — it goes from knowing only itself to knowing itself and its neighborhood.  
集約結果と自身の特徴を合わせて新しい特徴を生成する。ここがメッセージパッシングがノードを真に「変える」場所——自分だけを知る状態から、自分と近傍を知る状態に変わる。

---

## 5. Step-by-Step Message Passing / ステップごとのメッセージパッシング

### Step 1 — MESSAGE / メッセージ

```python
# Node 2's neighbors are 0, 1, 3, 4 / ノード2の隣接ノードは 0, 1, 3, 4
neighbors_of_2 = [0, 1, 3, 4]
messages_to_2 = X[neighbors_of_2]
print("Messages received by Node 2 / ノード2が受け取ったメッセージ:")
print(messages_to_2)
```

### Step 2 — AGGREGATE / 集約

```python
aggregated_2 = messages_to_2.mean(dim=0)
print("Aggregated message for Node 2 / ノード2の集約結果:")
print(aggregated_2)
```

> **Think / 考えてみよう:** The aggregated vector represents the combined influence of all neighboring nodes on this node. Node 2's neighbors range from Senior Professor (high) to PhD Year 1 (low) — the result is pulled toward their collective center of gravity.  
> 集約後のベクトルは、この ノードに対するすべての隣接ノードの複合的な影響力を表す。ノード2の隣接ノードは資深教授（高値）から博士1年（低値）まで幅広く——結果はそれらの重心に引き寄せられる。

### Step 3 — UPDATE / 更新

```python
updated_2 = (X[2] + aggregated_2) / 2
print("Node 2 BEFORE / メッセージパッシング前:", X[2])
print("Node 2 AFTER  / メッセージパッシング後:", updated_2)
```

### Matrix Form: All Nodes at Once / 行列形式：全ノードを一度に

```python
D_inv = torch.diag(1.0 / A.sum(dim=1))
A_norm = D_inv @ A

X_aggregated = A_norm @ X
print("Aggregated features for all nodes / 全ノードの集約後特徴:")
print(X_aggregated)
```

> **Key insight / 重要な洞察:** `A_norm @ X` simultaneously computes neighbor averaging for all 6 nodes. The graph is fundamentally a matrix — so aggregation maps naturally onto matrix multiplication.  
> `A_norm @ X` は6つの全ノードの隣接平均を同時に計算する。グラフは本質的に行列であるため、集約演算は自然に行列積に対応する。

---

## 6. Visualization / 可視化

### Graph Structure / グラフ構造

```python
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.from_numpy_array(A.numpy())
labels = {
    0: "Prof.\n(Senior)", 1: "Prof.\n(Assoc.)", 2: "Asst.\nProf.",
    3: "PhD\nYr3", 4: "PhD\nYr1", 5: "Visiting\nRes."
}

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, labels=labels, node_color=X[:, 1].numpy(),
        cmap=plt.cm.Blues, node_size=2000, font_size=8,
        font_weight='bold', edge_color='gray', width=2)
plt.title("Research Collaboration Network (color = h-index)\n研究コラボレーションネットワーク（色 = h指数）")
plt.tight_layout()
plt.show()
```

### Feature Change After Message Passing / メッセージパッシング後の特徴変化

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
feature_names = ['Publications', 'H-index', 'Years Active']
x_pos = np.arange(len(feature_names))
width = 0.12
colors = plt.cm.tab10(np.linspace(0, 1, 6))

for node_id in range(6):
    axes[0].bar(x_pos + node_id * width, X[node_id].numpy(), width,
                label=labels[node_id].replace('\n', ' '), color=colors[node_id])
axes[0].set_title("BEFORE / メッセージパッシング前")
axes[0].set_xticks(x_pos + width * 2.5)
axes[0].set_xticklabels(feature_names)
axes[0].set_ylim(0, 1.1)
axes[0].legend(fontsize=8)

for node_id in range(6):
    axes[1].bar(x_pos + node_id * width, X_aggregated[node_id].numpy(), width,
                label=labels[node_id].replace('\n', ' '), color=colors[node_id])
axes[1].set_title("AFTER 1 hop / メッセージパッシング後（1ホップ）")
axes[1].set_xticks(x_pos + width * 2.5)
axes[1].set_xticklabels(feature_names)
axes[1].set_ylim(0, 1.1)
axes[1].legend(fontsize=8)

plt.suptitle("Effect of One Round of Message Passing / 1ラウンドの効果", fontweight='bold')
plt.tight_layout()
plt.show()
```

> **Observe / 観察:** After one round, Node 4 rises and Node 0 falls — for the same reason. After message passing, what each node stores is no longer "what I originally am" but the result of mutual influence between itself and its neighborhood.  
> 1ラウンド後、ノード4は上昇しノード0は低下する——同じ理由で。メッセージパッシング後、各ノードが保持するのは「自分が元々何であるか」ではなく、自身と近傍の相互作用の結果である。

---

## 7. Adding Learnable Weights: A Simple GNN / 学習可能な重みの追加：シンプルなGNN

So far, message passing only **averages** — no learning. A real GNN adds learnable weight matrix W.  
ここまでのメッセージパッシングは**平均する**だけで学習なし。実際の GNN は学習可能な重み行列 W を追加する。

```
X' = σ(A_norm @ X @ W)
```

### 7.1 Forward Pass with Random Weights / ランダム重みによる順伝播

```python
import torch.nn as nn

torch.manual_seed(42)
W = torch.randn(3, 4) * 0.1

X_agg = A_norm @ X
X_transformed = X_agg @ W
X_out = F.relu(X_transformed)

print("Input shape / 入力形状:", X.shape)      # [6, 3]
print("Output shape / 出力形状:", X_out.shape)  # [6, 4]
```

> **Key insight / 重要な洞察:** Output dimension changed from 3 to 4. W can expand, compress, or rotate the feature space — this is the source of the model's expressive power.  
> 出力次元が3から4に変わった。W は特徴空間を拡張・圧縮・回転できる——モデルの表現力はここから来る。

### 7.2 Replace W with nn.Linear / W を nn.Linear に置き換える

```python
torch.manual_seed(42)
linear = nn.Linear(3, 4, bias=False)
X_out_linear = F.relu(linear(A_norm @ X))
print(X_out_linear[:3])
```

### 7.3 What Happens Between Layers? / 層と層の間で何が起きるか？

**Inside a single layer**: Message → Aggregate → Update. Input is X, output is updated H.  
**単一層の内部**：Message → Aggregate → Update。入力は X、出力は更新された H。

> **Core equivalence / コアの等価関係:**  
> **1 layer completes = 1 round of message passing = information propagates 1 hop.**  
> **1層の実行完了 = 1ラウンドのメッセージパッシング = 情報が1ホップ伝播する。**  
> After k layers, each node's features have absorbed information from within k hops.  
> k 層後、各ノードの特徴は k ホップ以内の情報を吸収している。

**Between layers**, the output H of the previous layer becomes the input to the next. Key points:  
**層と層の間**、前の層の出力 H が次の層への入力になる。重要な点：

- **Feature meaning changes / 特徴の意味が変わる**: H1 is no longer original features — it has absorbed 1-hop neighbor information. Layer 2 propagates one more hop on top of H1, so after layer 2 each node has seen within 2 hops.  
  H1 はすでに1ホップの隣接情報を融合しており、元の特徴量ではない。第2層は H1 の上でさらに1ホップ伝播するため、第2層後は各ノードが2ホップ以内を見ている。

- **Dimension controlled by W / 次元は W が制御する**: Node count (6) stays constant across all layers; only feature dimension changes.  
  ノード数（6）はすべての層で一定。変わるのは特徴次元のみ。

- **Non-linearity is necessary / 非線形性は必須**: Without activation functions, stacked linear transforms collapse into one layer. ReLU makes each layer learn a different representation.  
  活性化関数なしでは積み重ねた線形変換は1層に統合される。ReLU で各層が異なる表現を学習できる。

```
Input X   [6, 3]  ← original features / 元の特徴量
    ↓ Layer 1 (W1: 3→4, ReLU)
H1        [6, 4]  ← 1-hop info fused / 1ホップ情報を融合
    ↓ Layer 2 (W2: 4→2, ReLU)
H2        [6, 2]  ← 2-hop info fused / 2ホップ情報を融合
```

### 7.4 Stacking Two Layers / 2層積み重ね

```python
torch.manual_seed(42)
linear1 = nn.Linear(3, 4, bias=False)
linear2 = nn.Linear(4, 2, bias=False)

H1 = F.relu(linear1(A_norm @ X))       # 1-hop / 1ホップ
H2 = F.relu(linear2(A_norm @ H1))      # 2-hop / 2ホップ

print("After layer 1 / 第1層後:", H1.shape)   # [6, 4]
print("After layer 2 / 第2層後:", H2.shape)   # [6, 2]
```

> **Think / 考えてみよう:** After 2 layers, Node 4 has indirectly received information from Node 0 despite no direct edge. Path: Node 4 → Node 2 or 3 → Node 0.  
> 2層後、ノード4は直接エッジのないノード0の情報を間接的に受け取っている。経路：ノード4 → ノード2または3 → ノード0。

### 7.5 What Lab 3 Adds / Lab 3 で追加されること

| This lab / 本実験 | Lab 3 adds / Lab 3 で追加 |
|------------------|--------------------------|
| Random fixed W / ランダムで固定された W | W learned via backpropagation / 誤差逆伝播で学習される W |
| No task, no labels / タスクなし | Node classification on Cora / Cora でのノード分類 |
| Small hand-crafted graph / 手作りの小グラフ | Cora citation network (2708 nodes) / Cora 引用ネットワーク（2708ノード） |
| No training loop / 訓練ループなし | Full train / val / test loop / 完全な訓練・検証・テストループ |

---

## 8. Verification with PyG / PyGによる検証

```python
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

edge_index = A.nonzero().t().contiguous()
data = Data(x=X, edge_index=edge_index)

conv = GCNConv(in_channels=3, out_channels=3, bias=False)
with torch.no_grad():
    conv.lin.weight.copy_(torch.eye(3))

out = conv(data.x, data.edge_index)
print("PyG output / PyG 出力:"); print(out)
print("Hand-computed / 手計算:"); print(X_aggregated)
```

> **Note / 注意:** GCN uses symmetric normalization `D^{-1/2} A D^{-1/2}` not `D^{-1} A`, so values differ slightly. The key check is that the influence structure is identical.  
> GCN は `D^{-1} A` ではなく対称正規化 `D^{-1/2} A D^{-1/2}` を使うため値は若干異なる。重要な確認点は影響の構造が同じであること。

---

## 9. Reflection Questions / 振り返り問題

**Conceptual / 概念的:**

1. Node 5 only collaborates with Node 0. What information does Node 5 have after one round? After two rounds?  
   ノード5はノード0とのみ共著関係がある。1ラウンド後、ノード5はどんな情報を持つか？2ラウンド後は？

2. Why does the most isolated node show the largest feature change after message passing?  
   最も孤立したノードがメッセージパッシング後に最大の特徴変化を示すのはなぜか？

3. What would happen if you ran message passing for 10 rounds?  
   メッセージパッシングを10ラウンド実行するとどうなるか？

**Mathematical / 数学的:**

4. Compute the row of `A_norm @ X` for Node 3 by hand and confirm it matches the output.  
   `A_norm @ X` のノード3の行を手で計算し、出力と一致することを確認せよ。

5. Why does GCN use `D^{-1/2} A D^{-1/2}` instead of `D^{-1} A`?  
   GCN はなぜ `D^{-1} A` ではなく `D^{-1/2} A D^{-1/2}` を使うのか？

**Looking ahead / 先を見越して:**

6. In Lab 3, `X' = σ(A_norm @ X @ W)` with learnable W. Where does learning happen? What can the model learn that pure aggregation cannot?  
   Lab 3 では学習可能な W を持つ `X' = σ(A_norm @ X @ W)` を使う。どこで学習が起きるか？純粋な集約では学習できないことを何を学習できるか？

---

## 10. Self-Check Before Lab 3 / Lab 3前の自己確認

No submission needed. If you can answer these, you are ready for Lab 3.  
提出不要。答えられれば Lab 3 への準備ができている。

**Without looking at the code, can you describe the three steps of message passing?**  
コードを見ずに、メッセージパッシングの3ステップを説明できるか？

**What is `A_norm @ X` doing? What does row i of the result represent?**  
`A_norm @ X` は何をしているか？結果の第 i 行は何を表すか？

**What is the essential difference between the input to layer 2 and the input to layer 1?**  
第2層への入力と第1層への入力の本質的な違いは何か？

**1 layer = 1 hop — so what can 3 layers see? In this graph, what nodes can Node 5 receive information from after 3 layers?**  
1層 = 1ホップ——では3層は何を見られるか？このグラフでノード5は3層後にどのノードから情報を受け取れるか？

---

## Summary / まとめ

| Concept / 概念 | Key Takeaway / 重要な要点 |
|---------------|--------------------------|
| Message | Each neighbor sends its feature vector / 各隣接ノードが特徴ベクトルを送る |
| Aggregate | Combine messages (mean / sum / max) / メッセージを集約（平均・和・最大値） |
| Update | Merge aggregated result with own features / 集約結果と自身の特徴をマージ |
| Matrix form | `A_norm @ X` computes aggregation for all nodes simultaneously / 全ノードの集約を同時計算 |
| 1 layer = 1 hop | After k layers, each node has seen within k hops / k 層後、各ノードは k ホップ以内を見ている |
| Adding W | Learnable linear transformation of feature space / 特徴空間の学習可能な線形変換 |
| Over-smoothing | Too many layers → all features converge / 層が多すぎると全特徴が均一化 |

---

*goose@NGC Lab.*
