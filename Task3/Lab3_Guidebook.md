# Lab 3 Guidebook
## Graph Convolutional Networks — Node Classification on Cora
### 実験3ガイドブック：グラフ畳み込みネットワーク — Coraノード分類

---

## References / 参考文献

| # | Paper / 論文 | Purpose / 目的 |
|---|-------------|---------------|
| 1 | [Kipf & Welling (2017) — Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | GCN original paper / GCN 原論文 |
| 2 | [Srivastava et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) | Dropout original paper / Dropout 原論文 |
| 3 | [Li et al. (2018) — Deeper Insights into Graph Convolutional Networks for Semi-Supervised Classification](https://arxiv.org/abs/1801.07606) | First systematic analysis of over-smoothing / Over-smoothing の最初の系統的分析 |

---

## Task Overview / タスク概要

In Lab 2 you computed message passing by hand and understood what happens inside a GNN layer. Lab 3 adds two things on top of that:  
Lab 2 ではメッセージパッシングを手計算し、GNN 層の内部で何が起きているかを理解した。Lab 3 はその上に2つのことを追加する：

- **Learnable weights W / 学習可能な重み W**: trained via backpropagation, no longer random and fixed / 誤差逆伝播で学習、ランダムで固定ではなくなる
- **Real task / 実際のタスク**: node classification on the Cora citation network, 7 classes / Cora 引用ネットワークでのノード分類、7クラス

By the end of this lab you will have / この実験を終えると、自然と以下のものが手元に残る：

- A working 2-layer GCN with ~80% test accuracy / テスト精度約80%の動作する2層 GCN
- train / val / test loss and accuracy curves on W&B / W&B 上の train/val/test の loss と accuracy 曲線
- Direct observations of how Dropout affects overfitting in GNNs / GNN における Dropout の過学習への影響の直接観察
- Understanding of GCN's limitations, setting up Lab 4 (GAT) / GCN の限界の理解、Lab 4（GAT）への布石

---

## Core Questions / コア問題

1. **GCNConv does the same thing as `A_norm @ X @ W` from Lab 2 — what is the only difference?**  
   GCNConv は Lab 2 の `A_norm @ X @ W` と同じことをしている——唯一の違いは何か？

2. **With only 140 training nodes, how can GCN classify all 2708 nodes?**  
   訓練ノードが140個しかないのに、GCN はどうやって2708個全ノードを分類できるか？

3. **GCN treats all neighbors equally — what problem does this cause?**  
   GCN はすべての隣接ノードを同等に扱う——これはどんな問題を引き起こすか？

4. **What happens to node features as GCN layers get deeper, and why?**  
   GCN の層が深くなるにつれ、ノード特徴量に何が起きるか？なぜか？

5. **What does the train/val gap tell you, and how does Dropout affect it?**  
   train/val のギャップは何を示しているか？Dropout はそれにどう影響するか？

---

## Table of Contents / 目次

1. [From Lab 2 to Lab 3 / Lab 2 から Lab 3 へ](#1-from-lab-2-to-lab-3--lab-2-から-lab-3-へ)
2. [Setup & Dataset / セットアップとデータセット](#2-setup--dataset--セットアップとデータセット)
3. [Understanding Cora / Cora データセットの理解](#3-understanding-cora--cora-データセットの理解)
4. [Model Design: Choosing Hidden Dimensions / モデル設計：隠れ層の次元数の決め方](#4-model-design-choosing-hidden-dimensions--モデル設計隠れ層の次元数の決め方)
5. [Building the GCN / GCN の構築](#5-building-the-gcn--gcn-の構築)
6. [Training Loop / 訓練ループ](#6-training-loop--訓練ループ)
7. [Results Analysis / 結果分析](#7-results-analysis--結果分析)
8. [Inducing Over-smoothing / Over-smoothing を手動で引き起こす](#8-inducing-over-smoothing--over-smoothing-を手動で引き起こす)
9. [Controlled Experiment: Removing Dropout / 対照実験：Dropout の除去](#9-controlled-experiment-removing-dropout--対照実験dropout-の除去)
10. [Self-Check Before Lab 4 / Lab 4 前の自己確認](#10-self-check-before-lab-4--lab-4-前の自己確認)

---

## 1. From Lab 2 to Lab 3 / Lab 2 から Lab 3 へ

The core operation in Lab 2 was / Lab 2 のコア操作は：

```python
X_out = F.relu(A_norm @ X @ W)   # W was random and fixed / W はランダムで固定
```

Lab 3 does exactly the same operation, except / Lab 3 は全く同じ操作をする、ただし：

```python
X_out = F.relu(GCNConv(X, edge_index))   # W is learned through training / W は訓練で学習される
```

`GCNConv` is PyG's encapsulation of `A_norm @ X @ W`, with symmetric normalization and bias added. Everything you computed by hand in Lab 2 is now learned automatically.  
`GCNConv` は PyG による `A_norm @ X @ W` のカプセル化で、対称正規化と bias が追加されている。Lab 2 で手計算したものが、今は自動的に学習される。

| Lab 2 (by hand / 手計算) | Lab 3 (GCNConv) |
|-------------------------|----------------|
| `A_norm = D^{-1} @ A` | `D^{-1/2} A D^{-1/2}` (symmetric normalization / 対称正規化) |
| `A_norm @ X @ W` | `GCNConv(x, edge_index)` |
| Random fixed W / ランダム固定の W | W updated via `loss.backward()` |
| 6-node hand-crafted graph / 手作り6ノードグラフ | Cora (2708 nodes, 5429 edges) |
| No task / タスクなし | Node classification, 7 classes / ノード分類、7クラス |

---

## 2. Setup & Dataset / セットアップとデータセット

```bash
pip install torch_geometric
```

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

print(f"Nodes / ノード数: {data.num_nodes}")               # 2708
print(f"Edges / エッジ数: {data.num_edges}")               # 10556
print(f"Node features / ノード特徴次元: {data.num_features}") # 1433
print(f"Classes / クラス数: {dataset.num_classes}")        # 7
print(f"Train nodes / 訓練ノード数: {data.train_mask.sum()}") # 140
print(f"Val nodes / 検証ノード数: {data.val_mask.sum()}")     # 500
print(f"Test nodes / テストノード数: {data.test_mask.sum()}") # 1000
```

> **Note / 注意:** Only 140 nodes for training, yet we classify all 2708. This is semi-supervised learning — message passing lets the model leverage the graph structure of unlabeled nodes.  
> 訓練は140ノードのみで、2708ノード全体を分類する。これは半教師あり学習——メッセージパッシングにより、ラベルなしノードのグラフ構造も活用できる。

---

## 3. Understanding Cora / Cora データセットの理解

Cora is a paper citation network / Cora は論文引用ネットワーク：

- **Nodes / ノード:** 2708 papers / 論文
- **Edges / エッジ:** citation relationships / 引用関係
- **Node features / ノード特徴量:** 1433-dim bag-of-words (keyword presence in abstract) / 1433次元単語袋（アブストラクトのキーワード有無）
- **Labels / ラベル:** 7 research areas / 7つの研究分野

| Attribute / 属性 | Meaning / 意味 | Shape / 形状 |
|-----------------|--------------|-------------|
| `data.x` | Node feature matrix / ノード特徴行列 | [2708, 1433] |
| `data.edge_index` | Edge index COO format / エッジインデックス COO 形式 | [2, 10556] |
| `data.y` | Node labels / ノードラベル | [2708] |
| `data.train_mask` | Training mask / 訓練マスク | [2708] (140 True) |
| `data.val_mask` | Validation mask / 検証マスク | [2708] (500 True) |
| `data.test_mask` | Test mask / テストマスク | [2708] (1000 True) |

> **edge_index format / edge_index の形式:** `edge_index[0]` is the source list, `edge_index[1]` is the target list — COO sparse format, far more memory-efficient than a full adjacency matrix.  
> `edge_index[0]` は始点リスト、`edge_index[1]` は終点リスト——COO 疎形式で、完全な隣接行列よりはるかにメモリ効率が良い。

### Dataset Visualization / データセットの可視化

**3.1 Class Distribution / クラス分布**

```python
import matplotlib.pyplot as plt
import numpy as np

class_names = [
    'Theory', 'Reinforcement Learning', 'Genetic Algorithms',
    'Neural Networks', 'Probabilistic Methods', 'Case Based', 'Rule Learning'
]

labels = data.y.numpy()
class_counts = [(labels == i).sum() for i in range(7)]

plt.figure(figsize=(10, 4))
bars = plt.bar(class_names, class_counts, color=plt.cm.tab10(np.linspace(0, 1, 7)))
plt.xticks(rotation=20, ha='right')
plt.ylabel("Node count / ノード数")
plt.title("Cora Class Distribution / クラス分布（7研究分野）")
for bar, count in zip(bars, class_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(count), ha='center', fontsize=9)
plt.tight_layout()
plt.show()
```

> **Observe / 観察:** The 7 classes are not evenly distributed — common in real datasets, and affects accuracy.  
> 7クラスは均等に分布していない——実際のデータセットでよく見られ、精度に影響する。

**3.2 Feature Sparsity / 特徴量の疎性**

```python
features = data.x.numpy()
sparsity = (features == 0).mean()
print(f"Sparsity / 疎性: {sparsity:.2%}")  # ~99%

plt.figure(figsize=(14, 4))
plt.imshow(features[:50, :200], aspect='auto', cmap='Blues', interpolation='none')
plt.colorbar(label='Feature value / 特徴値')
plt.xlabel("Feature dim (first 200) / 特徴次元（前200次元）")
plt.ylabel("Node index (first 50) / ノード番号（前50個）")
plt.title("Cora Node Feature Heatmap (white=0, blue=1) / 節点特徴熱力図")
plt.tight_layout()
plt.show()
```

> **Observe / 観察:** Almost entirely white — each paper has only a few keywords. GCN compensates by letting nodes "borrow" neighbors' bag-of-words through message passing.  
> ほぼ全て白——各論文はわずかなキーワードのみ。GCN はメッセージパッシングで隣接ノードの単語袋を「借用」することで補う。

**3.3 Graph Structure / グラフ構造**

```python
import networkx as nx
from matplotlib.patches import Patch

edge_index_np = data.edge_index.numpy()
mask_sub = (edge_index_np[0] < 200) & (edge_index_np[1] < 200)
edges = list(zip(edge_index_np[0][mask_sub], edge_index_np[1][mask_sub]))

G_sub = nx.Graph()
G_sub.add_nodes_from(range(200))
G_sub.add_edges_from(edges)
node_colors = [plt.cm.tab10(data.y[i].item() / 6.0) for i in range(200)]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_sub, seed=42, k=0.3)
nx.draw(G_sub, pos, node_color=node_colors, node_size=40,
        edge_color='gray', alpha=0.8, width=0.5, with_labels=False)
legend_elements = [Patch(facecolor=plt.cm.tab10(i/6.0), label=class_names[i]) for i in range(7)]
plt.legend(handles=legend_elements, loc='upper left', fontsize=8)
plt.title("Cora Subgraph — first 200 nodes, colored by class / Cora サブグラフ", fontsize=12)
plt.tight_layout()
plt.show()
```

> **Observe / 観察:** Same-class nodes cluster together — they cite each other more. This is why GCN works: graph structure encodes class information, and message passing delivers it to each node.  
> 同クラスのノードは集まる傾向がある。これが GCN が機能する理由：グラフ構造がクラス情報をエンコードし、メッセージパッシングが各ノードに届ける。

---

## 4. Model Design: Choosing Hidden Dimensions / モデル設計：隠れ層の次元数の決め方

```
Input / 入力:   1433-dim (bag-of-words / 単語袋)
    ↓ GCNConv(1433, 64) + ReLU + Dropout
Hidden / 隠れ:  64-dim
    ↓ GCNConv(64, 7)
Output / 出力:  7-dim (logits)
```

**Why 64? / なぜ64か？** There is no fixed formula — use these guiding principles / 固定された公式はない——以下の指針を使う：

1. **Task complexity / タスクの複雑さ:** 7 classes → 64 is sufficient. 512 overfits; 8 underfits.  
   7クラス → 64で十分。512は過学習、8は未学習。

2. **Compression ratio / 圧縮比:** 1433 → 64 → 7. Large compression forces learning the essence, not memorizing features.  
   1433 → 64 → 7。大きな圧縮比はモデルに特徴の記憶ではなく本質の学習を強制する。

3. **Empirical rules / 経験則:** Powers of 2 (16, 32, 64, 128); start small on small datasets and watch val accuracy.  
   2の累乗（16, 32, 64, 128）；小さなデータセットでは小さく始めて val accuracy を観察する。

**Why 2 layers? / なぜ2層か？**

- 1 layer: 1-hop only — insufficient / 1層：1ホップのみ——不十分
- 2 layers: 2-hop — sufficient for Cora / 2層：2ホップ——Cora には十分
- 3+ layers: over-smoothing begins / 3層以上：over-smoothing が始まる

---

## 5. Building the GCN / GCN の構築

```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # No softmax — CrossEntropyLoss handles it / softmax なし——CrossEntropyLoss が処理


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(
    in_channels=dataset.num_features,   # 1433
    hidden_channels=64,
    out_channels=dataset.num_classes,   # 7
    dropout=0.5
).to(device)

data = data.to(device)
print(model)
```

**What Dropout does / Dropout の役割:**

- Training: randomly drops 50% of neurons — prevents relying on any single feature / 訓練中：50%のニューロンをランダムに無効化——単一特徴への依存を防ぐ
- Testing: automatically disabled by `model.eval()` / テスト中：`model.eval()` で自動無効化
- Critical in GNNs with few training nodes (only 140) / 訓練ノードが少ない（140個のみ）GNN で特に重要

> **Note / 注意:** `self.training` is controlled by `model.train()` / `model.eval()`, ensuring different Dropout behavior during training vs. testing.  
> `self.training` は `model.train()` / `model.eval()` で制御され、訓練とテストで Dropout の挙動が異なることを保証する。

---

## 6. Training Loop / 訓練ループ

```python
import wandb, json

with open('config.json', 'r') as f:
    config = json.load(f)
wandb.login(key=config['WANDB_API_KEY'])

wandb.init(
    project="gnn-labs",
    name="gcn-cora-baseline",
    config={
        "model": "GCN", "dataset": "Cora",
        "hidden_channels": 64, "dropout": 0.5,
        "lr": 0.01, "weight_decay": 5e-4, "epochs": 200,
    }
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    results = {}
    for split, mask in [('train', data.train_mask),
                         ('val',   data.val_mask),
                         ('test',  data.test_mask)]:
        results[f'{split}_loss'] = criterion(out[mask], data.y[mask]).item()
        results[f'{split}_acc']  = (pred[mask] == data.y[mask]).float().mean().item()
    return results


for epoch in range(1, 201):
    train()
    r = evaluate()
    wandb.log({"epoch": epoch, **r})
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | "
              f"train_loss={r['train_loss']:.4f} train_acc={r['train_acc']:.4f} | "
              f"val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.4f}")

wandb.finish()
print(f"\nFinal test accuracy / 最終テスト精度: {r['test_acc']:.4f}")
```

**Key differences from Lab 1 / Lab 1 との主な違い:**

| | Lab 1 (LeNet-5) | Lab 3 (GCN) |
|--|----------------|-------------|
| Input / 入力 | `(images, labels)` batches | Entire graph at once / グラフ全体を一度に |
| Training data / 訓練データ | DataLoader batches | `data.train_mask` filter |
| Loss / 損失 | Entire batch | Training nodes only / 訓練ノードのみ |
| Graph structure / グラフ構造 | None | `edge_index` passed to model |

> **Why no DataLoader? / なぜ DataLoader を使わないか？**  
> GCN message passing requires the entire graph. Passing only a subset loses neighbor information. The full graph goes through every forward call; masks determine which nodes contribute to the loss.  
> GCN のメッセージパッシングはグラフ全体を必要とする。一部だけ渡すと隣接情報が失われる。毎回の forward でグラフ全体を渡し、マスクで loss に寄与するノードを決定する。

---

## 7. Results Analysis / 結果分析

**Expected results / 期待される結果:**

- Train accuracy / 訓練精度: ~99%
- Val accuracy / 検証精度: ~79%
- Test accuracy / テスト精度: ~81%

**What to look for in W&B curves / W&B 曲線で注目すること:**

1. **Train accuracy quickly approaches 100%:** 140 nodes is too few — overfitting is easy. Dropout and weight_decay are the main defenses.  
   訓練精度が素早く100%に近づく：140ノードは少なすぎ——Dropout と weight_decay が主な防衛手段。

2. **Val loss may start rising:** Rising val loss despite falling train loss signals overfitting.  
   val loss が上昇し始める可能性がある：train loss が下がっても val loss の上昇は過学習を示す。

3. **GCN's bottleneck:** All neighbors use the same W — cannot distinguish which neighbors matter more. Lab 4 (GAT) solves this with attention weights.  
   GCN のボトルネック：全隣接ノードが同じ W を使う——どの隣接ノードがより重要かを区別できない。Lab 4（GAT）が注意重みで解決する。

---

## 8. Inducing Over-smoothing / Over-smoothing を手動で引き起こす

Forward pass only — no training needed. Goal: directly observe "more layers → features converge."  
順伝播のみ——訓練不要。目標：「層が増える → 特徴が収束する」を直接観察する。

### 8.1 Test Accuracy vs. Layer Count / 層数別テスト精度

```python
class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


for num_layers in [2, 4, 8, 16]:
    model_deep = DeepGCN(dataset.num_features, 64, dataset.num_classes, num_layers).to(device)
    optimizer_deep = torch.optim.Adam(model_deep.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        model_deep.train()
        optimizer_deep.zero_grad()
        out = model_deep(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer_deep.step()

    model_deep.eval()
    with torch.no_grad():
        out = model_deep(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"Layers / 層数={num_layers:2d} | Test acc / テスト精度={test_acc:.4f}")
```

**Expected output / 期待される出力:**

```
Layers=  2 | Test acc=0.8100
Layers=  4 | Test acc=0.7200
Layers=  8 | Test acc=0.4500
Layers= 16 | Test acc=0.1800
```

> Accuracy drops sharply with depth. At 16 layers it approaches random guessing (~14% for 7 classes).  
> 精度は深さとともに急激に低下する。16層では乱択（7クラス→約14%）に近づく。

### 8.2 Visualization: PCA of Node Features / 可視化：ノード特徴の PCA

```python
from sklearn.decomposition import PCA
import matplotlib.cm as cm

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, num_layers in zip(axes, [2, 4, 8, 16]):
    model_vis = DeepGCN(dataset.num_features, 64, dataset.num_classes, num_layers).to(device)
    model_vis.eval()
    with torch.no_grad():
        x = data.x
        for conv in model_vis.convs[:-1]:
            x = F.relu(conv(x, data.edge_index))

    features_2d = PCA(n_components=2).fit_transform(x.cpu().numpy())
    colors = cm.tab10(data.y.cpu().numpy() / 6.0)
    ax.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=5, alpha=0.6)
    ax.set_title(f"{num_layers} layers", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle("Node Feature Distribution (PCA) — Over-smoothing Effect / Over-smoothing 効果",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Expected observation / 期待される観察:**
- 2 layers: 7 distinguishable clusters / 2層：7つの識別可能な簇
- 4 layers: clusters blur / 4層：簇が曖昧に
- 8 layers: nearly merged / 8層：ほぼ混在
- 16 layers: all nodes collapse to one point / 16層：全ノードが1点に集まる

### 8.3 Quantifying Convergence: Standard Deviation / 収束の定量化：標準偏差

```python
print("Feature std per layer count (lower = more smoothed) / 各層数の特徴標準偏差")
print("-" * 55)

for num_layers in [2, 4, 8, 16]:
    model_std = DeepGCN(dataset.num_features, 64, dataset.num_classes, num_layers).to(device)
    model_std.eval()
    with torch.no_grad():
        x = data.x
        for conv in model_std.convs[:-1]:
            x = F.relu(conv(x, data.edge_index))
        std = x.std().item()
    print(f"Layers / 層数={num_layers:2d} | Std / 標準偏差={std:.6f}")
```

**Expected output / 期待される出力:**

```
Layers=  2 | Std=0.412000
Layers=  4 | Std=0.089000
Layers=  8 | Std=0.003000
Layers= 16 | Std=0.000021
```

> Std approaching 0 means all node features are nearly identical — the model loses the ability to distinguish nodes. This is the quantitative expression of "mixing paint."  
> 標準偏差が0に近づくと、全ノードの特徴がほぼ同一になる——モデルはノードを区別する能力を失う。「絵の具を混ぜる」の定量的な表れ。

Over-smoothing is not a bug — it is a fundamental limitation of message passing. The tension between deeper networks (larger receptive field) and this limitation is an active GNN research direction. GAT in Lab 4 partially mitigates it; the fundamental solution involves residual connections (which appear in Lab 5's Transformer).  
Over-smoothing はバグではなく、メッセージパッシングの本質的な限界である。深いネットワーク（より大きな受容野）とこの限界の間の矛盾は活発な GNN 研究の方向性。Lab 4 の GAT は部分的に緩和し、根本的な解法は残差接続（Lab 5 の Transformer に登場）。

---

## 9. Controlled Experiment: Removing Dropout / 対照実験：Dropout の除去

Reinitialize the model with `dropout=0.0` and run the same training loop. Compare W&B curves.  
`dropout=0.0` でモデルを再初期化し、同じ訓練ループを実行する。W&B 曲線を比較する。

```python
wandb.init(
    project="gnn-labs",
    name="gcn-cora-no-dropout",
    config={
        "model": "GCN", "dataset": "Cora",
        "hidden_channels": 64, "dropout": 0.0,
        "lr": 0.01, "weight_decay": 5e-4, "epochs": 200,
    }
)

model_no_dropout = GCN(
    in_channels=dataset.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    dropout=0.0
).to(device)
```

**Expected observation / 期待される観察:**
- Train accuracy reaches 100% faster / 訓練精度がより速く100%に達する
- Val accuracy lower — more overfitting / val 精度が低い——より過学習
- Larger train/val gap / train/val ギャップが大きい

---

## 10. Self-Check Before Lab 4 / Lab 4 前の自己確認

No submission needed. If you can answer these, you are ready for Lab 4 (GAT).  
提出不要。答えられれば Lab 4（GAT）への準備ができている。

**What is the relationship between GCNConv and `A_norm @ X @ W` from Lab 2?**  
GCNConv と Lab 2 の `A_norm @ X @ W` の関係は何か？

**With only 140 training nodes, why can GCN classify all 2708 nodes?**  
訓練ノードが140個しかないのに、GCN はなぜ2708個全ノードを分類できるか？

**GCN treats all neighbors equally — what specific problem does this cause?**  
GCN はすべての隣接ノードを同等に扱う——具体的にどんな問題が生じるか？

**After removing Dropout, the train/val gap widened. Is this the same type of problem as removing BatchNorm in Lab 1?**  
Dropout を除去すると train/val ギャップが広がった。これは Lab 1 で BatchNorm を除去した場合と本質的に同じ種類の問題か？

---

## Summary / まとめ

| Concept / 概念 | Key Takeaway / 重要な要点 |
|---------------|--------------------------|
| GCNConv | `A_norm @ X @ W` with learnable W / 学習可能な W を持つ `A_norm @ X @ W` |
| Semi-supervised / 半教師あり | 140 labeled nodes; message passing leverages full graph / 140ラベル付きノード；メッセージパッシングがグラフ全体を活用 |
| Mask mechanism / マスク機構 | Full graph forward; masks select nodes for loss / グラフ全体を forward；マスクで loss に寄与するノードを選択 |
| Dropout | Prevents overfitting; disabled during evaluation / 過学習防止；評価中は無効化 |
| Weight decay | L2 regularization / L2 正則化 |
| GCN limitation / GCN の限界 | Equal weight for all neighbors → GAT / 全隣接ノードに等しい重み → GAT |
| Over-smoothing | More layers → features converge → accuracy drops / 層が増える → 特徴が収束 → 精度低下 |

---

*goose@NGC Lab.*
