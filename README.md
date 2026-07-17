# Advanced_Tutorial_1

ARIM-ACADEMY-2026 基礎編：Scikit-learnを使った機械学習

Pythonの基礎的な文法を理解している方を対象に、pandas・matplotlib・seaborn・scikit-learnを使った探索的データ分析（EDA）・回帰・分類・次元削減・クラスタリングの一連の流れを、3種類のデータセット（住宅価格・大気腐食・茶の元素分析）を通じて学ぶ教材です。

> **重要：** このリポジトリ内のノートブックの一部（`Scikit-learn-1_EDA.ipynb`・`Scikit-learn-2_ML.ipynb`）は、実在しない教材用の架空データ（Japan Housing）を扱っています。詳細と注意点は各ノートブック冒頭の「データセットの背景と倫理的配慮」を参照してください。

---

## 目次

| No. | ノートブック | Colabで開く | データセット | 内容 |
| --- | --- | --- | --- | --- |
| 1 | [`Scikit-learn-1_EDA.ipynb`](./Scikit-learn-1_EDA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-1_EDA.ipynb) | Japan Housing（架空・教材用） | 探索的データ分析（EDA）：概要統計、箱ひげ図、ペアプロット、相関係数、ヒートマップ、単回帰の基礎 |
| 2 | [`Scikit-learn-2_ML.ipynb`](./Scikit-learn-2_ML.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-2_ML.ipynb) | Japan Housing（架空・教材用） | 回帰モデルの構築と比較：線形重回帰・Lasso回帰・ランダムフォレスト、train/test分割、過学習、モデル比較 |
| 3 | [`Scikit-learn-3_EDA.ipynb`](./Scikit-learn-3_EDA.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-3_EDA.ipynb) | 大気腐食データセット（実データ） | 探索的データ分析：住宅価格編と同じ手順を、NIMSの実際の研究データに適用 |
| 4 | [`Scikit-learn-4_ML.ipynb`](./Scikit-learn-4_ML.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-4_ML.ipynb) | 大気腐食データセット（実データ） | 回帰モデルの構築と比較、データリーケージへの注意、k-foldクロスバリデーション、地点外挿の限界 |
| 5 | [`Scikit-learn-5_Classification.ipynb`](./Scikit-learn-5_Classification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-5_Classification.ipynb) | 茶の元素分析データセット（実データ） | 教師あり学習による分類：決定木・ランダムフォレスト・SVM、混同行列、適合率・再現率 |
| 6 | [`Scikit-learn-6_PCA_Clustering.ipynb`](./Scikit-learn-6_PCA_Clustering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_1/blob/main/Scikit-learn-6_PCA_Clustering.ipynb) | 茶の元素分析データセット（実データ） | 教師なし学習：主成分分析（PCA）・t-SNE・UMAPによる次元削減、階層クラスタリング・k-means |

ノートブックは1→6の順に読み進めることを想定しています。1・2（住宅価格）と3・4（大気腐食）は同じコード構成をあえて踏襲しており、「同じ手順が題材を変えても通用すること」を確認できるように作られています。5・6は分類・教師なし学習という新しいテーマを扱います。

---

## 対象読者・前提知識

- Python基礎文法（変数、リスト、for文、関数の呼び出し）を理解している方
- pandas・numpy・matplotlib・seaborn・scikit-learnに初めて触れる方
- 統計学の予備知識（相関係数や回帰分析の理論的背景）は前提としません。各ノートブック内で都度説明します

## 動作環境

- Python 3.10以降
- pandas 2.x / numpy 1.26以降 / matplotlib 3.8以降 / seaborn 0.12以降
- scikit-learn 1.2以降（`Pipeline`・`StandardScaler`・`train_test_split`・`GroupKFold`などを使用）
- `matplotlib_fontja`（日本語フォント表示用）
- `umap-learn`（ノートブック6のみ、UMAPで使用。ノートブック内で自動インストールされます）

いずれもGoogle Colabの標準環境（2026年時点）であれば満たされます。

## 使い方（Google Colab）

各ノートブックの冒頭にある「教材への接続」セルを実行すると、このリポジトリを自動的にクローンし、`data/`フォルダ内のデータセットを読み込む準備が整います。

```python
!git clone https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_1.git
%cd Advanced_Tutorial_1
```

ローカル環境で実行する場合は、このセルは不要です。リポジトリを手元にクローンし、ノートブックと同じ階層に`data/`フォルダがあることを確認してから実行してください。

```bash
git clone https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_1.git
cd Advanced_Tutorial_1
pip install -r requirements.txt  # 未整備の場合は上記「動作環境」を参照して個別にインストールしてください
jupyter lab
```

---

## データセットと出典

### Japan Housing（ノートブック1・2）
Boston Housingデータセットを日本の不動産情報に合わせて変数名・単位を翻案した、**実在しない教材用の架空データ**です。統計的な性質（分布・相関構造）は元データを再現していますが、実際の不動産価格の予測・投資判断・政策立案には使用しないでください。

Boston Housingの原変数の一部（人種構成に基づく`B`変数）は、科学的根拠のない人種主義的な仮定に基づいて作成されたことが問題視され、scikit-learn 1.0（2021年）で非推奨化、1.2（2022年）で完全に削除されています。本教材における取り扱いと経緯の詳細は、`Scikit-learn-1_EDA.ipynb`冒頭の「データセットの背景と倫理的配慮」を参照してください。

### 大気腐食データセット（ノートブック3・4）
物質・材料研究機構（NIMS）における実際の研究成果に基づくデータです。日本国内6地点の標準試験片による大気腐食量と、気象庁の気象観測データを組み合わせています。

> 松波 成行, 柳生 進二郎, 篠原 正, 片山 英樹, 須藤 仁, 服部 康男, 平口 博丸："海塩輸送シミュレーションと気象情報を用いた機械学習に基づく大気腐食量評価モデル開発と高精細腐食環境地図の作成", 土木学会論文集A1（構造・地震工学） Vol.75, p141-160 (2019). https://www.jstage.jst.go.jp/article/jscejseee/75/2/75_141/_article/-char/ja/

本講義用にアレンジされたデータであり、論文で使用されたデータセットそのものとは異なります。

### 茶の元素分析データセット（ノートブック5・6）
4種類の茶葉（ブラックセイロン、ブラックトルコ、グリーンセイロン、グリーントルコ）を対象に、ICP-OESで分析した9元素（Al, Ca, Cu, Fe, K, Mg, Mn, Na, Zn）の濃度データです。

> Durmus, Y., Atasoy, A.D. & Atasoy, A.F. Mathematical optimization of multilinear and artificial neural network regressions for mineral composition of different tea types infusions. *Sci Rep* 14, 18285 (2024). https://doi.org/10.1038/s41598-024-69149-1

本講義用に、上記論文の実験データをもとに整理したものです。

---

## ライセンス

各ノートブックのコード部分はMITライセンスで提供します。データセットのライセンス・利用条件は上記の出典元に従ってください。Japan Housingデータセットは架空データのため、この限りではありません。

## 更新履歴

- 各ノートブックの詳細な変更点は、ノートブック内の記述および本リポジトリのコミット履歴を参照してください。
