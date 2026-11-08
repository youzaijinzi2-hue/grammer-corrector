# grammer-corrector

自作の Encoder–Decoder Transformer を使った英文校正の実験用 CLI です。現在は、`He go outside.` を `He goes outside.` に直すような、限定された三人称単数現在形の誤りを扱います。モデルは合成データで学習するため、既知の単語だけで書かれた文でも、文型によっては誤修正することがあります。

## セットアップ

Python 3.12 以上と uv が必要です。`pyproject.toml` は `transformer-from-scratch` をローカルの編集可能な依存関係として参照します。両方のリポジトリを同じ親フォルダに置いてください。

```text
親フォルダ/
├── transformer-from-scratch/
└── grammer-corrector/
```

以降のコマンドは `grammer-corrector` のルートで実行します。

```sh
uv sync --locked
```

## 学習・評価・校正

```sh
data=src/grammer_corrector/resources/datasets/agreement
uv run grammer-corrector train --data "$data/train.jsonl" --output runs/model.pt --include-identity
uv run grammer-corrector evaluate --data "$data/eval.jsonl" --checkpoint runs/model.pt
uv run python -m grammer_corrector.cli runs/model.pt
```

`train` は指定した学習ファイルの全件を使います。`--include-identity` を付けると、正しい文をそのまま返す例も追加します。学習の既定値は100エポック、乱数シード42です。入力文と正解文はそれぞれ開始・終了マーカーを含めて16トークンまでです。

`evaluate` は保存済みモデルを読み込み、文単位の完全一致、誤文の修正件数、正文の保持件数を表示します。最後のコマンドは同じチェックポイントで対話型 CLI を起動します。

## データ量の比較実験

```sh
uv run python -m grammer_corrector.data_experiment --save-dir runs/agreement
```

修正ペアを12・50・100組使う3条件を比較します。各条件では、同数の正文保持ペア（正しい文をそのまま返す例）を加え、CPU で100エポック学習します。乱数シードは42です。評価結果には、誤文を正しく修正した件数、正文を保持した件数、語彙外のトークンを含む入力件数が表示されます。

`--save-dir` を指定すると、モデルの重み・設定・語彙表を含むチェックポイントが条件ごとに保存されます。保存先は自動で作成され、同名のファイルがあれば上書きされます。省略時は学習と評価を行い、チェックポイントの保存を省略します。リポジトリ直下の `runs/` は Git の管理対象から外しています。

## 対話型 CLI

保存した50組のモデルを指定して起動します。

```sh
uv run python -m grammer_corrector.cli \
  runs/agreement/50-correction-50-identity-seed42-epochs100.pt
```

```text
> He go outside.
He goes outside.
> He goes outside.
He goes outside.
> :quit
```

英文を1行ずつ入力します。語彙外のトークンを含む文、モデルの入力上限を超えた文、校正結果の生成が完了しなかった文では、理由を表示して原文を返します。`:quit` または Ctrl+D で終了できます。

## データ構成

| ファイル | 内容 |
|---|---|
| [agreement.json](src/grammer_corrector/resources/agreement.json) | 主語・動詞の原形と三単現形・後ろに続く語句など、文を作る素材 |
| [train.jsonl](src/grammer_corrector/resources/datasets/agreement/train.jsonl) | 学習候補の修正ペア125組。実験はファイルの先頭から指定件数を使用 |
| [eval.jsonl](src/grammer_corrector/resources/datasets/agreement/eval.jsonl) | 学習から取り分けた評価用の12組。先頭4組が当初の評価例、残り8組が追加分 |

JSONL の各行は `source`（入力文）と `target`（期待する出力文）のペアです。正文保持ペアは、選んだ学習ペアの `target` を入力と出力の両方に使って、実行時に追加します。実験は固定の JSONL を読み込むため、素材の JSON を編集した場合は、学習に反映したいペアを別途 JSONL に反映します。

学習では入力文を Encoder に渡し、正解文を1トークンずらして Decoder 入力と予測の正解にします。評価と CLI では、Decoder が文頭トークンから1トークンずつ生成します。評価ファイルの正解文は、生成後の完全一致の判定に使います。

## テスト

```sh
uv run python -m pytest -q
```
