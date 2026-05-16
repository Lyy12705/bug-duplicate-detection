# Duplicate Ticket Detection

這個專案先實作大專生計畫中的第一階段：判斷新進 ticket 是否可能是歷史 ticket 的重複回報。

目前可直接離線執行的是 `TF-IDF + cosine ranking` baseline；同時也保留文獻 Isotani et al. (2023) 的主方法接口：title/content 分開向量化、用 duplicate ticket 關係建立 triplet、以 SBERT triplet loss 微調，最後用 cosine similarity 排序並用 MAP 評估。

## 文獻方法對應

文獻流程可以落成下面幾個模組：

1. 取出 ticket 的 `title` 與 `content`。
2. 分別建立 title model 與 content model。
3. 從 duplicate 關係建立 `(anchor, positive, negative)` triplets。
4. 以 cosine distance 的 triplet loss 微調 SBERT。
5. 對新 ticket 與歷史 tickets 計算 title similarity / content similarity。
6. 用 `max(title_similarity, content_similarity)` 或加權組合得到 report similarity。
7. 依 similarity 排序，並用 MAP 評估 duplicate ranking 品質。

## 資料格式

建議 CSV 或 JSONL 至少包含：

| 欄位 | 說明 |
|---|---|
| `ticket_id` | ticket/bug id |
| `title` | 標題或 summary |
| `description` | 主要描述、重現步驟、log 等內容 |
| `duplicate_of` | 若此 ticket 是 duplicate，填 master/original ticket id |
| `duplicate_group` | 可選；同一組 duplicate ticket 填相同 group id |

如果資料來自 Bugzilla，也支援常見欄位別名，例如 `id`、`bug_id`、`summary`、`dupe_of`。

## 快速開始

在尚未安裝套件成 package 前，可用 `PYTHONPATH=src` 執行：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli evaluate \
  --tickets examples/sample_tickets.csv \
  --method tfidf \
  --combine max
```

查看某張 ticket 的候選重複清單：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli rank \
  --tickets examples/sample_tickets.csv \
  --query-ticket-id T-100 \
  --top-k 5
```

建立 SBERT fine-tuning 用的 triplets：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli build-triplets \
  --tickets examples/sample_tickets.csv \
  --field content \
  --output data/triplets_content.csv
```

## 下載 Bugzilla 資料

可用 `scripts/fetch_bugzilla.py` 從 Bugzilla REST API 匯出 CSV。範例：

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugzilla.mozilla.org/rest \
  --product Firefox \
  --limit 200 \
  --with-comments \
  --output data/mozilla_firefox_sample.csv
```

若要使用 Eclipse/Mozilla 類型資料集，重點是保留 duplicate 關係欄位。Bugzilla 中通常是 `dupe_of`，匯出後會轉成此專案使用的 `duplicate_of`。

建議先抓 `resolution=DUPLICATE`，並加上 `--include-duplicate-masters`，確保每筆 duplicate ticket 的 master/original ticket 也在資料集中：

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugzilla.mozilla.org/rest \
  --product Firefox \
  --resolution DUPLICATE \
  --limit 500 \
  --with-comments \
  --include-duplicate-masters \
  --output data/mozilla_firefox_duplicates.csv
```

Eclipse 舊 Bugzilla 也可用相同流程：

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugs.eclipse.org/bugs/rest \
  --product Platform \
  --resolution DUPLICATE \
  --limit 500 \
  --with-comments \
  --include-duplicate-masters \
  --output data/eclipse_platform_duplicates.csv
```

## Train/Test 與 5-Fold 評估

產生 train/test CSV：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli split-data \
  --tickets data/mozilla_firefox_duplicates.csv \
  --test-size 0.2 \
  --train-output data/splits/mozilla_train.csv \
  --test-output data/splits/mozilla_test.csv
```

TF-IDF baseline 的 train/test 評估：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli evaluate-split \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method tfidf \
  --test-size 0.2 \
  --top-k 10
```

TF-IDF baseline 的 5-fold cross-validation：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli cross-validate \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method tfidf \
  --folds 5 \
  --top-k 10
```

SBERT 的 5-fold cross-validation：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli cross-validate \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method sbert \
  --folds 5 \
  --epochs 1 \
  --max-triplets 2000 \
  --top-k 10
```

輸出中的 `MAP` 對應文獻的 ranking metric；`top_1_accuracy` 表示第一名是否為正確 duplicate；`top_10_hit_rate` 表示前 10 名內是否至少出現一筆正確 duplicate；`MRR` 表示第一個正確 duplicate 排名越前越好。

## SBERT fine-tuning

文獻主方法需要額外套件：

```bash
python3 -m pip install -e ".[sbert]"
```

安裝後可在程式中使用 `SbertDuplicateDetector`。由於 SBERT 模型下載與訓練需要網路和較多時間，這個 repo 先把可測試的資料流程、triplet 建立與 baseline 完整放好。

安裝 optional dependency 後也可以直接訓練：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli train-sbert \
  --tickets data/mozilla_firefox_sample.csv \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --epochs 1 \
  --output-dir models/duplicate_sbert
```

用訓練好的模型排序候選重複 ticket：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli rank-sbert \
  --tickets data/mozilla_firefox_sample.csv \
  --model-dir models/duplicate_sbert \
  --query-ticket-id 123456 \
  --top-k 10
```

評估目前模型表現：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli evaluate-sbert \
  --tickets data/mozilla_firefox_sample.csv \
  --model-dir models/duplicate_sbert \
  --top-k 10
```

輸出中的 `MAP` 是文獻主要使用的排名品質指標；`top_1_accuracy` 表示第一名候選是否就是 duplicate；`top_10_hit_rate` 表示前 10 名內是否至少有一筆正確 duplicate。
