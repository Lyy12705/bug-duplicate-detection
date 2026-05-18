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

若要評估 threshold 的 false positive，也需要另外抓一批「已知不是 duplicate」的 ticket。例如抓 Firefox 中 resolution 不是 `DUPLICATE` 的 resolved tickets：

```bash
python3 scripts/fetch_bugzilla.py \
  --base-url https://bugzilla.mozilla.org/rest \
  --product Firefox \
  --status RESOLVED \
  --resolution FIXED \
  --resolution WORKSFORME \
  --resolution INVALID \
  --limit 1000 \
  --with-comments \
  --output data/mozilla_firefox_non_duplicates.csv
```

接著把 duplicate 與 non-duplicate 合成 threshold / confusion matrix 專用資料集：

```bash
python3 scripts/build_duplicate_decision_dataset.py \
  --duplicates data/mozilla_firefox_duplicates.csv \
  --non-duplicates data/mozilla_firefox_non_duplicates.csv \
  --negative-ratio 1.0 \
  --output data/mozilla_firefox_decision_eval.csv
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

## 判斷新 Ticket 是否為 Duplicate

文獻方法的核心輸出是「依相似度排序的歷史 ticket 清單」。系統若要進一步自動標記 `Duplicate`，需要在驗證資料上選一個 similarity threshold，再用第一名候選 ticket 的分數做二元判斷。

先用 train/validation split 自動找 threshold：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets examples/sample_tickets.csv \
  --method tfidf \
  --combine max
```

`tune-threshold` 會輸出 query-level confusion matrix。這裡的 `false_positives` 就是「validation 中沒有已知 duplicate 的 ticket，卻因為最高相似度超過 threshold 而被判定為 duplicate」的數量：

```text
matrix_actual_by_predicted
actual,predicted_non_duplicate,predicted_duplicate
non_duplicate,true_negatives,false_positives
duplicate,false_negatives,true_positives
```

如果輸出 `warning=no_negative_queries`，代表這次 validation split 裡沒有「已知非 duplicate」的 query，因此無法估計 false positive。要評估「不是重複卻被判成重複」，資料集需要混入 non-duplicate tickets。

混入 non-duplicate 後，用 decision dataset 重新校準 threshold：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --method tfidf \
  --combine weighted:0.4,0.6
```

若目標是降低「不是重複卻被判成重複」的 false positives，可以提高 precision 約束來選比較保守的 threshold。實務上建議先用 `0.90`，不要一開始就拉到 `0.95`，否則 recall 會掉太多：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --method tfidf \
  --combine weighted:0.4,0.6 \
  --min-precision 0.90 \
  --output-json reports/tfidf_decision_threshold_p90.json \
  --output-decisions-csv reports/tfidf_decision_threshold_p90_decisions.csv
```

SBERT 版本會先依文獻方式用 train split 建立 title/content triplets 並 fine-tune：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method sbert \
  --combine max \
  --epochs 1 \
  --max-triplets 2000
```

得到 threshold 後，就可以對一張歷史 ticket 或新進 JSON ticket 做判斷：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli detect-duplicate \
  --tickets examples/sample_tickets.csv \
  --query-ticket-id T-100 \
  --method tfidf \
  --threshold 0.10 \
  --top-k 5
```

上面的 `0.10` 只是 sample dataset 的示範值；實驗時請改成 `tune-threshold` 輸出的 `threshold`。

若使用已訓練好的 SBERT 模型：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli detect-duplicate \
  --tickets data/mozilla_firefox_sample.csv \
  --query-json data/new_ticket.json \
  --method sbert \
  --model-dir models/duplicate_sbert \
  --threshold 0.70 \
  --top-k 10
```

輸出的 `is_duplicate=true/false` 是系統判斷；`best_candidate_id` 是最可能重複的歷史 ticket；後面的排名清單則保留給工程師複核。

評估目前模型表現：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli evaluate-sbert \
  --tickets data/mozilla_firefox_sample.csv \
  --model-dir models/duplicate_sbert \
  --top-k 10
```

輸出中的 `MAP` 是文獻主要使用的排名品質指標；`top_1_accuracy` 表示第一名候選是否就是 duplicate；`top_10_hit_rate` 表示前 10 名內是否至少有一筆正確 duplicate。

## 一次跑完整比較

`scripts/run_duplicate_experiments.py` 會一次跑：

- TF-IDF baseline
- `max`、`title75`、`content75`、`mean`、`weighted:0.4,0.6`、`weighted:0.6,0.4` 六種 title/content similarity 組合
- SBERT `epochs=2`
- SBERT `all-MiniLM-L6-v2` 與 `all-mpnet-base-v2` base model 比較

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --sbert-epochs 2 \
  --max-triplets 2000 \
  --output-csv reports/duplicate_experiment_results.csv \
  --output-md reports/duplicate_experiment_results.md
```

如果 base model 已經下載過，可加上 `--local-files-only` 避免每次連 Hugging Face 檢查。若 `all-mpnet-base-v2` 尚未下載，第一次請不要加 `--local-files-only`。

如果只想先看 TF-IDF 和六種 combine 的快速 baseline：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-sbert \
  --output-csv reports/tfidf_combine_results.csv \
  --output-md reports/tfidf_combine_results.md
```

如果只是要查看已跑完實驗的最後指標，不需要重新跑完整實驗，可直接讀既有 CSV：

```bash
python3 scripts/show_duplicate_results.py \
  --results-csv reports/tfidf_combine_results.csv
```

預設會顯示 `fold=mean` 中目前最佳的一筆 ranking 結果，包含 MAP、Top-1、Top-k hit rate、Recall、MRR。若沒有指定 `--results-csv`，腳本會自動掃描 `reports/*.csv`，並依照目前排序指標選出分數最好的實驗結果 CSV：

```bash
python3 scripts/show_duplicate_results.py
```

若想改成看最新產生的 CSV，或回到舊的預設檔 `reports/duplicate_experiment_results.csv`：

```bash
python3 scripts/show_duplicate_results.py --select latest
python3 scripts/show_duplicate_results.py --select default
```

輸出會用 terminal-friendly 摘要顯示最佳 ranking 結果，不再列出完整 ranking 表格，也不顯示 `Precision@k`，避免和 threshold 的 decision precision 混淆。若要改用其他指標選最佳結果：

```bash
python3 scripts/show_duplicate_results.py --sort-by topk
```

也可以依 recall 排序：

```bash
python3 scripts/show_duplicate_results.py --sort-by recall
```

`show_duplicate_results.py` 會自動顯示最佳 ranking 結果，並在 `reports/` 找到 threshold JSON 時一起顯示 confusion matrix。若要產生或更新 confusion matrix，先在 `tune-threshold` 加上 `--output-json`：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --method tfidf \
  --combine weighted:0.4,0.6 \
  --min-precision 0.90 \
  --output-json reports/tfidf_decision_threshold_p90.json \
  --output-decisions-csv reports/tfidf_decision_threshold_p90_decisions.csv
```

再用 `show_duplicate_results.py` 一起顯示最佳 ranking 結果和混淆矩陣：

```bash
python3 scripts/show_duplicate_results.py
```

若要指定某一個 threshold JSON：

```bash
python3 scripts/show_duplicate_results.py \
  --decision-json reports/tfidf_decision_threshold_p90.json
```

若只想看最佳 ranking 結果、不顯示混淆矩陣：

```bash
python3 scripts/show_duplicate_results.py --no-show-decision-matrix
```

若要分析 false positives：

```bash
python3 scripts/analyze_false_positives.py \
  --decisions-csv reports/tfidf_decision_threshold_p90_decisions.csv \
  --output-csv reports/tfidf_false_positives_p90.csv \
  --summary-md reports/tfidf_false_positive_summary_p90.md
```

若目前最大的問題是漏抓 duplicate，可分析 false negatives。這會區分「Top-1 其實是正確 duplicate 但分數低於 threshold」與「Top-1 本身就排錯」：

```bash
python3 scripts/analyze_false_negatives.py \
  --decisions-csv reports/sbert_decision_threshold_p90_decisions.csv \
  --output-csv reports/sbert_false_negatives_p90.csv \
  --summary-md reports/sbert_false_negative_summary_p90.md
```

若不想只固定 `min_precision=0.90`，可直接用既有 decisions CSV 重新掃不同 threshold，不需要重跑 SBERT：

```bash
python3 scripts/sweep_decision_thresholds.py \
  --decisions-csv reports/sbert_decision_threshold_p90_decisions.csv \
  --scenario best_f1:-:- \
  --scenario precision80_recall30:0.80:0.30 \
  --scenario precision70_recall50:0.70:0.50 \
  --output-csv reports/sbert_threshold_sweep.csv
```

若已經知道希望 recall 至少達到某個值，也可以在 `tune-threshold` 直接加上 `--min-recall`，例如 `--min-precision 0.70 --min-recall 0.50`。

若要把 duplicate / non-duplicate 判斷從單一 threshold 改成二元分類器，可用 decisions CSV 訓練 logistic regression。特徵包含 Top-1 score、title/content score、Top-1 與 Top-2 的 margin，以及 `component`、`product`、`severity`、`priority` 是否一致；若有 TF-IDF 或 BM25 的 decisions CSV，也可以用 `--extra-decisions-csv tfidf=...` 或 `--extra-decisions-csv bm25=...` 合併額外分數：

```bash
python3 scripts/train_duplicate_decision_classifier.py \
  --decisions-csv reports/sbert_decision_threshold_p90_decisions.csv \
  --extra-decisions-csv tfidf=reports/tfidf_decision_threshold_p90_decisions.csv \
  --output-json reports/decision_classifier_metrics.json \
  --output-model models/duplicate_decision_classifier.joblib \
  --output-predictions-csv reports/decision_classifier_predictions.csv
```

若要快速試 SBERT 流程，可先降低 triplets：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-tfidf \
  --sbert-base-models sentence-transformers/all-MiniLM-L6-v2 \
  --sbert-epochs 2 \
  --max-triplets 500 \
  --local-files-only \
  --output-csv reports/sbert_minilm_e2_quick.csv \
  --output-md reports/sbert_minilm_e2_quick.md
```

若要測試 hard negative mining，讓模型學習較難分辨的負樣本：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-tfidf \
  --sbert-base-models sentence-transformers/all-MiniLM-L6-v2 \
  --sbert-epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --output-csv reports/sbert_hard_negative.csv \
  --output-md reports/sbert_hard_negative.md
```

若要優先改善 Top-1，可在同一次實驗中加入 metadata reranking。這不會多訓練 SBERT，只是在每個 fold 取得相似度矩陣後，額外比較幾組 `component` / `product` 欄位加權：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-tfidf \
  --sbert-base-models sentence-transformers/all-MiniLM-L6-v2 \
  --sbert-epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --rerank-grid \
  --output-csv reports/sbert_hard_negative_rerank.csv \
  --output-md reports/sbert_hard_negative_rerank.md
```

若已經知道要用哪組 rerank 權重，也可以只跑指定設定：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-tfidf \
  --sbert-base-models sentence-transformers/all-MiniLM-L6-v2 \
  --sbert-epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --rerank \
  --rerank-fields component:0.02 \
  --output-csv reports/sbert_hard_negative_rerank_component002.csv \
  --output-md reports/sbert_hard_negative_rerank_component002.md
```

threshold 也可以和 rerank 一起校準，讓 `detect-duplicate` 使用 rerank 後的分數做 duplicate / non-duplicate 判斷：

```bash
PYTHONPATH=src python3 -m duplicate_ticket_detection.cli tune-threshold \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --method sbert \
  --combine weighted:0.6,0.4 \
  --epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --rerank \
  --rerank-fields component:0.02 \
  --min-precision 0.90 \
  --output-json reports/sbert_decision_threshold_p90.json \
  --output-decisions-csv reports/sbert_decision_threshold_p90_decisions.csv
```

若要讓 decision threshold 跟目前最佳 Ranking 流程對齊，可改用 Top-1 mined hard negative 版本。下面這個流程會先挖錯誤 Top-1 當 hard negatives，重新訓練 SBERT，再用 `weighted:0.6,0.4 + rerank component:0.02` 調 threshold：

```bash
python3 scripts/tune_top1_mined_threshold.py \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --combine weighted:0.6,0.4 \
  --rerank-fields component:0.02 \
  --epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --min-precision 0.70 \
  --min-recall 0.50 \
  --output-json reports/top1_mined_decision_threshold_balanced.json \
  --output-decisions-csv reports/top1_mined_decision_threshold_balanced_decisions.csv
```

如果不想預先固定 precision / recall 約束，而是直接找 F1 最佳的 threshold，可以拿掉 `--min-precision` 和 `--min-recall`：

```bash
python3 scripts/tune_top1_mined_threshold.py \
  --tickets data/mozilla_firefox_decision_eval.csv \
  --combine weighted:0.6,0.4 \
  --rerank-fields component:0.02 \
  --epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --output-json reports/top1_mined_decision_threshold_best_f1.json \
  --output-decisions-csv reports/top1_mined_decision_threshold_best_f1_decisions.csv
```

若要測試更多 metadata 特徵，可把 rerank 欄位改成：

```bash
--rerank-fields component:0.02 product:0.01 severity:0.005 priority:0.005
```

目前較建議把系統定位為 duplicate candidate recommendation：先顯示 Top-k 疑似重複 ticket 給工程師複核。只有在 classifier 或 threshold 很有信心時，才自動標記 duplicate；中間分數區間保留人工確認，可以避免 false negative 過高。

## 半自動 duplicate 複核流程

目前實驗顯示 Ranking 指標比單一 threshold decision 更穩定，因此系統先採用半自動化流程：

1. 模型產生 Top-k duplicate candidates。
2. 系統輸出每個候選的 similarity score、title/content score、metadata 是否一致，以及 TF-IDF 是否也把它排在前面。
3. 工程師查看候選清單後決定是否標記 duplicate。
4. 工程師確認過的錯誤案例可回收成 hard negatives，持續改善 Top-1。

產生工程師複核 CSV：

```bash
python3 scripts/export_duplicate_review_queue.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method tfidf \
  --combine weighted:0.6,0.4 \
  --top-k 10 \
  --rerank-fields component:0.02 \
  --output-csv reports/duplicate_review_queue.csv \
  --summary-md reports/duplicate_review_queue_summary.md
```

`export_duplicate_review_queue.py` 預設會啟用 stable rerank：主模型排序仍是主要訊號，但會混合 TF-IDF 排名一致性、metadata 一致性與 Top-1 margin，讓工程師看到的候選順序更穩定。若要調整穩定排序權重：

```bash
python3 scripts/export_duplicate_review_queue.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method sbert \
  --model-dir models/duplicate_sbert \
  --combine weighted:0.6,0.4 \
  --top-k 10 \
  --rerank-fields component:0.02 \
  --stable-tfidf-weight 0.30 \
  --stable-metadata-weight 0.08 \
  --stable-agreement-weight 0.08
```

若要回到單一模型排序，可加上 `--no-stable-rerank`。

若已經有訓練好的 SBERT 模型，可改用：

```bash
python3 scripts/export_duplicate_review_queue.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method sbert \
  --model-dir models/duplicate_sbert \
  --combine weighted:0.6,0.4 \
  --top-k 10 \
  --rerank-fields component:0.02 \
  --output-csv reports/duplicate_review_queue.csv \
  --summary-md reports/duplicate_review_queue_summary.md
```

若要針對一張新 ticket 輸出候選清單：

```bash
python3 scripts/export_duplicate_review_queue.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method sbert \
  --model-dir models/duplicate_sbert \
  --query-json data/new_ticket.json \
  --top-k 10 \
  --rerank-fields component:0.02
```

輸出的 CSV 不會直接說「一定重複」，而是提供 `review_priority`、`review_confidence`、`rank_agreement`、`stable_score`、`model_score`、`tfidf_score`、`component_match`、`product_match` 等欄位，讓工程師做最後判斷。`review_confidence=strong` 表示多個訊號一致；`needs_review` 表示候選仍值得看，但排序信心較低。

若要針對 Top-1 錯誤做 error analysis：

```bash
python3 scripts/analyze_duplicate_errors.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --method tfidf \
  --combine mean \
  --output-csv reports/top1_error_analysis.csv \
  --summary-md reports/top1_error_summary.md
```

若要進一步改善 Top-1，可跑專用實驗：先用第一輪 SBERT 在 train fold 找出「錯排第 1 名」的候選，再把這些錯誤 Top-1 當成額外 hard negatives 重新訓練，最後一次測多組 metadata rerank 權重：

```bash
python3 scripts/run_top1_improvement_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --epochs 1 \
  --max-triplets 2000 \
  --negative-strategy hard \
  --negatives-per-anchor 3 \
  --output-csv reports/top1_improvement_experiment_results.csv \
  --output-md reports/top1_improvement_experiment_results.md
```

這個腳本會輸出：

- `reports/top1_improvement_experiment_results.csv`：各組 combine/rerank 的 MAP、Top-1、Top-k、Recall、MRR。
- `reports/top1_mined_hard_negatives.csv`：每個 fold 從 train 資料挖出的錯誤 Top-1 hard negatives。
- `reports/top1_improvement_error_analysis.csv`：重新訓練與 rerank 後仍排錯第 1 名的案例，方便繼續分析錯誤是否集中在特定 product/component。

若要檢查 duplicate master ticket 是否缺失：

```bash
python3 scripts/validate_duplicate_dataset.py \
  --tickets data/mozilla_firefox_duplicates.csv
```

若 SBERT 在 Mac 上跑到一半因為 PyTorch/MPS 或記憶體問題中斷，可先改用較保守設定：

```bash
python3 scripts/run_duplicate_experiments.py \
  --tickets data/mozilla_firefox_duplicates.csv \
  --folds 5 \
  --top-k 10 \
  --skip-tfidf \
  --sbert-base-models sentence-transformers/all-MiniLM-L6-v2 \
  --sbert-epochs 1 \
  --max-triplets 500 \
  --batch-size 4 \
  --device cpu \
  --continue-on-error \
  --output-csv reports/sbert_minilm_stable.csv \
  --output-md reports/sbert_minilm_stable.md
```

`run_duplicate_experiments.py` 會在 TF-IDF 完成後，以及每個 SBERT fold 完成後自動寫入 checkpoint。若後面某個 fold 失敗，仍可用 `scripts/show_duplicate_results.py --results-csv <結果檔>` 查看已完成的結果。
