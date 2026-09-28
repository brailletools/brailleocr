# Labels

Human-verified Braille cell labels produced with `experiments/label_tool.py`.

## Layout

    labels/corrections/<image>.corrections.json   ← save downloads here
    labels/csv/<image>.csv                        ← generated, Angelina format

## Workflow

1. Generate a queue:            `python experiments/label_batch.py --source braille_natural --limit 212`
2. Correct pages in the browser, saving each download into `labels/corrections/`.
3. Merge into a dataset:        `python experiments/merge_corrections.py labels/corrections --out labels/csv`

`labels/csv/` is Angelina-format (`l;t;r;b;label_int`, fractional coordinates),
which `angelina_data.load_csv`, `prepare_yolo_dataset.py` and `extract_crops.py`
already read — no new loader needed.

## What a corrections file means

Ground truth = the page's predictions, with edits and moves applied, deletions
removed and additions included. **Anything the labeler did not touch is recorded
as confirmed correct**, so these files are only meaningful alongside the
`predictions` block each one carries (they are self-contained; the page that
produced them is not needed).

Cells left at `000000` are *unlabeled*, not empty, and `merge_corrections.py`
skips them rather than writing them as real labels.
