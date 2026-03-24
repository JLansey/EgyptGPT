# EgyptGPT outputs

This document describes the local output layout for the EgyptGPT workflow. These files are generated during experiments, kept out of git, and organized by pipeline stage.

## Pipeline

The working pipeline has three stages:

1. Generate raw Gardiner sign sequences.
2. Translate those sign sequences into English text.
3. Rate the resulting translations for `quality` and `interest`.

## Directory layout

All generated artifacts live under `out/`.

```text
out/
  01-generate/
    legacy-temperature-sweep/
  02-translate/
    legacy-temperature-sweep/
    current-best-model/
    archive/
  03-rate/
    legacy-temperature-sweep/
    current-best-model/
    archive/
```

## Stage 1: generate

`out/01-generate/legacy-temperature-sweep/`

- `generated_gardiner_0.7_full.txt`: larger raw generation dump for the older `0.7` run.
- `generated_gardiner_0.7_subset.txt`: smaller raw generation subset for the older `0.7` run.
- `generated_gardiner_1.0_subset.txt`: smaller raw generation subset for the older `1.0` run.

These are the stage-1 text outputs before translation.

## Stage 2: translate

`out/02-translate/legacy-temperature-sweep/`

- `translated_temperature_0.5.csv` through `translated_temperature_1.0.csv`: older translated samples grouped by decoding temperature.

This is the earlier baseline-style comparison set. It is useful for seeing how translation quality changed across temperatures before the newer larger run.

`out/02-translate/current-best-model/`

- `translated_current_best_full.csv`: the largest current translation export found in this workspace.
- `translated_current_best_first_8408.csv`: earlier partial export from the newer run.
- `translated_current_best_first_9101.csv`: later partial export from the newer run.

This folder is the place to look when you want the newer, better model outputs instead of the old temperature sweep.

`out/02-translate/archive/`

- `translated_current_best_first_5407.csv`
- `translated_current_best_first_5407_copy.csv`

These appear to be older scratch exports and one duplicate copy. They are kept for reference but are not the primary outputs.

## Stage 3: rate

`out/03-rate/legacy-temperature-sweep/`

- `scored_temperature_sweep.csv`: rated output for the smaller legacy comparison set.

`out/03-rate/current-best-model/`

- `batch_input_current_best.jsonl`: OpenAI batch request payload used to score the newer translation set.
- `scored_current_best.csv`: rated results for the newer translation set.

The row count of `batch_input_current_best.jsonl` matches `scored_current_best.csv`, so these two files belong together as the stage-3 evaluation artifacts for the current-best run.

`out/03-rate/archive/`

- `egypt_char_zip_snapshot.zip`: loose archive snapshot that was previously sitting in `data/`.

## Comparison guidance

If you want to show the difference between the older and newer model outputs, compare:

- old translations: `out/02-translate/legacy-temperature-sweep/`
- old scored subset: `out/03-rate/legacy-temperature-sweep/scored_temperature_sweep.csv`
- newer translations: `out/02-translate/current-best-model/translated_current_best_full.csv`
- newer scored set: `out/03-rate/current-best-model/scored_current_best.csv`

In other words: the legacy temperature sweep is the smaller earlier baseline, and the `current-best-model` folders are where the larger newer outputs live.

## Notebooks

Local notebooks now live in `notebooks/` instead of the repo root:

- `notebooks/running_EgyptGPT.ipynb`: end-to-end experimental notebook for generating and translating outputs.
- `notebooks/review_translations.ipynb`: manual review and scoring notebook.

See `notebooks/README.md` for notebook-specific notes.
