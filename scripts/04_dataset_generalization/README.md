# WikiText-103 GPT-2 Data

This directory prepares WikiText-103 raw text with the GPT-2 tokenizer and writes
nanoGPT-compatible `uint16` binaries:

- `train.bin`
- `val.bin`
- `meta.pkl`
- `prepare_summary.json`

Smoke test:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_smoke \
  --max-train-tokens 10000 \
  --max-val-tokens 2000 \
  --force
```

Publication dataset-generalization subset:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --force
```

If `huggingface.co` is unstable, use a mirror endpoint:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --hf-endpoint https://hf-mirror.com \
  --load-retries 12 \
  --retry-sleep 10 \
  --force
```

The default Hugging Face dataset id is `wikitext` with config
`wikitext-103-raw-v1`.

If streaming repeatedly fails, download/cache the split first:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --hf-endpoint https://hf-mirror.com \
  --no-streaming \
  --cache-dir data/.hf_cache \
  --load-retries 12 \
  --retry-sleep 10 \
  --force
```

If Hugging Face metadata access is unavailable, bypass `datasets` and use the
direct Parquet files shown in the Hugging Face repo:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --source hf-parquet \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --cache-dir data/.hf_cache \
  --load-retries 12 \
  --retry-sleep 10 \
  --force
```

If direct Hugging Face Parquet download is also unavailable, bypass `datasets` and use the
official WikiText raw zip:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --source raw-zip \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --cache-dir data/.hf_cache \
  --load-retries 12 \
  --retry-sleep 10 \
  --force
```

If you manually downloaded `wikitext-103-raw-v1.zip`, pass it directly:

```bash
python scripts/04_dataset_generalization/prepare_wikitext103_gpt2.py \
  --source raw-zip \
  --zip-path data/wikitext-103-raw-v1.zip \
  --output-dir backends/nanogpt/data/wikitext103_gpt2_50m \
  --max-train-tokens 50000000 \
  --max-val-tokens 1000000 \
  --force
```
