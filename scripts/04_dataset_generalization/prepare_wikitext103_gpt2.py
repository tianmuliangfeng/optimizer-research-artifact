"""
Prepare a WikiText-103 GPT-2-tokenized subset for nanoGPT experiments.

The script reads WikiText-103 raw from Hugging Face, encodes with GPT-2 BPE,
and writes uint16 train.bin/val.bin files compatible with train.py.
"""

import argparse
import json
import os
import pickle
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from array import array
from pathlib import Path


DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DEFAULT_MAX_TRAIN_TOKENS = 50_000_000
DEFAULT_MAX_VAL_TOKENS = 1_000_000
WRITE_CHUNK_TOKENS = 1_000_000
DEFAULT_RAW_ZIP_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
HF_PARQUET_BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
RAW_ZIP_ROOT = "wikitext-103-raw"
RAW_SPLIT_FILES = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "valid": "wiki.valid.raw",
    "val": "wiki.valid.raw",
    "test": "wiki.test.raw",
}
HF_PARQUET_SPLIT_FILES = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "valid": ["validation-00000-of-00001.parquet"],
    "val": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def default_output_dir():
    source_repo = os.environ.get("SELECTIVE_NEWTON_MUON_SOURCE_REPO")
    if source_repo:
        return Path(source_repo).expanduser().resolve() / "data" / "wikitext103_gpt2_50m"
    artifact_root = Path(__file__).resolve().parents[2]
    return artifact_root / "backends" / "nanogpt" / "data" / "wikitext103_gpt2_50m"


def flush_tokens(path, tokens):
    with open(path, "ab") as f:
        array("H", tokens).tofile(f)
    tokens.clear()


def download_with_retries(url, path, retries, retry_sleep):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    current_url = url
    for attempt in range(1, retries + 1):
        try:
            print(f"downloading {current_url}")
            request = urllib.request.Request(
                current_url,
                headers={"User-Agent": "Mozilla/5.0 nanogpt-wikitext-prepare"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return str(path)
        except urllib.error.HTTPError as exc:
            last_error = exc
            redirected = s3_permanent_redirect_url(current_url, exc)
            if redirected and redirected != current_url:
                print(f"S3 redirected download endpoint to {redirected}")
                current_url = redirected
                continue
            if attempt >= retries:
                break
            wait = retry_sleep * attempt
            print(f"download failed on attempt {attempt}/{retries}: {exc}")
            print(f"retrying in {wait:.1f}s...")
            time.sleep(wait)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = retry_sleep * attempt
            print(f"download failed on attempt {attempt}/{retries}: {exc}")
            print(f"retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_error


def s3_permanent_redirect_url(url, http_error):
    if http_error.code not in (301, 302, 307, 308):
        return None
    location = http_error.headers.get("Location")
    if location:
        return urllib.parse.urljoin(url, location)

    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "s3.amazonaws.com":
        return None
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        return None
    bucket = path_parts[0]
    key = "/".join(path_parts[1:])
    region = http_error.headers.get("x-amz-bucket-region") or "us-west-2"
    return f"{parsed.scheme}://s3.{region}.amazonaws.com/{bucket}/{key}"


def ensure_raw_zip(*, zip_path, download_url, cache_dir, retries, retry_sleep):
    if zip_path:
        zip_path = Path(zip_path)
    else:
        cache_root = Path(cache_dir or Path(__file__).resolve().parent)
        zip_path = cache_root / "wikitext-103-raw-v1.zip"
    if zip_path.exists():
        return str(zip_path)
    return download_with_retries(download_url, zip_path, retries, retry_sleep)


def raw_split_member(split):
    if split not in RAW_SPLIT_FILES:
        raise ValueError(f"unsupported raw WikiText split: {split}")
    return f"{RAW_ZIP_ROOT}/{RAW_SPLIT_FILES[split]}"


def encode_raw_zip_split(*, zip_path, split, enc, eos, max_tokens, output_path):
    member = raw_split_member(split)
    buffer = []
    token_count = 0
    line_count = 0
    nonempty_line_count = 0

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        if member not in names:
            raise FileNotFoundError(f"{member} not found in {zip_path}; found examples: {list(sorted(names))[:5]}")
        with zf.open(member) as raw:
            for raw_line in raw:
                line_count += 1
                text = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not text.strip():
                    continue
                nonempty_line_count += 1
                tokens = enc.encode_ordinary(text)
                tokens.append(eos)
                keep = min(len(tokens), max_tokens - token_count)
                if keep > 0:
                    buffer.extend(tokens[:keep])
                    token_count += keep
                    if len(buffer) >= WRITE_CHUNK_TOKENS:
                        flush_tokens(output_path, buffer)
                if token_count >= max_tokens:
                    break

    if buffer:
        flush_tokens(output_path, buffer)

    return {
        "split": split,
        "tokens": token_count,
        "lines_seen": line_count,
        "nonempty_lines_seen": nonempty_line_count,
        "reached_token_limit": token_count >= max_tokens,
        "source_member": member,
    }


def parquet_files_for_split(split):
    if split not in HF_PARQUET_SPLIT_FILES:
        raise ValueError(f"unsupported HF parquet WikiText split: {split}")
    return HF_PARQUET_SPLIT_FILES[split]


def ensure_hf_parquet_files(*, split, cache_dir, retries, retry_sleep):
    cache_root = Path(cache_dir or Path(__file__).resolve().parent) / "wikitext-103-raw-v1-parquet"
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in parquet_files_for_split(split):
        path = cache_root / filename
        if not path.exists():
            url = f"{HF_PARQUET_BASE_URL}/{filename}"
            download_with_retries(url, path, retries, retry_sleep)
        paths.append(str(path))
    return paths


def parquet_text_batches(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading direct Hugging Face parquet files requires pyarrow. "
            "Install it or use --source raw-zip."
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["text"], batch_size=4096):
        column = batch.column("text")
        for value in column.to_pylist():
            yield value or ""


def encode_hf_parquet_split(*, split, enc, eos, max_tokens, output_path, cache_dir, retries, retry_sleep):
    parquet_paths = ensure_hf_parquet_files(
        split=split,
        cache_dir=cache_dir,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    buffer = []
    token_count = 0
    row_count = 0
    nonempty_row_count = 0

    for parquet_path in parquet_paths:
        for text in parquet_text_batches(parquet_path):
            row_count += 1
            if not text.strip():
                continue
            nonempty_row_count += 1
            tokens = enc.encode_ordinary(text)
            tokens.append(eos)
            keep = min(len(tokens), max_tokens - token_count)
            if keep > 0:
                buffer.extend(tokens[:keep])
                token_count += keep
                if len(buffer) >= WRITE_CHUNK_TOKENS:
                    flush_tokens(output_path, buffer)
            if token_count >= max_tokens:
                break
        if token_count >= max_tokens:
            break

    if buffer:
        flush_tokens(output_path, buffer)

    return {
        "split": split,
        "tokens": token_count,
        "rows_seen": row_count,
        "nonempty_rows_seen": nonempty_row_count,
        "reached_token_limit": token_count >= max_tokens,
        "source_parquet_files": parquet_paths,
    }


def load_split_with_retries(
    *,
    dataset_name,
    dataset_config,
    split,
    streaming,
    cache_dir,
    retries,
    retry_sleep,
):
    from datasets import load_dataset

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=streaming,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            wait = retry_sleep * attempt
            print(
                f"load_dataset failed for split={split!r} on attempt {attempt}/{retries}: {exc}"
            )
            print(f"retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"failed to load split={split!r} after {retries} attempts") from last_error


def encode_split(
    *,
    dataset_name,
    dataset_config,
    split,
    enc,
    eos,
    max_tokens,
    output_path,
    streaming,
    cache_dir,
    retries,
    retry_sleep,
):
    dataset = load_split_with_retries(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    buffer = []
    token_count = 0
    document_count = 0
    nonempty_document_count = 0

    for row in dataset:
        document_count += 1
        text = row.get("text", "")
        if not text or not text.strip():
            continue

        nonempty_document_count += 1
        tokens = enc.encode_ordinary(text)
        tokens.append(eos)
        if not tokens:
            continue

        keep = min(len(tokens), max_tokens - token_count)
        if keep > 0:
            buffer.extend(tokens[:keep])
            token_count += keep
            if len(buffer) >= WRITE_CHUNK_TOKENS:
                flush_tokens(output_path, buffer)

        if token_count >= max_tokens:
            break

    if buffer:
        flush_tokens(output_path, buffer)

    return {
        "split": split,
        "tokens": token_count,
        "documents_seen": document_count,
        "nonempty_documents_seen": nonempty_document_count,
        "reached_token_limit": token_count >= max_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-tokens", type=int, default=DEFAULT_MAX_TRAIN_TOKENS)
    parser.add_argument("--max-val-tokens", type=int, default=DEFAULT_MAX_VAL_TOKENS)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--dataset-config", default=DATASET_CONFIG)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--load-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--source",
        default="hf",
        choices=["hf", "raw-zip", "hf-parquet"],
        help="data source: Hugging Face datasets API, direct WikiText raw zip, or direct Hugging Face parquet files",
    )
    parser.add_argument("--download-url", default=DEFAULT_RAW_ZIP_URL)
    parser.add_argument("--zip-path", default=None, help="local wikitext-103-raw-v1.zip path")
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="optional Hugging Face endpoint mirror, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="download/cache the full WikiText split before tokenization instead of streaming",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "directory to write train.bin, val.bin, and meta.pkl; defaults to "
            "backends/nanogpt/data/wikitext103_gpt2_50m inside this repository"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output directory with train.bin or val.bin",
    )
    args = parser.parse_args()

    if args.max_train_tokens <= 0 or args.max_val_tokens <= 0:
        raise ValueError("--max-train-tokens and --max-val-tokens must be positive")
    if args.load_retries <= 0:
        raise ValueError("--load-retries must be positive")
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    output_dir = os.path.abspath(args.output_dir or default_output_dir())
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")
    summary_path = os.path.join(output_dir, "prepare_summary.json")

    for path in (train_path, val_path, meta_path, summary_path):
        if os.path.exists(path):
            if not args.force:
                raise FileExistsError(f"{path} already exists; pass --force to overwrite")
            os.remove(path)

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eos = enc.eot_token

    if args.source == "raw-zip":
        zip_path = ensure_raw_zip(
            zip_path=args.zip_path,
            download_url=args.download_url,
            cache_dir=args.cache_dir,
            retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )
        train_summary = encode_raw_zip_split(
            zip_path=zip_path,
            split=args.train_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_train_tokens,
            output_path=train_path,
        )
        val_summary = encode_raw_zip_split(
            zip_path=zip_path,
            split=args.val_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_val_tokens,
            output_path=val_path,
        )
    elif args.source == "hf-parquet":
        zip_path = None
        train_summary = encode_hf_parquet_split(
            split=args.train_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_train_tokens,
            output_path=train_path,
            cache_dir=args.cache_dir,
            retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )
        val_summary = encode_hf_parquet_split(
            split=args.val_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_val_tokens,
            output_path=val_path,
            cache_dir=args.cache_dir,
            retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )
    else:
        zip_path = None
        train_summary = encode_split(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.train_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_train_tokens,
            output_path=train_path,
            streaming=not args.no_streaming,
            cache_dir=args.cache_dir,
            retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )
        val_summary = encode_split(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.val_split,
            enc=enc,
            eos=eos,
            max_tokens=args.max_val_tokens,
            output_path=val_path,
            streaming=not args.no_streaming,
            cache_dir=args.cache_dir,
            retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )

    meta = {
        "vocab_size": enc.n_vocab,
        "tokenizer": "gpt2",
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "train_tokens": train_summary["tokens"],
        "val_tokens": val_summary["tokens"],
        "output_dir": output_dir,
        "streaming": not args.no_streaming,
        "hf_endpoint": args.hf_endpoint,
        "cache_dir": args.cache_dir,
        "source": args.source,
        "zip_path": zip_path,
        "download_url": args.download_url if args.source == "raw-zip" else None,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    summary = {
        **meta,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "train_bin": train_path,
        "val_bin": val_path,
        "meta_pkl": meta_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"dataset: {args.dataset_name}")
    print(f"config: {args.dataset_config}")
    print("tokenizer: gpt2")
    print(f"vocab size: {enc.n_vocab:,}")
    print(f"output dir: {output_dir}")
    print(f"train has {train_summary['tokens']:,} tokens")
    print(f"val has {val_summary['tokens']:,} tokens")
    if not train_summary["reached_token_limit"]:
        print("warning: train split ended before max train token target")
    if not val_summary["reached_token_limit"]:
        print("warning: val split ended before max val token target")


if __name__ == "__main__":
    main()
