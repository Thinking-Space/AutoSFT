import os
import sys
from tqdm import tqdm
import multiprocessing
from functools import partial
import traceback
from pathlib import Path

from autosft.core.common import REPO_ROOT

import argparse
import json


def add_tokenizer_args(parser):
    parser.add_argument(
        "--tokenizer-model", default=os.environ.get("AUTO_SFT_SFT_TOKENIZER_MODEL")
    )


def build_tokenizer(args):
    from transformers import AutoTokenizer

    if not args.tokenizer_model:
        raise ValueError(
            "--tokenizer-model or AUTO_SFT_SFT_TOKENIZER_MODEL is required"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, local_files_only=True
    )
    template = REPO_ROOT / "src/autosft/data/templates/qwen3.jinja"
    tokenizer.chat_template = template.read_text()

    class Tokenizer:
        def __init__(self, backend):
            self.tokenizer = backend
            self.eos_token_id = backend.eos_token_id
            self.pad_token_id = backend.pad_token_id

        def encode(self, text, **kwargs):
            return self.tokenizer.encode(text, **kwargs)

    return Tokenizer(tokenizer)


from autosft.data.training_data import write_token_dataset


def build_char_to_token_map(tokenizer, text, tokens):
    """Map character positions to token indices."""
    char_to_token = {}

    # Prefer native offset mappings when supported.
    try:
        encoding = tokenizer.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        if "offset_mapping" in encoding:
            for token_idx, (start, end) in enumerate(encoding["offset_mapping"]):
                for char_idx in range(start, end):
                    if char_idx not in char_to_token:
                        char_to_token[char_idx] = token_idx
            return char_to_token
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        pass

    # Fall back to incremental decoding.
    current_text = ""
    for token_idx, token_id in enumerate(tokens):
        # Decode through the current token.
        decoded = tokenizer.tokenizer.decode(
            tokens[: token_idx + 1], skip_special_tokens=False
        )
        # Map newly decoded characters.
        for char_idx in range(len(current_text), len(decoded)):
            if char_idx < len(text):
                char_to_token[char_idx] = token_idx
        current_text = decoded

    return char_to_token


def encode_row(data, tokenizer):
    """Encode a parsed row as (tokens, loss_mask).

    Conversion and --count-only share this tokenization path so their per-row
    token counts agree. The token dataset uses mask 0 for supervised assistant tokens
    and mask 1 for tokens excluded from the loss.
    """
    try:
        tokens, loss_mask = [], []
        if "input" in data and "output" in data:
            # check data["input"] and data["output"]
            if (
                type(data["input"]) != str
                or type(data["output"]) != str
                or len(data["output"]) == 0
            ):
                return None

            input_ids = tokenizer.encode(data["input"], add_special_tokens=True)
            output_ids = tokenizer.encode(data["output"], add_special_tokens=False)
            tokens.extend(input_ids)
            tokens.extend(output_ids)
            loss_mask.extend([1] * (len(input_ids) - 1) + [0])
            loss_mask.extend([0] * len(output_ids))
            tokens.append(tokenizer.eos_token_id)
            loss_mask.append(1)

        # messages chatml format
        elif "messages" in data:
            messages = data["messages"]
            if "tools" in data:
                tools = data["tools"]
            else:
                tools = None

            # Render the complete conversation.
            full_text = tokenizer.tokenizer.apply_chat_template(
                messages, tokenize=False, tools=tools
            )

            # Tokenize with character offsets when supported.
            try:
                #     full_text,
                #     add_special_tokens=False,
                #     return_offsets_mapping=True
                # )
                encoding = tokenizer.tokenizer(
                    full_text, add_special_tokens=True, return_offsets_mapping=True
                )
                tokens = encoding["input_ids"]
                offset_mapping = encoding.get("offset_mapping", None)
            except (AttributeError, TypeError, ValueError, NotImplementedError):
                # Fall back when offset mappings are unavailable.
                tokens = tokenizer.tokenizer(full_text, add_special_tokens=True)[
                    "input_ids"
                ]
                offset_mapping = None

            # Exclude all tokens from the loss initially.
            loss_mask = [1] * len(tokens)

            # Locate assistant boundaries for supervised tokens.
            assistant_start_marker = "<|im_start|>assistant"
            assistant_end_marker = "<|im_end|>"

            # Locate assistant spans, including their markers.
            assistant_regions = []
            pos = 0
            while True:
                start_pos = full_text.find(assistant_start_marker, pos)
                if start_pos == -1:
                    break
                # Find the corresponding <|im_end|>.
                end_pos = full_text.find(assistant_end_marker, start_pos)
                start_pos += len(assistant_start_marker)
                if end_pos == -1:
                    # The last message may be incomplete.
                    end_pos = len(full_text)
                    region_end = end_pos
                else:
                    # Preserve the established mask: end markers remain excluded.
                    region_end = end_pos

                # Preserve the assistant tag at the start and include EOS.
                assistant_regions.append((start_pos, region_end))
                pos = (
                    end_pos + len(assistant_end_marker)
                    if end_pos < len(full_text)
                    else len(full_text)
                )

            # Append EOS when required and absent.
            if len(tokens) > 0 and tokens[-1] != tokenizer.eos_token_id:
                tokens.append(tokenizer.eos_token_id)
                loss_mask.append(1)  # Excluded unless covered by an assistant span.

            # Map character positions to token positions.
            if offset_mapping is not None:
                # Use native offsets.
                for region_start, region_end in assistant_regions:
                    for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                        # Supervise tokens overlapping an assistant span.
                        if char_start < region_end and char_end > region_start:
                            loss_mask[token_idx] = 0
            else:
                # Fall back to the decoding helper.
                char_to_token = build_char_to_token_map(tokenizer, full_text, tokens)
                for region_start, region_end in assistant_regions:
                    for char_idx in range(region_start, region_end):
                        token_idx = char_to_token.get(char_idx)
                        if token_idx is not None and token_idx < len(loss_mask):
                            loss_mask[token_idx] = 0

            # Count assistant tokens for validation.
            assistant_token_count = sum(1 for m in loss_mask if m == 0)
            if assistant_token_count == 0 and any(
                msg.get("role") == "assistant" for msg in messages
            ):
                pass

        else:
            return None

        assert len(tokens) == len(loss_mask)
        return tokens, loss_mask
    except Exception as e:
        print(f"Error processing line: {e}")
        print(traceback.format_exc())
        return None


def count_line(line, tokenizer):
    """Return (pool_row_id, total_tokens, loss_tokens) for --count-only.

    Read stable identities from source rows rather than physical file order.
    Loss tokens have mask 0 and belong to supervised assistant spans.
    """
    try:
        data = json.loads(line)
    except Exception:
        return None
    encoded = encode_row(data, tokenizer)
    if encoded is None:
        return None
    tokens, loss_mask = encoded
    row_id = data.get("pool_row_id")
    return (
        row_id if isinstance(row_id, int) else -1,
        len(tokens),
        sum(1 for m in loss_mask if m == 0),
    )


g_tokenizer = None


def init_worker(args):
    global g_tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    g_tokenizer = build_tokenizer(args)


def count_worker_wrapper(line):
    return count_line(line, g_tokenizer)


def encode_training_line(item):
    line_number, line = item
    encoded = encode_row(json.loads(line), g_tokenizer)
    if encoded is None:
        raise ValueError(f"Cannot encode training row at line {line_number}")
    return encoded


def run_write(args):
    init_worker(args)

    def lines():
        with open(args.json_file, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    yield line_number, line

    if args.workers > 1:
        with multiprocessing.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(args,),
        ) as pool:
            manifest = write_token_dataset(
                Path(args.output_sstable),
                pool.imap(encode_training_line, lines(), chunksize=200),
                pad_token_id=g_tokenizer.eos_token_id,
            )
    else:
        manifest = write_token_dataset(
            Path(args.output_sstable),
            map(encode_training_line, lines()),
            pad_token_id=g_tokenizer.eos_token_id,
        )
    print(f"written_rows {manifest['rows']}")
    print(f"total_token {manifest['total_tokens']}")


def run():
    # This process only tokenizes and serializes; no model backend is needed.
    # Set before Transformers is imported, and inherit in worker processes.
    os.environ["USE_TORCH"] = "0"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="converter")
    group.add_argument(
        "--json-file", "-i", type=str, required=True, help="input raw json file."
    )
    group.add_argument(
        "--output-data",
        "-o",
        dest="output_sstable",
        type=str,
        default=None,
        help="Output public token dataset manifest. Required unless --count-only.",
    )
    group.add_argument(
        "--tokenizer",
        "-v",
        choices=["qwen3"],
        required=True,
        help="fixed tokenizer version.",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of worker processes. 0 means use all available CPUs.",
    )
    group.add_argument(
        "--count-only",
        action="store_true",
        help="Do not write a dataset; only measure per-row token lengths "
        "and write them to --count-output as TSV "
        '"pool_row_id<TAB>total_tokens<TAB>loss_tokens". Uses the same '
        "encode_row() path as conversion, so the counts match total_token.",
    )
    group.add_argument(
        "--count-output",
        type=str,
        default=None,
        help="Output TSV path for --count-only.",
    )
    add_tokenizer_args(parser)
    args = parser.parse_args()
    args.tokenizer_version = args.tokenizer

    assert os.path.exists(args.json_file), (
        f"input json file {args.json_file} does not exist"
    )
    if args.count_only:
        assert args.count_output, "--count-only requires --count-output"
    else:
        assert args.output_sstable, "--output-sstable is required unless --count-only"

    # Determine number of workers
    if args.workers <= 0:
        # Default to CPU count, minimum 1
        args.workers = max(1, (os.cpu_count() or 1) - 2)

    if args.count_only:
        run_count_only(args)
        return

    run_write(args)


def run_count_only(args):
    """Measure per-row token lengths without building an SSTable.

    Streams the input (the AutoSFT pool is ~1.5GB, so readlines() is not an
    option here) and writes one TSV row per usable input row. Rows the
    converter would skip are simply absent from the output, and the caller
    reconciles them against the pool row count.
    """

    def iter_lines():
        with open(args.json_file, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    yield line

    written = 0
    total_tokens = 0
    loss_tokens = 0
    # Create the destination directory rather than failing on it: the caller
    # names a scratch path, and a missing parent is not a measurement error.
    count_dir = os.path.dirname(os.path.abspath(args.count_output))
    if count_dir:
        os.makedirs(count_dir, exist_ok=True)
    tmp_path = f"{args.count_output}.partial"
    with open(tmp_path, "w", encoding="utf-8") as out:
        if args.workers > 1:
            print(f"Counting tokens with {args.workers} workers...", flush=True)
            with multiprocessing.Pool(
                processes=args.workers, initializer=init_worker, initargs=(args,)
            ) as pool:
                results = pool.imap(count_worker_wrapper, iter_lines(), chunksize=200)
                for res in tqdm(results, desc="Counting tokens"):
                    if not res:
                        continue
                    row_id, n_tokens, n_loss = res
                    out.write(f"{row_id}\t{n_tokens}\t{n_loss}\n")
                    written += 1
                    total_tokens += n_tokens
                    loss_tokens += n_loss
        else:
            print("Counting tokens with single thread...", flush=True)
            tokenizer = build_tokenizer(args)
            for line in tqdm(iter_lines(), desc="Counting tokens"):
                res = count_line(line, tokenizer)
                if not res:
                    continue
                row_id, n_tokens, n_loss = res
                out.write(f"{row_id}\t{n_tokens}\t{n_loss}\n")
                written += 1
                total_tokens += n_tokens
                loss_tokens += n_loss
    os.replace(tmp_path, args.count_output)
    # Mirror the converter's "total_token N" line so callers can cross-check the
    # index against a real conversion of the same input.
    print(f"count_rows {written}")
    print(f"count_loss_token {loss_tokens}")
    print(f"total_token {total_tokens}")


if __name__ == "__main__":
    # Set start method to fork to leverage copy-on-write if possible/safe,
    # but for tokenizer safety with locks, 'spawn' is sometimes safer but slower.
    # Linux default is fork.
    run()
