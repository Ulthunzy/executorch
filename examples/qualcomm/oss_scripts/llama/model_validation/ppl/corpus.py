"""Portable fixed text, token IDs and windows, independent of model manifests."""

import importlib.metadata
import sys
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import digest, file_hash, read_json, write_json

SCHEMA = "executorch.ppl.corpus.v1"
TOKENIZER_FILES = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab*",
    "merges.txt",
    "*.model",
    "chat_template*",
)


def windows(length, context_len, stride):
    """Partition targets 1..N-1 using the previously validated overlapping rule."""
    if any(type(value) is not int for value in (length, context_len, stride)):
        raise ValueError("Token count, context length and stride must be integers")
    if length < 2 or context_len < 2 or not 1 <= stride < context_len:
        raise ValueError(
            "Require >= 2 tokens, context-len >= 2, 1 <= stride < context-len"
        )
    previous = 1
    result = []
    for begin in range(0, length - 1, stride):
        end = min(begin + context_len, length)
        result.append(
            dict(id=f"W{len(result):06d}", begin=begin, end=end, target_begin=previous)
        )
        previous = end
        if end == length:
            break
    return result


def load_tokenizer(directory):
    """Load only a local tokenizer, with custom remote code disabled."""
    from transformers import AutoTokenizer

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("--tokenizer must be a local directory")
    return AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )


def tokenizer_identity(tokenizer):
    """Bind the full token-to-ID mapping and special-token semantics."""
    vocabulary = tokenizer.get_vocab()
    return dict(
        vocab_sha256=digest(vocabulary),
        vocab_size=len(vocabulary),
        max_token_id=max(vocabulary.values()),
        special_tokens_sha256=digest(tokenizer.special_tokens_map),
    )


def prepare(tokenizer_directory, text_path, context_len, stride, output):
    """Publish a fully validated corpus atomically, preserving failed staging."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Corpus output must be empty; prepared corpora are immutable")
    tokenizer_directory = Path(tokenizer_directory).resolve()
    text_bytes = Path(text_path).read_bytes()
    text = text_bytes.decode("utf-8")
    tokenizer = load_tokenizer(tokenizer_directory)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if any(type(token) is not int or not 0 <= token < 2**31 for token in ids):
        raise ValueError("Expected nonnegative token IDs representable as signed int32")
    evaluation_windows = windows(len(ids), context_len, stride)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent)
    )
    try:
        (staging / "text.txt").write_bytes(text_bytes)
        np.asarray(ids, dtype="<u8").tofile(staging / "tokens.u64")
        write_json(staging / "windows.json", evaluation_windows)
        snapshot = staging / "tokenizer"
        snapshot.mkdir()
        tokenizer.save_pretrained(snapshot)
        snapshot_files = {
            p.relative_to(staging).as_posix(): file_hash(p)
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        }
        manifest = dict(
            schema=SCHEMA,
            token_dtype="uint64_le",
            token_count=len(ids),
            target_count=len(ids) - 1,
            context_len=context_len,
            stride=stride,
            tokenization=dict(
                add_special_tokens=False, chat_template=False, text_encoding="utf-8"
            ),
            tokenizer=tokenizer_identity(tokenizer),
            files={
                name: file_hash(staging / name)
                for name in ("text.txt", "tokens.u64", "windows.json")
            },
            tokenizer_files=snapshot_files,
            versions={
                name: importlib.metadata.version(name)
                for name in ("transformers", "tokenizers")
            },
        )
        write_json(staging / "manifest.json", manifest)
        load_corpus(staging)
        staging.rename(output)
    except BaseException:
        print(f"Corpus staging retained for inspection: {staging}", file=sys.stderr)
        raise
    return manifest


def load_corpus(root):
    """Validate hashes and exact target coverage before using any cached score."""
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("token_dtype") != "uint64_le":
        raise ValueError("Unsupported corpus schema; use ppl prepare")
    files = manifest.get("files", {})
    if set(files) != {"text.txt", "tokens.u64", "windows.json"}:
        raise ValueError("Invalid corpus file manifest")
    snapshots = manifest.get("tokenizer_files", {})
    actual = {
        p.relative_to(root).as_posix()
        for p in (root / "tokenizer").rglob("*")
        if p.is_file()
    }
    if not snapshots or set(snapshots) != actual:
        raise ValueError("Tokenizer snapshot files changed")
    for name, expected in {**files, **snapshots}.items():
        path = root / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Corpus hash mismatch: {name}")
    if (root / "tokens.u64").stat().st_size % 8:
        raise ValueError("Invalid tokens.u64 byte length")
    tokens = np.fromfile(root / "tokens.u64", dtype="<u8")
    if (
        len(tokens) != manifest["token_count"]
        or len(tokens) < 2
        or tokens.max() >= 2**31
    ):
        raise ValueError("Invalid corpus tokens")
    expected_windows = windows(len(tokens), manifest["context_len"], manifest["stride"])
    saved_windows = read_json(root / "windows.json")
    if saved_windows != expected_windows or manifest["target_count"] != len(tokens) - 1:
        raise ValueError("Invalid windows: missing, repeated or misaligned targets")
    if manifest["tokenization"] != dict(
        add_special_tokens=False, chat_template=False, text_encoding="utf-8"
    ):
        raise ValueError("Unsupported tokenization policy")
    return manifest, tokens.astype(np.int64), saved_windows


def check_tokenizer(directory, root, manifest, tokens):
    """Reject incompatible vocabularies or differing tokenization of the fixed text."""
    directory = Path(directory)
    text = (Path(root) / "text.txt").read_bytes().decode("utf-8")
    if directory.is_file():
        from tokenizers import Tokenizer

        native = Tokenizer.from_file(str(directory))
        if digest(native.get_vocab()) != manifest["tokenizer"]["vocab_sha256"]:
            raise ValueError("Tokenizer vocabulary differs from the frozen corpus")
        if native.encode(text, add_special_tokens=False).ids != tokens.tolist():
            raise ValueError(
                "Tokenizer produces different token IDs for the fixed text"
            )
        return dict(
            vocab_sha256=manifest["tokenizer"]["vocab_sha256"],
            files={directory.name: file_hash(directory)},
            corpus_ids_equal=True,
            format="tokenizer.json",
        )
    tokenizer = load_tokenizer(directory)
    if tokenizer_identity(tokenizer) != manifest["tokenizer"]:
        raise ValueError(
            "Tokenizer vocabulary or special tokens differ from the frozen corpus"
        )
    if tokenizer.encode(text, add_special_tokens=False) != tokens.tolist():
        raise ValueError("Tokenizer produces different token IDs for the fixed text")
    paths = {
        p
        for pattern in TOKENIZER_FILES
        for p in Path(directory).glob(pattern)
        if p.is_file()
    }
    return dict(
        **tokenizer_identity(tokenizer),
        files={p.name: file_hash(p) for p in sorted(paths)},
        corpus_ids_equal=True,
    )
