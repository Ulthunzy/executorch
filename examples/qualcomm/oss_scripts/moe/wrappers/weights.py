"""Split an already shared QNN context using SDK-reported weight ranges."""

import hashlib
import json
import struct
import shutil


def separate_context(context, requests, output, manifest=None):
    ranges = sorted(
        {tuple(map(int, row.split()[:2])) for row in requests.read_text().splitlines()}
    )
    size = context.stat().st_size
    if not ranges:
        raise ValueError("No weight requests were captured")
    cursor, metadata = 0, []
    for offset, length in ranges:
        if (
            offset % 4096
            or length % 4096
            or length <= 0
            or offset < cursor
            or offset + length > size
        ):
            raise ValueError("Expected disjoint page-aligned callback sections")
        if offset > cursor:
            metadata.append((cursor, offset - cursor))
        cursor = offset + length
    if cursor < size:
        metadata.append((cursor, size - cursor))
    output.mkdir(parents=True, exist_ok=True)
    graph_path = output / "graph.qnn"
    weight_path = output / "weights.bin"
    if context.resolve() in (graph_path.resolve(), weight_path.resolve()):
        raise ValueError("Output must differ from the source context")
    header = struct.pack("<8s4Q", b"Q35CTX01", 1, size, len(metadata), len(ranges))
    table_size = len(header) + 24 * (len(metadata) + len(ranges))
    graph_cursor = (table_size + 4095) & ~4095
    graph_records, weight_records = [], []
    with context.open("rb") as source, graph_path.open("wb") as graph:
        graph.seek(graph_cursor)
        for offset, length in metadata:
            graph_records.append((offset, length, graph_cursor))
            source.seek(offset)
            graph.write(source.read(length))
            graph_cursor += length
        with weight_path.open("wb") as weights:
            for offset, length in ranges:
                weight_records.append((offset, length, weights.tell()))
                source.seek(offset)
                remaining = length
                while remaining:
                    block = source.read(min(remaining, 4 * 1024 * 1024))
                    if not block:
                        raise ValueError("Truncated source context")
                    weights.write(block)
                    remaining -= len(block)
        graph.seek(0)
        graph.write(header)
        for record in graph_records + weight_records:
            graph.write(struct.pack("<3Q", *record))
    with context.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    report = dict(
        version=1,
        source=str(context.resolve()),
        source_sha256=digest,
        logical_context_bytes=size,
        graph_bytes=graph_path.stat().st_size,
        weight_bytes=weight_path.stat().st_size,
        graph_sections=graph_records,
        weight_sections=weight_records,
        layout="Opaque context image reconstructed with read-only file mappings; no weight copy into graph file",
    )
    (output / "WEIGHTS.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = manifest or context.parent / "GRAPHS.json"
    if manifest.is_file():
        for path in [manifest, *sorted(manifest.parent.glob("*.io"))]:
            target = output / path.name
            if path.resolve() != target.resolve():
                shutil.copy2(path, target)
    return report
