"""Check legacy public JVM descriptors against a replacement AAR (or classes.jar)."""

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


def public_api(path, javap, directory):
    if path.suffix == ".aar":
        with zipfile.ZipFile(path) as archive:
            jar = directory / "classes.jar"
            jar.write_bytes(archive.read("classes.jar"))
    else:
        jar = path
    with zipfile.ZipFile(jar) as archive:
        classes = [
            name[:-6].replace("/", ".")
            for name in archive.namelist()
            if name.endswith(".class")
        ]
    output = subprocess.check_output(
        [javap, "-classpath", str(jar), "-public", "-s", *classes], text=True
    )
    api = {}
    current = None
    declaration = ""
    for line in output.splitlines():
        if line.endswith("{"):
            match = re.search(r"(?:class|interface|enum) ([\w.$]+)", line)
            current = match[1] if match and line.startswith("public ") else None
            if current:
                api[current] = {"declaration": line, "members": {}}
        elif line.strip().startswith("public "):
            declaration = line.strip()
        elif current and line.strip().startswith("descriptor:"):
            name = declaration.split("(")[0].split()[-1].rstrip(";")
            descriptor = line.split("descriptor:", 1)[1].strip()
            api[current]["members"][name, descriptor] = declaration
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--javap", default="javap")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        baseline = public_api(args.baseline, args.javap, Path(first))
        candidate = public_api(args.candidate, args.javap, Path(second))
    failures = []
    count = 0
    for name, old in baseline.items():
        new = candidate.get(name)
        if new is None:
            failures.append(f"Missing class: {name}")
            continue
        for key, declaration in old["members"].items():
            count += 1
            replacement = new["members"].get(key)
            if replacement is None:
                failures.append(f"Missing member: {name}: {declaration} {key[1]}")
            elif (" static " in declaration) != (" static " in replacement):
                failures.append(
                    f"Changed static/instance member: {name}: {declaration}"
                )
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"PASS: {len(baseline)} public classes, {count} legacy member descriptors retained"
    )


if __name__ == "__main__":
    main()
