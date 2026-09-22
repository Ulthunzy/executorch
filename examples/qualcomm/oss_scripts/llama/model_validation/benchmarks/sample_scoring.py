# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conservative sample scoring; not an OpenCompass leaderboard reproduction."""

import re
import unicodedata
from fractions import Fraction
from functools import lru_cache
from importlib import metadata

CHOICE_BENCHMARKS = {
    "MMLU-Pro",
    "MMLU-Redux",
    "C-Eval",
    "SuperGPQA",
    "MMLU",
    "LongBench-v2",
    "CMMLU",
    "HellaSwag",
    "ARC-Challenge",
}


def boxed_answers(text):
    """Extract balanced LaTeX boxes, including nested fractions and radicals."""
    answers = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, start = 1, match.end()
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            if depth == 0:
                answers.append(text[start:index])
                break
    return answers


def normalize_math(text):
    """Normalize harmless LaTeX spelling differences without executing expressions."""
    text = unicodedata.normalize("NFKC", text).strip().strip("$")
    for value in (r"\left", r"\right", r"\!", r"\,", r"\;", r"\ "):
        text = text.replace(value, "")
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = re.sub(r"\^\{?\\circ\}?|°", "", text)
    return re.sub(r"\s+", "", text)


def numeric_value(text):
    """Recognize finite decimal or rational answers without eval or symbolic parsing."""
    text = normalize_math(text)
    if "," in text and not re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        return None
    text = text.replace(",", "")
    exponent = re.search(r"[eE]([-+]?\d+)$", text)
    if len(text) > 512 or (
        exponent and (len(exponent[1]) > 5 or abs(int(exponent[1])) > 1000)
    ):
        return None
    fraction = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
    if fraction:
        text = "/".join(fraction.groups())
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\d+/\d+", text):
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract_choice(answer):
    """Use an explicit answer marker, final box, or standalone final option."""
    normalized = answer.replace("**", "").strip()
    marked = list(
        re.finditer(
            r"(?:^|\n)\s*(?:(?:THE\s+)?(?:FINAL\s+)?ANSWER\s*(?:IS)?|最终答案(?:是|为)?|正确(?:答案|选项)(?:是|为)?|答案(?:是|为)?)"
            r"\s*[:：]?\s*[\(\[（]?\s*([A-P])\b",
            normalized,
            re.IGNORECASE,
        )
    )
    if marked:
        tail = normalized[marked[-1].end() :].split("\n", 1)[0]
        if re.match(
            r"\s*(?:or\b|and\b|或者|或|和|与|/|,)\s*[A-P]\b", tail, re.IGNORECASE
        ):
            return None
        return marked[-1][1].upper()
    boxes = boxed_answers(normalized)
    if boxes:
        candidate = boxes[-1].strip().upper()
        if re.fullmatch(r"[A-P]", candidate):
            return candidate
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) == 1:
        labeled = re.findall(
            r"(?:^|\s)[\(\[（]?([A-P])[\)\]）]?[.。:：、](?=\s|[^A-Za-z])",
            lines[0],
            re.IGNORECASE,
        )
        if len(labeled) == 1 and re.match(
            r"^[\(\[（]?[A-P][\)\]）]?[.。:：、]", lines[0], re.IGNORECASE
        ):
            return labeled[0].upper()
    if lines and re.fullmatch(
        r"[\(\[（]?[A-P][\)\]）]?[.。]?", lines[-1], re.IGNORECASE
    ):
        return re.search(r"[A-P]", lines[-1], re.IGNORECASE)[0].upper()
    return None


def extract_number(answer):
    """Require a box, answer marker, or standalone numeric final line."""
    boxes = boxed_answers(answer)
    if boxes:
        return boxes[-1]
    marked = re.findall(
        r"(?:####|(?:final\s+)?answer\s*(?:is|:)|答案\s*(?:是|为|[:：]))\s*"
        r"\$?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)(?![\w.])",
        answer,
        re.IGNORECASE,
    )
    if marked:
        return marked[-1]
    equality = re.search(
        r"[=＝]\s*\$?([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\$?"
        r"\s*(?:<\|[^<>\r\n]+\|>\s*)*$",
        answer,
    )
    if equality:
        return equality[1]
    lines = [line.strip().strip("$") for line in answer.splitlines() if line.strip()]
    if lines and numeric_value(lines[-1]) is not None:
        return lines[-1]
    return None


@lru_cache(maxsize=1)
def ifeval_tools():
    """Load the installed harness only when NLTK resources already exist locally."""
    import nltk

    # The harness otherwise downloads punkt_tab during import.
    nltk.data.find("tokenizers/punkt_tab")
    from lm_eval.tasks.ifeval.utils import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    return (
        InputExample,
        test_instruction_following_strict,
        test_instruction_following_loose,
        metadata.version("lm_eval"),
    )


def score_answer(benchmark, reference, answer):
    """Return an auditable verdict, leaving unsupported equivalences unjudged."""
    verdict = {
        "correct": None,
        "prediction": None,
        "target": None,
        "score_method": "unscored",
    }
    if benchmark.startswith("repeat/"):
        return {**verdict, "score_method": "no_reference"}
    if not answer:
        return {**verdict, "correct": False, "score_method": "no_final_answer"}
    if benchmark in CHOICE_BENCHMARKS:
        target = str(reference).strip().upper()
        if not re.fullmatch(r"[A-P]", target):
            raise ValueError(f"Invalid choice reference: {reference!r}")
        prediction = extract_choice(answer)
        return {
            **verdict,
            "target": target,
            "prediction": prediction,
            "correct": prediction == target if prediction else None,
            "score_method": "explicit_choice" if prediction else "needs_review",
        }
    if benchmark == "IFEval":
        try:
            example_type, strict, loose, version = ifeval_tools()
            example = example_type(
                **{
                    key: reference[key]
                    for key in ("key", "instruction_id_list", "prompt", "kwargs")
                }
            )
            strict_result = strict(example, answer)
            loose_result = loose(example, answer)
            return {
                **verdict,
                "correct": strict_result.follow_all_instructions,
                "score_method": "lm_eval_ifeval_strict",
                "ifeval_strict": strict_result.follow_instruction_list,
                "ifeval_loose": loose_result.follow_instruction_list,
                "lm_eval_version": version,
            }
        except (ImportError, LookupError, ValueError, TypeError, KeyError) as error:
            return {
                **verdict,
                "score_method": "grader_unavailable",
                "score_error": str(error),
            }
    if benchmark in ("AIME25", "AIME26", "GSM8K", "MATH-500"):
        target = str(reference)
        if benchmark == "GSM8K":
            if "####" not in target:
                raise ValueError("GSM8K reference lacks the final #### answer")
            target = target.rsplit("####", 1)[-1].strip()
        elif benchmark == "MATH-500":
            boxes = boxed_answers(target)
            if not boxes:
                raise ValueError("MATH-500 reference lacks a boxed answer")
            target = boxes[-1]
        prediction = extract_number(answer)
        if prediction is None:
            return {**verdict, "target": target, "score_method": "needs_review"}
        a, b = numeric_value(prediction), numeric_value(target)
        if a is not None and b is not None:
            correct, method = a == b, "exact_number"
        elif normalize_math(prediction) == normalize_math(target):
            correct, method = True, "normalized_latex_match"
        else:
            correct, method = None, "symbolic_review_required"
        return {
            **verdict,
            "target": target,
            "prediction": prediction,
            "correct": correct,
            "score_method": method,
        }
    return {**verdict, "score_method": "unsupported_benchmark"}
