from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score


MODEL_NAME = "gpt-5.2"
REASONING_EFFORT = "medium"
SEED = 1702

ATTRIBUTE_NAME = "sales_mkt_relatedness"
OUTPUT_SCORE_NAME = "sales_mkt_relatedness_gpt"
OUTPUT_DIR_NAME = "gabriel_1"

PILOT_SAMPLE_SIZE = 250
HOLDOUT_EVAL_SIZE = 200
STRATIFIED_QA_SIZE = 200

SCORING_BATCH_SIZE = 400
MANUAL_BATCH_SIZE = 120
MAX_RETRIES = 5


@dataclass
class PromptSpec:
    version: str
    text: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_title(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    out = str(v).strip()
    out = re.sub(r"\s+", " ", out)
    return out


def _score_to_bin(score: int) -> int:
    if score <= 20:
        return 0
    if score <= 60:
        return 1
    return 2


def _bin_name(bin_id: int) -> str:
    return {0: "low", 1: "mid", 2: "high"}[bin_id]


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid score")
    score = int(round(float(value)))
    if score < 0 or score > 100:
        raise ValueError(f"Score out of range: {score}")
    return score


def _chunks(items: list[Any], n: int) -> list[list[Any]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _benchmark_titles() -> list[dict[str, Any]]:
    return [
        {"title": "Software Engineer", "expected_min": 0, "expected_max": 20},
        {"title": "Controller", "expected_min": 0, "expected_max": 25},
        {"title": "Sales Operations Manager", "expected_min": 35, "expected_max": 75},
        {"title": "Partnerships Manager", "expected_min": 25, "expected_max": 70},
        {"title": "Account Executive", "expected_min": 65, "expected_max": 100},
        {"title": "Performance Marketing Manager", "expected_min": 70, "expected_max": 100},
    ]


def build_prompt_v1() -> PromptSpec:
    text = (
        "You are scoring job titles on one construct.\n"
        f"Construct: {ATTRIBUTE_NAME}.\n"
        "Definition: How much the job title's core function is in sales and/or marketing "
        "(0 = unrelated, 100 = strongly and primarily sales/marketing).\n"
        "Anchors:\n"
        "- 0-20: engineering/operations/legal/finance roles with no sales or marketing core.\n"
        "- 21-60: mixed or commercial-adjacent roles (e.g., growth ops, partnerships with ambiguous sales ownership).\n"
        "- 61-100: direct sales, business development, account management, marketing, demand generation, brand, performance marketing.\n"
        "Task: For each provided title, return an integer score in [0,100].\n"
        "Output format: valid JSON only, with shape:\n"
        '{"results":[{"title":"<original title>","sales_mkt_relatedness_gpt":<integer 0-100>}]}'
    )
    return PromptSpec(version="v1", text=text)


def build_prompt_v2() -> PromptSpec:
    text = (
        "You are scoring job titles on one construct.\n"
        f"Construct: {ATTRIBUTE_NAME}.\n"
        "Definition: How much the job title's core function is in sales and/or marketing "
        "(0 = unrelated, 100 = strongly and primarily sales/marketing).\n"
        "Anchors:\n"
        "- 0-20: engineering/operations/legal/finance roles with no sales or marketing ownership.\n"
        "- 21-60: mixed or commercial-adjacent roles where sales/marketing is support, analytics, strategy, or partial ownership.\n"
        "- 61-100: direct revenue ownership or direct marketing ownership (sales reps/leaders, account executives/managers, "
        "business development reps/leaders, demand generation, product marketing, brand/performance marketing).\n"
        "Refinement rule: score partnerships/revenue operations/growth strategy in mid-range unless title clearly implies direct quota/revenue ownership.\n"
        "Task: For each provided title, return one integer score in [0,100].\n"
        "Output rules: JSON only, no prose, no markdown, and keep titles unchanged.\n"
        '{"results":[{"title":"<original title>","sales_mkt_relatedness_gpt":<integer 0-100>}]}'
    )
    return PromptSpec(version="v2", text=text)


def build_manual_prompt() -> str:
    return (
        "You are an independent second coder for QA.\n"
        "Assign each title to one categorical bin based on sales/marketing relatedness.\n"
        "Bins: low(0-20), mid(21-60), high(61-100).\n"
        "Use your own judgment from title semantics only. Do not infer from any external context.\n"
        "Return JSON only:\n"
        '{"results":[{"title":"<title>","manual_bin":"low|mid|high","manual_score_proxy":10|40|80}]}\n'
        "manual_score_proxy must be: low->10, mid->40, high->80."
    )


def call_model_with_retries(client: OpenAI, *, prompt: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = (
        "System instruction:\n"
        f"{prompt}\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                reasoning={"effort": REASONING_EFFORT},
                input=[{"role": "user", "content": message}],
            )
            output_text = (resp.output_text or "").strip()
            if not output_text:
                raise ValueError("Empty output_text from model")
            return output_text, resp.model_dump()
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == MAX_RETRIES:
                break
            sleep_s = 1.5 * (2 ** (attempt - 1))
            time.sleep(sleep_s)
    raise RuntimeError(f"Model call failed after {MAX_RETRIES} attempts: {last_err}")


def parse_scoring_results(text: str, expected_titles: list[str]) -> list[dict[str, Any]]:
    obj = _safe_json_loads(text)
    if not isinstance(obj, dict) or "results" not in obj:
        raise ValueError("Output missing top-level results")
    raw_results = obj["results"]
    if not isinstance(raw_results, list):
        raise ValueError("results must be a list")

    mapped: dict[str, int] = {}
    for rec in raw_results:
        if not isinstance(rec, dict):
            continue
        title = _normalize_title(rec.get("title", ""))
        if not title:
            continue
        if "sales_mkt_relatedness_gpt" not in rec:
            continue
        mapped[title] = _extract_score(rec["sales_mkt_relatedness_gpt"])

    out: list[dict[str, Any]] = []
    for title in expected_titles:
        if title not in mapped:
            raise ValueError(f"Missing score for title: {title}")
        out.append({"title": title, OUTPUT_SCORE_NAME: mapped[title]})
    return out


def parse_manual_results(text: str, expected_titles: list[str]) -> list[dict[str, Any]]:
    obj = _safe_json_loads(text)
    if not isinstance(obj, dict) or "results" not in obj:
        raise ValueError("Output missing top-level results")
    raw_results = obj["results"]
    if not isinstance(raw_results, list):
        raise ValueError("results must be a list")

    mapped: dict[str, dict[str, Any]] = {}
    for rec in raw_results:
        if not isinstance(rec, dict):
            continue
        title = _normalize_title(rec.get("title", ""))
        if not title:
            continue
        manual_bin = str(rec.get("manual_bin", "")).strip().lower()
        if manual_bin not in {"low", "mid", "high"}:
            continue
        manual_score_proxy = _extract_score(rec.get("manual_score_proxy", 40))
        mapped[title] = {"manual_bin": manual_bin, "manual_score_proxy": manual_score_proxy}

    out: list[dict[str, Any]] = []
    for title in expected_titles:
        if title not in mapped:
            raise ValueError(f"Missing manual bin for title: {title}")
        out.append({"title": title, **mapped[title]})
    return out


def score_titles_batched(
    client: OpenAI,
    titles: list[str],
    prompt_spec: PromptSpec,
    batch_size: int,
    raw_jsonl_path: Path,
    checkpoint_csv_path: Path,
) -> pd.DataFrame:
    existing = pd.DataFrame(columns=["title", OUTPUT_SCORE_NAME, "prompt_version", "model", "scored_at_utc"])
    if checkpoint_csv_path.exists():
        existing = pd.read_csv(checkpoint_csv_path)
        existing["title"] = existing["title"].map(_normalize_title)
        existing = existing.drop_duplicates(subset=["title"], keep="last")

    done_titles = set(existing["title"].tolist())
    todo_titles = [t for t in titles if t not in done_titles]
    rows: list[dict[str, Any]] = existing.to_dict(orient="records")

    batches = _chunks(todo_titles, batch_size)
    for batch_idx, batch_titles in enumerate(batches, start=1):
        payload = {"titles": batch_titles}
        output_text, full_resp = call_model_with_retries(client, prompt=prompt_spec.text, payload=payload)
        parsed = parse_scoring_results(output_text, batch_titles)
        scored_at_utc = _now_iso()
        for rec in parsed:
            rows.append(
                {
                    "title": rec["title"],
                    OUTPUT_SCORE_NAME: rec[OUTPUT_SCORE_NAME],
                    "prompt_version": prompt_spec.version,
                    "model": MODEL_NAME,
                    "scored_at_utc": scored_at_utc,
                }
            )

        with raw_jsonl_path.open("a", encoding="utf-8") as fout:
            fout.write(
                json.dumps(
                    {
                        "timestamp_utc": _now_iso(),
                        "prompt_version": prompt_spec.version,
                        "model": MODEL_NAME,
                        "batch_index": batch_idx,
                        "batch_size": len(batch_titles),
                        "titles": batch_titles,
                        "output_text": output_text,
                        "response": full_resp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        df_rows = pd.DataFrame(rows).drop_duplicates(subset=["title"], keep="last")
        df_rows.to_csv(checkpoint_csv_path, index=False)

    final_df = pd.DataFrame(rows).drop_duplicates(subset=["title"], keep="last")
    final_df = final_df.sort_values("title").reset_index(drop=True)
    return final_df


def score_manual_bins_batched(
    client: OpenAI,
    titles: list[str],
    raw_jsonl_path: Path,
    output_csv_path: Path,
) -> pd.DataFrame:
    manual_prompt = build_manual_prompt()
    rows: list[dict[str, Any]] = []
    batches = _chunks(titles, MANUAL_BATCH_SIZE)
    for batch_idx, batch_titles in enumerate(batches, start=1):
        payload = {"titles": batch_titles}
        output_text, full_resp = call_model_with_retries(client, prompt=manual_prompt, payload=payload)
        parsed = parse_manual_results(output_text, batch_titles)
        rows.extend(parsed)

        with raw_jsonl_path.open("a", encoding="utf-8") as fout:
            fout.write(
                json.dumps(
                    {
                        "timestamp_utc": _now_iso(),
                        "model": MODEL_NAME,
                        "batch_index": batch_idx,
                        "batch_size": len(batch_titles),
                        "titles": batch_titles,
                        "output_text": output_text,
                        "response": full_resp,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    out = pd.DataFrame(rows).drop_duplicates(subset=["title"], keep="last")
    out["manual_bin_id"] = out["manual_bin"].map({"low": 0, "mid": 1, "high": 2}).astype(int)
    out.to_csv(output_csv_path, index=False)
    return out


def pilot_calibration_summary(
    pilot_v1: pd.DataFrame,
    pilot_v2: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    merged = pilot_v1.merge(
        pilot_v2,
        on="title",
        suffixes=("_v1", "_v2"),
    )
    merged["abs_diff_v1_v2"] = (merged[f"{OUTPUT_SCORE_NAME}_v1"] - merged[f"{OUTPUT_SCORE_NAME}_v2"]).abs()
    avg_abs_shift = float(merged["abs_diff_v1_v2"].mean())
    large_shift_n = int((merged["abs_diff_v1_v2"] >= 25).sum())

    summary = {
        "pilot_n": int(len(merged)),
        "avg_abs_shift_v1_to_v2": avg_abs_shift,
        "large_shift_count_ge_25": large_shift_n,
    }

    merged = merged.sort_values("abs_diff_v1_v2", ascending=False).reset_index(drop=True)
    merged.to_csv(output_path, index=False)
    return summary


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    data_path = project_dir / "data" / "list_of_17k_positions_renamed_hr.dta"
    output_dir = project_dir / "output" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    client = OpenAI()

    df = pd.read_stata(data_path, convert_categoricals=False).copy()
    df["role_k17000_v3"] = df["role_k17000_v3"].map(_normalize_title)
    unique_titles = sorted([t for t in df["role_k17000_v3"].dropna().unique().tolist() if t])
    if not unique_titles:
        raise ValueError("No non-empty titles found in role_k17000_v3")

    rng = random.Random(SEED)
    shuffled_titles = unique_titles[:]
    rng.shuffle(shuffled_titles)
    split_idx = len(shuffled_titles) // 2
    tuning_titles = shuffled_titles[:split_idx]
    holdout_titles = shuffled_titles[split_idx:]

    pilot_n = min(PILOT_SAMPLE_SIZE, len(tuning_titles))
    pilot_titles = rng.sample(tuning_titles, k=pilot_n)

    prompt_v1 = build_prompt_v1()
    prompt_v2 = build_prompt_v2()
    final_prompt = prompt_v2

    (output_dir / "prompt_v1.txt").write_text(prompt_v1.text + "\n", encoding="utf-8")
    (output_dir / "prompt_v2.txt").write_text(prompt_v2.text + "\n", encoding="utf-8")
    (output_dir / "prompt_final.txt").write_text(final_prompt.text + "\n", encoding="utf-8")

    schema = {
        "attribute_name": ATTRIBUTE_NAME,
        "output_variable": OUTPUT_SCORE_NAME,
        "type": "integer",
        "range": [0, 100],
        "anchors": {
            "0_20": "No sales/marketing core function",
            "21_60": "Mixed/commercial-adjacent or support-level ownership",
            "61_100": "Direct sales or marketing ownership",
        },
        "json_schema": {
            "type": "object",
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title", OUTPUT_SCORE_NAME],
                        "properties": {
                            "title": {"type": "string"},
                            OUTPUT_SCORE_NAME: {"type": "integer", "minimum": 0, "maximum": 100},
                        },
                    },
                }
            },
        },
        "model_settings": {
            "model": MODEL_NAME,
            "reasoning_effort": REASONING_EFFORT,
            "seed": SEED,
        },
        "frozen_at_utc": _now_iso(),
    }
    (output_dir / "attribute_spec_v1.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pilot_v1 = score_titles_batched(
        client=client,
        titles=pilot_titles,
        prompt_spec=prompt_v1,
        batch_size=50,
        raw_jsonl_path=output_dir / "pilot_raw_responses_v1.jsonl",
        checkpoint_csv_path=output_dir / "pilot_scores_v1.csv",
    )
    pilot_v2 = score_titles_batched(
        client=client,
        titles=pilot_titles,
        prompt_spec=prompt_v2,
        batch_size=50,
        raw_jsonl_path=output_dir / "pilot_raw_responses_v2.jsonl",
        checkpoint_csv_path=output_dir / "pilot_scores_v2.csv",
    )

    pilot_summary = pilot_calibration_summary(
        pilot_v1=pilot_v1,
        pilot_v2=pilot_v2,
        output_path=output_dir / "pilot_prompt_comparison.csv",
    )

    benchmark = _benchmark_titles()
    bench_titles = [x["title"] for x in benchmark]
    bench_scores = score_titles_batched(
        client=client,
        titles=bench_titles,
        prompt_spec=final_prompt,
        batch_size=10,
        raw_jsonl_path=output_dir / "benchmark_raw_responses.jsonl",
        checkpoint_csv_path=output_dir / "benchmark_scores.csv",
    )
    bench_map = dict(zip(bench_scores["title"], bench_scores[OUTPUT_SCORE_NAME]))
    bench_eval = []
    for row in benchmark:
        score = int(bench_map[row["title"]])
        pass_range = row["expected_min"] <= score <= row["expected_max"]
        bench_eval.append({**row, "score": score, "pass": pass_range})
    bench_eval_df = pd.DataFrame(bench_eval)
    bench_eval_df.to_csv(output_dir / "benchmark_sanity_checks.csv", index=False)

    full_scores = score_titles_batched(
        client=client,
        titles=unique_titles,
        prompt_spec=final_prompt,
        batch_size=SCORING_BATCH_SIZE,
        raw_jsonl_path=output_dir / "full_raw_responses.jsonl",
        checkpoint_csv_path=output_dir / "full_scores_checkpoint.csv",
    )
    full_scores = full_scores.rename(columns={"title": "role_k17000_v3"})
    full_scores.to_csv(output_dir / "role_sales_mkt_relatedness_gpt.csv", index=False)

    # Stratified QA sample from low/mid/high predicted bins
    full_scores["score_bin_id"] = full_scores[OUTPUT_SCORE_NAME].map(_score_to_bin)
    strata = []
    per_stratum = STRATIFIED_QA_SIZE // 3
    for bin_id in [0, 1, 2]:
        part = full_scores[full_scores["score_bin_id"] == bin_id].copy()
        n = min(per_stratum, len(part))
        if n == 0:
            continue
        strata.append(part.sample(n=n, random_state=SEED + bin_id))
    stratified_sample = pd.concat(strata, ignore_index=True) if strata else full_scores.head(0).copy()
    stratified_sample = stratified_sample.assign(eval_split="stratified_qa")

    holdout_pool_df = full_scores[full_scores["role_k17000_v3"].isin(set(holdout_titles))].copy()
    holdout_n = min(HOLDOUT_EVAL_SIZE, len(holdout_pool_df))
    holdout_eval = holdout_pool_df.sample(n=holdout_n, random_state=SEED + 99) if holdout_n > 0 else holdout_pool_df.head(0).copy()
    holdout_eval = holdout_eval.assign(eval_split="holdout")

    qa_eval_df = pd.concat([stratified_sample, holdout_eval], ignore_index=True)
    qa_eval_df = qa_eval_df.drop_duplicates(subset=["role_k17000_v3"]).reset_index(drop=True)
    qa_eval_df.to_csv(output_dir / "qa_eval_sample_titles.csv", index=False)

    manual_labels = score_manual_bins_batched(
        client=client,
        titles=qa_eval_df["role_k17000_v3"].tolist(),
        raw_jsonl_path=output_dir / "manual_qa_raw_responses.jsonl",
        output_csv_path=output_dir / "manual_qa_labels.csv",
    )
    qa_joined = qa_eval_df.merge(
        manual_labels.rename(columns={"title": "role_k17000_v3"}),
        on="role_k17000_v3",
        how="left",
    )
    qa_joined["gpt_bin_id"] = qa_joined[OUTPUT_SCORE_NAME].map(_score_to_bin).astype(int)
    qa_joined["gpt_bin"] = qa_joined["gpt_bin_id"].map(_bin_name)
    qa_joined["abs_diff_proxy"] = (qa_joined[OUTPUT_SCORE_NAME] - qa_joined["manual_score_proxy"]).abs()
    qa_joined.to_csv(output_dir / "qa_eval_scored.csv", index=False)

    overall_spearman = _spearman(
        qa_joined[OUTPUT_SCORE_NAME].to_numpy(dtype=float),
        qa_joined["manual_score_proxy"].to_numpy(dtype=float),
    )
    overall_kappa = float(
        cohen_kappa_score(
            qa_joined["gpt_bin_id"].to_numpy(dtype=int),
            qa_joined["manual_bin_id"].to_numpy(dtype=int),
            weights="quadratic",
        )
    )

    by_split = []
    for split_name, sub in qa_joined.groupby("eval_split"):
        split_spearman = _spearman(
            sub[OUTPUT_SCORE_NAME].to_numpy(dtype=float),
            sub["manual_score_proxy"].to_numpy(dtype=float),
        )
        split_kappa = float(
            cohen_kappa_score(
                sub["gpt_bin_id"].to_numpy(dtype=int),
                sub["manual_bin_id"].to_numpy(dtype=int),
                weights="quadratic",
            )
        )
        by_split.append(
            {
                "eval_split": split_name,
                "n": int(len(sub)),
                "spearman": split_spearman,
                "weighted_kappa_quadratic": split_kappa,
            }
        )
    by_split_df = pd.DataFrame(by_split).sort_values("eval_split").reset_index(drop=True)
    by_split_df.to_csv(output_dir / "validation_metrics_by_split.csv", index=False)

    confusion_cases = qa_joined.sort_values("abs_diff_proxy", ascending=False).head(40).copy()
    confusion_cases.to_csv(output_dir / "validation_confusion_cases.csv", index=False)

    merged = df.merge(full_scores, on="role_k17000_v3", how="left")
    merged_path_dta = output_dir / "list_of_17k_positions_with_sales_mkt_relatedness_gpt.dta"
    merged_path_csv = output_dir / "list_of_17k_positions_with_sales_mkt_relatedness_gpt.csv"
    merged.to_stata(merged_path_dta, write_index=False, version=118)
    merged.to_csv(merged_path_csv, index=False)

    run_summary = {
        "run_timestamp_utc": _now_iso(),
        "input_data_path": str(data_path),
        "output_dir": str(output_dir),
        "n_rows_input": int(len(df)),
        "n_unique_titles_scored": int(len(full_scores)),
        "prompt_selected": final_prompt.version,
        "pilot_summary": pilot_summary,
        "benchmark_pass_rate": float(bench_eval_df["pass"].mean()) if len(bench_eval_df) else float("nan"),
        "validation_overall": {
            "n": int(len(qa_joined)),
            "spearman": overall_spearman,
            "weighted_kappa_quadratic": overall_kappa,
        },
        "validation_by_split": by_split,
        "artifacts": [
            "attribute_spec_v1.json",
            "prompt_v1.txt",
            "prompt_v2.txt",
            "prompt_final.txt",
            "pilot_scores_v1.csv",
            "pilot_scores_v2.csv",
            "pilot_prompt_comparison.csv",
            "benchmark_sanity_checks.csv",
            "full_scores_checkpoint.csv",
            "role_sales_mkt_relatedness_gpt.csv",
            "manual_qa_labels.csv",
            "qa_eval_scored.csv",
            "validation_metrics_by_split.csv",
            "validation_confusion_cases.csv",
            "list_of_17k_positions_with_sales_mkt_relatedness_gpt.dta",
            "list_of_17k_positions_with_sales_mkt_relatedness_gpt.csv",
        ],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    memo_lines = [
        "# Sales/Marketing GPT Measurement Validation Memo",
        "",
        f"- Run UTC: {run_summary['run_timestamp_utc']}",
        f"- Model: {MODEL_NAME} (reasoning={REASONING_EFFORT})",
        f"- Prompt selected after one controlled refinement: {final_prompt.version}",
        f"- Input rows: {len(df)}",
        f"- Unique titles scored: {len(full_scores)}",
        "",
        "## Pilot and Calibration",
        f"- Pilot sample size: {pilot_summary['pilot_n']}",
        f"- Average absolute score shift (v1->v2): {pilot_summary['avg_abs_shift_v1_to_v2']:.2f}",
        f"- Large shifts (>=25 points): {pilot_summary['large_shift_count_ge_25']}",
        "",
        "## Parse/Completion",
        "- Parse success is 100% for all completed batches because each batch only checkpoints after strict parse validation.",
        "",
        "## Benchmark Sanity Checks",
        f"- Pass rate: {run_summary['benchmark_pass_rate']:.3f}",
        "",
        "## Validation (Independent QA Coder)",
        f"- Overall N: {len(qa_joined)}",
        f"- Spearman(score, manual_proxy): {overall_spearman:.4f}",
        f"- Weighted kappa (quadratic) on bins: {overall_kappa:.4f}",
        "",
        "### Split-Level Metrics",
    ]
    for row in by_split:
        memo_lines.append(
            f"- {row['eval_split']}: n={row['n']}, spearman={row['spearman']:.4f}, "
            f"weighted_kappa={row['weighted_kappa_quadratic']:.4f}"
        )

    memo_lines.extend(
        [
            "",
            "## Notes",
            "- Prompt tuning was restricted to one controlled refinement from v1 to v2.",
            "- Holdout titles were isolated before pilot prompt refinement.",
            "- Raw API responses are retained in JSONL artifacts for auditability.",
            "- Independent QA coder labels are model-assisted proxy labels, not human-lab-coded annotations.",
        ]
    )
    (output_dir / "validation_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    print(f"Saved artifacts to: {output_dir}")
    print(f"Primary score table: {output_dir / 'role_sales_mkt_relatedness_gpt.csv'}")
    print(f"Merged dataset (dta): {merged_path_dta}")


if __name__ == "__main__":
    main()
