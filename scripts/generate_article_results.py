#!/usr/bin/env python3
"""Generate publication-ready tables and charts from report JSON files."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from lxml import etree


plt: Any = None


# Editorial palette for publication figures (colorblind-friendly, softer contrast)
MODEL_PALETTE = [
    "#E56B6F",  # warm coral
    "#EAAC8B",  # soft apricot
    "#3A7CA5",  # steel blue
    "#2A9D8F",  # teal
    "#355070",  # deep slate blue
    "#6D597A",  # muted violet
    "#B56576",  # dusty rose
    "#264653",  # deep teal
]


REPORT_PATTERN = re.compile(r"^(?P<model>.+?)_(?P<prompt>Prompt_\d+)_(?P<document>TR\d+_p\d+-\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tables and diagrams by document, element, model and prompt.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing evaluation report JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("article_results"),
        help="Output directory for generated tables and figures.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("data/gt"),
        help="Directory containing ground-truth TEI XML files used to infer entry support.",
    )
    return parser.parse_args()


def natural_sort_key(text: str) -> list[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def prompt_label(prompt: str) -> str:
    match = re.match(r"^Prompt_(\d+)$", prompt)
    if match:
        return f"P{match.group(1)}"
    return prompt


def safe_metric(value: Any) -> Any:
    if value is None:
        return ""
    return float(value)


def infer_gold_entry_counts(document: str, gt_dir: Path, cache: dict[str, dict[str, int]]) -> dict[str, int]:
    if document in cache:
        return cache[document]

    xml_path = gt_dir / f"{document}.xml"
    if not xml_path.exists():
        counts = {"mainEntry": 0, "relatedEntry": 0}
        cache[document] = counts
        return counts

    root = etree.fromstring(xml_path.read_text(encoding="utf-8").encode("utf-8"))
    ns = {"tei": root.nsmap.get(None)} if None in root.nsmap else {}

    if ns:
        main_count = len(root.xpath(".//tei:entry[@type='mainEntry']", namespaces=ns))
        related_count = len(root.xpath(".//tei:entry[@type='relatedEntry']", namespaces=ns))
    else:
        main_count = len(root.xpath(".//entry[@type='mainEntry']"))
        related_count = len(root.xpath(".//entry[@type='relatedEntry']"))

    counts = {"mainEntry": main_count, "relatedEntry": related_count}
    cache[document] = counts
    return counts


def format_score_value(key: str, value: Any) -> Any:
    if value == "" or value is None:
        return ""
    if (
        "precision" in key
        or "recall" in key
        or key == "f1"
        or key.endswith("_f1")
        or key.startswith("mean_f1_")
        or re.match(r"^P\d+$", key) is not None
    ):
        return f"{float(value):.2f}"
    return value


def load_reports(reports_dir: Path, gt_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    entry_rows: list[dict[str, Any]] = []
    tag_rows: list[dict[str, Any]] = []

    models: set[str] = set()
    prompts: set[str] = set()
    documents: set[str] = set()
    tags: set[str] = set()
    gold_entry_cache: dict[str, dict[str, int]] = {}

    for report_file in sorted(reports_dir.glob("*.json")):
        match = REPORT_PATTERN.match(report_file.name)
        if not match:
            continue

        model = match.group("model")
        prompt = match.group("prompt")
        document = match.group("document")

        models.add(model)
        prompts.add(prompt)
        documents.add(document)

        with report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", {})
        for element in ("mainEntry", "relatedEntry"):
            metrics = entries.get(element, {})
            inferred_gold = infer_gold_entry_counts(document, gt_dir, gold_entry_cache)
            entry_rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "document": document,
                    "element": element,
                    "precision": safe_metric(metrics.get("precision")),
                    "recall": safe_metric(metrics.get("recall")),
                    "f1": safe_metric(metrics.get("f1")),
                    "pred_count": metrics.get("pred_count", ""),
                    "gold_count": metrics.get("gold_count", inferred_gold.get(element, "")),
                    "source_file": report_file.name,
                }
            )

        for tag, metrics in data.get("tags", {}).items():
            tags.add(tag)
            tag_rows.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "document": document,
                    "tag": tag,
                    "precision": safe_metric(metrics.get("precision")),
                    "recall": safe_metric(metrics.get("recall")),
                    "f1": safe_metric(metrics.get("f1")),
                    "pred_count": metrics.get("pred_count", ""),
                    "gold_count": metrics.get("gold_count", ""),
                    "source_file": report_file.name,
                }
            )

    return (
        entry_rows,
        tag_rows,
        sorted(models, key=natural_sort_key),
        sorted(prompts, key=natural_sort_key),
        sorted(documents, key=natural_sort_key),
        sorted(tags, key=natural_sort_key),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        formatted_rows = []
        for row in rows:
            formatted_rows.append({k: format_score_value(k, row.get(k, "")) for k in fieldnames})
        writer.writerows(formatted_rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    table_rows = [[str(format_score_value(col, row.get(col, ""))) for col in columns] for row in rows]

    widths = []
    for idx, col in enumerate(columns):
        max_cell = max([len(col)] + [len(r[idx]) for r in table_rows])
        widths.append(max_cell)

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    header = fmt_row(columns)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"

    lines = [header, sep]
    for row in table_rows:
        lines.append(fmt_row(row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pivot_entries_by_doc(entry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in entry_rows:
        key = (row["document"], row["model"], row["prompt"])
        if key not in index:
            index[key] = {
                "document": row["document"],
                "model": row["model"],
                "prompt": row["prompt"],
                "mainEntry_precision": "",
                "mainEntry_recall": "",
                "mainEntry_f1": "",
                "relatedEntry_precision": "",
                "relatedEntry_recall": "",
                "relatedEntry_f1": "",
            }

        prefix = row["element"]
        index[key][f"{prefix}_precision"] = row["precision"]
        index[key][f"{prefix}_recall"] = row["recall"]
        index[key][f"{prefix}_f1"] = row["f1"]

    return sorted(index.values(), key=lambda x: (natural_sort_key(x["document"]), natural_sort_key(x["model"]), natural_sort_key(x["prompt"])))


def mean_f1_by_model_prompt(entry_rows: list[dict[str, Any]], tag_rows: list[dict[str, Any]], tags: list[str]) -> list[dict[str, Any]]:
    collector: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in entry_rows:
        if row["f1"] != "":
            collector[(row["model"], row["prompt"])][row["element"]].append(float(row["f1"]))

    for row in tag_rows:
        if row["f1"] != "":
            collector[(row["model"], row["prompt"])][f"tag:{row['tag']}"] .append(float(row["f1"]))

    out_rows: list[dict[str, Any]] = []
    for (model, prompt), by_element in sorted(collector.items(), key=lambda x: (natural_sort_key(x[0][0]), natural_sort_key(x[0][1]))):
        row: dict[str, Any] = {"model": model, "prompt": prompt}
        for element in ["mainEntry", "relatedEntry"] + [f"tag:{t}" for t in tags]:
            values = by_element.get(element, [])
            row[f"mean_f1_{element}"] = round(sum(values) / len(values), 6) if values else ""
        out_rows.append(row)

    return out_rows


def mean_scores_overall(entry_rows: list[dict[str, Any]], tag_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collector: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "precision": [],
            "recall": [],
            "f1": [],
            "documents": set(),
            "support": 0,
        }
    )

    for row in entry_rows:
        key = (row["model"], row["prompt"], "entry", row["element"])
        bucket = collector[key]
        if row["precision"] != "":
            bucket["precision"].append(float(row["precision"]))
        if row["recall"] != "":
            bucket["recall"].append(float(row["recall"]))
        if row["f1"] != "":
            bucket["f1"].append(float(row["f1"]))
        bucket["documents"].add(row["document"])
        gold_count = row.get("gold_count", "")
        if gold_count != "" and gold_count is not None:
            bucket["support"] += int(gold_count)

    for row in tag_rows:
        key = (row["model"], row["prompt"], "tag", row["tag"])
        bucket = collector[key]
        if row["precision"] != "":
            bucket["precision"].append(float(row["precision"]))
        if row["recall"] != "":
            bucket["recall"].append(float(row["recall"]))
        if row["f1"] != "":
            bucket["f1"].append(float(row["f1"]))
        bucket["documents"].add(row["document"])
        gold_count = row.get("gold_count", "")
        if gold_count != "" and gold_count is not None:
            bucket["support"] += int(gold_count)

    out_rows: list[dict[str, Any]] = []
    for (model, prompt, group_type, group_name), values in sorted(
        collector.items(),
        key=lambda x: (
            natural_sort_key(x[0][0]),
            natural_sort_key(x[0][1]),
            0 if x[0][2] == "entry" else 1,
            natural_sort_key(x[0][3]),
        ),
    ):
        support = values["support"]
        if support == 0:
            support = len(values["documents"])

        out_rows.append(
            {
                "model": model,
                "prompt": prompt,
                "group_type": group_type,
                "group_name": group_name,
                "mean_precision": round(sum(values["precision"]) / len(values["precision"]), 6)
                if values["precision"]
                else "",
                "mean_recall": round(sum(values["recall"]) / len(values["recall"]), 6)
                if values["recall"]
                else "",
                "mean_f1": round(sum(values["f1"]) / len(values["f1"]), 6)
                if values["f1"]
                else "",
                "support": support,
                "n_documents": len(values["documents"]),
            }
        )

    return out_rows


def build_entry_prompt_matrix_rows(
    overall_rows: list[dict[str, Any]],
    target_entry: str,
    models: list[str],
    prompts: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    prompt_columns = [prompt_label(p) for p in prompts]

    lookup: dict[tuple[str, str], Any] = {}
    support_lookup: dict[str, int] = {}
    for row in overall_rows:
        if row.get("group_type") == "entry" and row.get("group_name") == target_entry:
            lookup[(row["model"], row["prompt"])] = row.get("mean_f1", "")
            support = row.get("support", "")
            if support != "" and support is not None:
                support_lookup[row["model"]] = max(int(support_lookup.get(row["model"], 0)), int(support))

    matrix_rows: list[dict[str, Any]] = []
    for model in models:
        out_row: dict[str, Any] = {
            "model": model,
            "support": support_lookup.get(model, ""),
        }
        for prompt in prompts:
            out_row[prompt_label(prompt)] = lookup.get((model, prompt), "")
        matrix_rows.append(out_row)

    return matrix_rows, ["support"] + prompt_columns


def build_prompt1_document_rows_by_model_element(
    entry_rows: list[dict[str, Any]],
    models: list[str],
    prompt: str = "Prompt_1",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    filtered = [r for r in entry_rows if r["prompt"] == prompt]

    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in filtered:
        element = row["element"]
        key = (row["model"], element, row["document"])
        if key not in index:
            index[key] = {
                "document": row["document"],
                "support": row.get("gold_count", ""),
                "precision": "",
                "recall": "",
                "f1": "",
            }

        index[key]["precision"] = row["precision"]
        index[key]["recall"] = row["recall"]
        index[key]["f1"] = row["f1"]

    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model in models:
        out[model] = {}
        for element in ["mainEntry", "relatedEntry"]:
            model_element_rows = [
                row for (m, e, _doc), row in index.items() if m == model and e == element
            ]
            out[model][element] = sorted(model_element_rows, key=lambda r: natural_sort_key(r["document"]))

    return out


def build_matrix(rows: list[dict[str, Any]],
                 documents: list[str],
                 model_prompts: list[str],
                 value_getter: Callable[[dict[str, Any]], Any]) -> list[list[float]]:
    lookup: dict[tuple[str, str], float] = {}
    for row in rows:
        mp = f"{row['model']} | {row['prompt']}"
        val = value_getter(row)
        if val != "":
            lookup[(row["document"], mp)] = float(val)

    matrix: list[list[float]] = []
    for doc in documents:
        matrix.append([lookup.get((doc, mp), math.nan) for mp in model_prompts])
    return matrix


def plot_heatmap(ax: Any, matrix: list[list[float]], x_labels: list[str], y_labels: list[str], title: str) -> None:
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                continue
            color = "white" if value < 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")


def plot_entries_heatmaps(entry_rows: list[dict[str, Any]], documents: list[str], model_prompts: list[str], out_path: Path) -> None:
    main_rows = [r for r in entry_rows if r["element"] == "mainEntry"]
    rel_rows = [r for r in entry_rows if r["element"] == "relatedEntry"]

    main_matrix = build_matrix(main_rows, documents, model_prompts, lambda x: x["f1"])
    rel_matrix = build_matrix(rel_rows, documents, model_prompts, lambda x: x["f1"])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(documents) * 0.45)), constrained_layout=True)
    plot_heatmap(axes[0], main_matrix, model_prompts, documents, "F1 by Document - mainEntry")
    plot_heatmap(axes[1], rel_matrix, model_prompts, documents, "F1 by Document - relatedEntry")
    fig.suptitle("Entry-level performance by document/model/prompt", fontsize=13)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_tags_heatmaps(tag_rows: list[dict[str, Any]], tags: list[str], documents: list[str], model_prompts: list[str], out_path: Path) -> None:
    if not tags:
        return

    n_cols = 2
    n_rows = math.ceil(len(tags) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, max(5, n_rows * max(3.5, len(documents) * 0.22))),
        constrained_layout=True,
    )

    if hasattr(axes, "ravel"):
        axes = list(axes.ravel())
    else:
        axes = [axes]

    for idx, tag in enumerate(tags):
        ax = axes[idx]
        selected = [r for r in tag_rows if r["tag"] == tag]
        matrix = build_matrix(selected, documents, model_prompts, lambda x: x["f1"])
        plot_heatmap(ax, matrix, model_prompts, documents, f"F1 by Document - tag:{tag}")

    for idx in range(len(tags), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Tag-level performance by document/model/prompt", fontsize=13)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _bar_chart_single(
    ax: Any,
    lookup: dict,
    keys: list,
    key_labels: list[str],
    models: list[str],
    title: str,
    ylabel: str,
    x_rotation: int = 30,
    show_values: bool = True,
) -> None:
    """Draw a grouped bar chart on *ax* (one bar per model per X group)."""
    n_models = len(models)
    width = 0.8 / max(1, n_models)
    x_positions = list(range(len(keys)))

    for idx, model in enumerate(models):
        y_values = [lookup.get((model, k), math.nan) for k in keys]
        shifted = [x + (idx - (n_models - 1) / 2) * width for x in x_positions]
        bars = ax.bar(
            shifted,
            y_values,
            width=width,
            label=model,
            color=MODEL_PALETTE[idx % len(MODEL_PALETTE)],
        )
        if show_values:
            for bar, val in zip(bars, y_values):
                if not math.isnan(val):
                    label_y = min(val, 1.0) - 0.02 if val > 0.90 else min(val, 1.0) + 0.01
                    label_va = "top" if val > 0.90 else "bottom"
                    label_color = "white" if val > 0.90 else "black"
                    label_weight = "bold" if val > 0.90 else "normal"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        f"{val:.2f}",
                        ha="center",
                        va=label_va,
                        fontsize=8,
                        rotation=45,
                        color=label_color,
                        fontweight=label_weight,
                    )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(key_labels, rotation=x_rotation, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)


def plot_prompt1_document_bars_by_model(
    entry_rows: list[dict[str, Any]],
    models: list[str],
    documents: list[str],
    out_path: Path,
    prompt: str = "Prompt_1",
) -> None:
    """One PNG per entry type: X=documents, bars=models, Y=F1 – Prompt 1."""
    filtered = [r for r in entry_rows if r["prompt"] == prompt]

    for element in ["mainEntry", "relatedEntry"]:
        lookup: dict[tuple[str, str], float] = {}
        for row in filtered:
            if row["element"] == element and row["f1"] != "":
                lookup[(row["model"], row["document"])] = float(row["f1"])

        fig, ax = plt.subplots(figsize=(max(10, len(documents) * 0.9), 5))
        _bar_chart_single(
            ax, lookup, documents, documents, models,
            title=f"{element} – F1 par document (Prompt 1)",
            ylabel="F1",
            x_rotation=30,
            show_values=False,
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(4, len(models)),
            borderaxespad=0,
            fontsize=9,
            title="Modèle",
        )
        fig.tight_layout()
        stem = out_path.stem
        element_path = out_path.with_name(f"{stem}_{element}.png")
        fig.savefig(element_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mean_f1_by_model(
    overall_rows: list[dict[str, Any]],
    models: list[str],
    prompts: list[str],
    out_path: Path,
) -> None:
    """One PNG per entry type: X=prompts, bars=models, Y=mean F1."""
    elements = ["mainEntry", "relatedEntry"]
    plabels = [prompt_label(p) for p in prompts]

    for element in elements:
        lookup: dict[tuple[str, str], float] = {}
        for row in overall_rows:
            if row.get("group_type") == "entry" and row.get("group_name") == element:
                val = row.get("mean_f1", "")
                if val != "":
                    lookup[(row["model"], row["prompt"])] = float(val)

        fig, ax = plt.subplots(figsize=(max(8, len(prompts) * 2.0), 5))
        _bar_chart_single(
            ax, lookup, prompts, plabels, models,
            title=f"{element} – Mean F1 par prompt et par modèle",
            ylabel="Mean F1",
            x_rotation=0,
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=min(4, len(models)),
            borderaxespad=0,
            fontsize=9,
            title="Modèle",
        )
        fig.tight_layout()
        stem = out_path.stem
        element_path = out_path.with_name(f"{stem}_{element}.png")
        fig.savefig(element_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mean_f1_bar(mean_rows: list[dict[str, Any]], tags: list[str], out_path: Path) -> None:
    elements = ["mainEntry", "relatedEntry"] + [f"tag:{t}" for t in tags]
    series_labels = [f"{r['model']} | {r['prompt']}" for r in mean_rows]

    fig, ax = plt.subplots(figsize=(max(10, len(series_labels) * 1.5), 6), constrained_layout=True)

    width = 0.8 / max(1, len(elements))
    x_positions = list(range(len(series_labels)))

    for idx, element in enumerate(elements):
        y_values = []
        for row in mean_rows:
            value = row.get(f"mean_f1_{element}", "")
            y_values.append(float(value) if value != "" else math.nan)

        shifted = [x + (idx - (len(elements) - 1) / 2) * width for x in x_positions]
        ax.bar(
            shifted,
            y_values,
            width=width,
            label=element,
            color=MODEL_PALETTE[idx % len(MODEL_PALETTE)],
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean F1")
    ax.set_title("Mean F1 by element, model and prompt")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(series_labels, rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    global plt

    args = parse_args()

    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    reports_dir = args.reports_dir
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    entry_rows, tag_rows, models, prompts, documents, tags = load_reports(reports_dir, args.gt_dir)
    if not entry_rows and not tag_rows:
        raise SystemExit(f"No valid report JSON files found in {reports_dir}")

    observed_model_prompts = {(r["model"], r["prompt"]) for r in entry_rows}
    observed_model_prompts.update({(r["model"], r["prompt"]) for r in tag_rows})
    model_prompts = [
        f"{model} | {prompt}"
        for model, prompt in sorted(
            observed_model_prompts,
            key=lambda x: (natural_sort_key(x[0]), natural_sort_key(x[1])),
        )
    ]

    write_csv(
        tables_dir / "entries_long.csv",
        entry_rows,
        ["model", "prompt", "document", "element", "precision", "recall", "f1", "source_file"],
    )
    write_csv(
        tables_dir / "tags_long.csv",
        tag_rows,
        ["model", "prompt", "document", "tag", "precision", "recall", "f1", "pred_count", "gold_count", "source_file"],
    )

    entries_pivot_rows = pivot_entries_by_doc(entry_rows)
    write_csv(
        tables_dir / "entries_by_document.csv",
        entries_pivot_rows,
        [
            "document",
            "model",
            "prompt",
            "mainEntry_precision",
            "mainEntry_recall",
            "mainEntry_f1",
            "relatedEntry_precision",
            "relatedEntry_recall",
            "relatedEntry_f1",
        ],
    )

    write_markdown_table(
        tables_dir / "entries_by_document.md",
        entries_pivot_rows,
        [
            "document",
            "model",
            "prompt",
            "mainEntry_precision",
            "mainEntry_recall",
            "mainEntry_f1",
            "relatedEntry_precision",
            "relatedEntry_recall",
            "relatedEntry_f1",
        ],
    )

    tags_for_md = sorted(
        tag_rows,
        key=lambda r: (
            natural_sort_key(r["document"]),
            natural_sort_key(r["model"]),
            natural_sort_key(r["prompt"]),
            natural_sort_key(r["tag"]),
        ),
    )
    write_markdown_table(
        tables_dir / "tags_by_document.md",
        tags_for_md,
        ["document", "model", "prompt", "tag", "precision", "recall", "f1", "pred_count", "gold_count"],
    )

    mean_rows = mean_f1_by_model_prompt(entry_rows, tag_rows, tags)
    mean_fieldnames = ["model", "prompt"] + [f"mean_f1_{e}" for e in ["mainEntry", "relatedEntry"] + [f"tag:{t}" for t in tags]]
    write_csv(tables_dir / "mean_f1_by_model_prompt.csv", mean_rows, mean_fieldnames)
    write_markdown_table(tables_dir / "mean_f1_by_model_prompt.md", mean_rows, mean_fieldnames)

    overall_rows = mean_scores_overall(entry_rows, tag_rows)
    overall_fieldnames = [
        "model",
        "prompt",
        "group_type",
        "group_name",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "support",
        "n_documents",
    ]
    write_csv(tables_dir / "mean_scores_overall_documents.csv", overall_rows, overall_fieldnames)
    write_markdown_table(tables_dir / "mean_scores_overall_documents.md", overall_rows, overall_fieldnames)

    main_matrix_rows, prompt_columns = build_entry_prompt_matrix_rows(
        overall_rows,
        "mainEntry",
        models,
        prompts,
    )
    related_matrix_rows, _ = build_entry_prompt_matrix_rows(
        overall_rows,
        "relatedEntry",
        models,
        prompts,
    )

    matrix_fieldnames = ["model"] + prompt_columns
    write_csv(
        tables_dir / "scores_matrix_mainEntry.csv",
        main_matrix_rows,
        matrix_fieldnames,
    )
    write_csv(
        tables_dir / "scores_matrix_relatedEntry.csv",
        related_matrix_rows,
        matrix_fieldnames,
    )

    matrix_md_path = tables_dir / "scores_matrix_entries.md"
    matrix_md_parts = [
        "## mainEntry (mean F1)",
        "",
    ]
    main_md_path_tmp = tables_dir / "_tmp_main_matrix.md"
    related_md_path_tmp = tables_dir / "_tmp_related_matrix.md"

    write_markdown_table(main_md_path_tmp, main_matrix_rows, matrix_fieldnames)
    write_markdown_table(related_md_path_tmp, related_matrix_rows, matrix_fieldnames)

    matrix_md_parts.append(main_md_path_tmp.read_text(encoding="utf-8").rstrip())
    matrix_md_parts.extend([
        "",
        "## relatedEntry (mean F1)",
        "",
        related_md_path_tmp.read_text(encoding="utf-8").rstrip(),
        "",
    ])
    matrix_md_path.write_text("\n".join(matrix_md_parts), encoding="utf-8")

    main_md_path_tmp.unlink(missing_ok=True)
    related_md_path_tmp.unlink(missing_ok=True)

    prompt1_rows_by_model_element = build_prompt1_document_rows_by_model_element(entry_rows, models, prompt="Prompt_1")
    prompt1_columns = ["document", "support", "precision", "recall", "f1"]

    prompt1_md_path = tables_dir / "scores_by_document_prompt1_by_model.md"
    prompt1_md_parts = []
    for model in models:
        prompt1_md_parts.extend([f"## {model} - Prompt 1", ""])

        for element in ["mainEntry", "relatedEntry"]:
            rows = prompt1_rows_by_model_element.get(model, {}).get(element, [])
            prompt1_md_parts.extend([f"### {element}", ""])

            model_csv_path = tables_dir / f"scores_by_document_prompt1_{model}_{element}.csv"
            write_csv(model_csv_path, rows, prompt1_columns)

            model_tmp_md_path = tables_dir / f"_tmp_scores_by_document_prompt1_{model}_{element}.md"
            write_markdown_table(model_tmp_md_path, rows, prompt1_columns)
            prompt1_md_parts.append(model_tmp_md_path.read_text(encoding="utf-8").rstrip())
            prompt1_md_parts.extend(["", ""])
            model_tmp_md_path.unlink(missing_ok=True)

    prompt1_md_path.write_text("\n".join(prompt1_md_parts).rstrip() + "\n", encoding="utf-8")

    # Remove legacy combined Prompt_1 per-model CSV files kept from older runs.
    for model in models:
        (tables_dir / f"scores_by_document_prompt1_{model}.csv").unlink(missing_ok=True)

    plot_entries_heatmaps(entry_rows, documents, model_prompts, figures_dir / "entries_f1_heatmaps.png")
    plot_tags_heatmaps(tag_rows, tags, documents, model_prompts, figures_dir / "tags_f1_heatmaps.png")
    plot_mean_f1_bar(mean_rows, tags, figures_dir / "mean_f1_by_element_model_prompt.png")
    plot_prompt1_document_bars_by_model(entry_rows, models, documents, figures_dir / "prompt1_f1_by_document_and_model.png")
    plot_mean_f1_by_model(overall_rows, models, prompts, figures_dir / "mean_f1_by_model.png")

    print("Generated article assets:")
    print(f"- Tables: {tables_dir}")
    print(f"- Figures: {figures_dir}")


if __name__ == "__main__":
    main()
