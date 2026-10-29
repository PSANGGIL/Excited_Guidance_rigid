#!/usr/bin/env python3
"""Plot audited XTB/sTDA distributions, separating all successes from top 20."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CALCULATION = "xtb_stda_diffusion_coords_12x1_20261022"
TOP20 = "top20_target_match_rdkit2d_20261022/selected_molecules.csv"
RUNS = {
    "hp1kg": {
        "directory": "generation_stage8_hp1kg_best_e3911_est0_n1000_cfg1_20261022",
        "label": "HP1KG e3911", "condition": "est", "target": 0.0,
        "color": "#268293",
    },
    "stgg": {
        "directory": "generation_stage8_stgg_best_e10_s1_4ev_n1000_cfg1_20261022",
        "label": "STGG e10", "condition": "s1", "target": 4.0,
        "color": "#cf8141",
    },
}
FIELDS = {"s1": "s1_energy_eV", "t1": "t1_energy_eV", "est": "calculated_est_eV"}
LABELS = {"s1": "S1", "t1": "T1", "est": "EST = S1 - T1"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_energies(row):
    values = {key: Decimal(row[field]) for key, field in FIELDS.items()}
    if not all(value.is_finite() for value in values.values()):
        raise ValueError("Nonfinite energy")
    if values["s1"] - values["t1"] != values["est"]:
        raise ValueError("EST disagrees with S1 - T1")


def load_data(root):
    data, source_hashes = {}, {}
    for name, spec in RUNS.items():
        run = root / "runs" / spec["directory"]
        calculation = run / CALCULATION
        paths = [run / "manifest.json", calculation / "protocol.json",
                 calculation / "status.json", calculation / "results.csv",
                 calculation / TOP20]
        for path in paths:
            source_hashes[str(path)] = sha256(path)
        manifest = json.loads(paths[0].read_text())
        protocol = json.loads(paths[1].read_text())
        if json.loads(paths[2].read_text())["phase"] != "complete":
            raise ValueError(f"{name}: calculation is not complete")
        if manifest["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint hash mismatch")
        target = Decimal(str(spec["target"]))
        target_key = f"target_{spec['condition']}_eV"
        if any(Decimal(str(doc[target_key])) != target for doc in (manifest, protocol)):
            raise ValueError(f"{name}: target mismatch")
        results = read_csv(paths[3])
        if len({row["generation_index"] for row in results}) != len(results):
            raise ValueError(f"{name}: duplicated generation indices")
        success = [row for row in results if row["status"] == "ok"]
        if not success:
            raise ValueError(f"{name}: no successful calculations")
        indexed = {row["generation_index"]: row for row in success}
        for row in success:
            check_energies(row)
            if row["condition_name"] != spec["condition"] or Decimal(row[target_key]) != target:
                raise ValueError(f"{name}: result condition mismatch")
            for key in ("xtb_returncode", "xtb4stda_returncode", "singlet_returncode", "triplet_returncode"):
                if int(row[key]) != 0:
                    raise ValueError(f"{name}: nonzero return code in success row")
        selected = read_csv(paths[4])
        if len(selected) != 20 or len({row["generation_index"] for row in selected}) != 20:
            raise ValueError(f"{name}: expected 20 different selected indices")
        if len({row["canonical_smiles"] for row in selected}) != 20:
            raise ValueError(f"{name}: duplicated graph in top 20")
        for row in selected:
            check_energies(row)
            source = indexed[row["generation_index"]]
            if any(Decimal(row[field]) != Decimal(source[field]) for field in FIELDS.values()):
                raise ValueError(f"{name}: selected energy differs from full result")
            if row["db_member"].lower() != "false":
                raise ValueError(f"{name}: DB member in top 20")
        data[name] = {"all_success": success, "top20": selected}
    return data, source_hashes


def values_for(data, model, group, metric):
    return np.array([float(row[FIELDS[metric]]) for row in data[model][group]], dtype=float)


def coarse_edges(values, width):
    low = int(np.floor(np.min(values) / width))
    high = max(low + 1, int(np.ceil(np.max(values) / width)))
    return np.arange(low, high + 1, dtype=float) * width


def bin_edges(data):
    energies = np.concatenate([values_for(data, name, "all_success", metric)
                               for name in RUNS for metric in ("s1", "t1")])
    # Common S1/T1 bins and limits across models and subsets.
    common_energy = coarse_edges(np.append(energies, [0.0, 4.0]), 0.5)
    common_est = coarse_edges(np.concatenate([
        values_for(data, name, "all_success", "est") for name in RUNS]), 0.25)
    result = {}
    for name in RUNS:
        for group in ("all_success", "top20"):
            for metric in FIELDS:
                if metric != "est":
                    edges = common_energy.copy()
                elif group == "all_success":
                    edges = common_est.copy()
                elif name == "hp1kg":
                    # STDA reports to 0.001 eV; center one bin on each reported value.
                    vals = values_for(data, name, group, metric)
                    low, high = int(round(vals.min() * 1000)), int(round(vals.max() * 1000))
                    edges = (np.arange(low, high + 2, dtype=float) - 0.5) / 1000.0
                else:
                    edges = coarse_edges(np.append(values_for(data, name, group, metric), 0.0), 0.1)
                result[(name, group, metric)] = edges
    return result


def draw_panel(ax, data, edges, name, group, metric):
    spec = RUNS[name]
    vals = values_for(data, name, group, metric)
    bins = edges[(name, group, metric)]
    counts, _ = np.histogram(vals, bins=bins)
    if int(counts.sum()) != len(vals):
        raise ValueError(f"Histogram loses observations: {name}/{group}/{metric}")
    ax.hist(vals, bins=bins, color=spec["color"], edgecolor="white", linewidth=1.1, alpha=0.9)
    ax.axvline(vals.mean(), color="#183245", linestyle="--", linewidth=1.8, label="Mean")
    if metric == spec["condition"]:
        ax.axvline(spec["target"], color="#b82c3b", linewidth=1.7,
                   linestyle=":", label=f"Target = {spec['target']:g} eV")
    elif metric == "est":
        ax.axvline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    zoomed = name == "hp1kg" and group == "top20" and metric == "est"
    precision = 5 if zoomed else 3
    zoom_label = " | zoomed x-axis" if zoomed else ""
    ax.set_title(
        f"{spec['label']} | {LABELS[metric]}{zoom_label}\n"
        f"n = {len(vals)}    mean = {vals.mean():.{precision}f}    SD = {vals.std(ddof=0):.{precision}f} eV",
        fontsize=11.7, pad=13, linespacing=1.7)
    ax.set_xlabel(f"{LABELS[metric]} (eV)", labelpad=8)
    ax.set_ylabel("Molecule count", labelpad=8)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, max(1.0, counts.max()) * 1.3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    if zoomed:
        ax.xaxis.set_major_locator(MultipleLocator(0.001))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/stage8_energy_histograms_20261022")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    data, source_hashes = load_data(args.root.resolve())
    edges = bin_edges(data)
    stats, histogram_rows, value_rows = [], [], []
    for name in RUNS:
        for group in ("all_success", "top20"):
            for row in data[name][group]:
                value_rows.append(dict(model=name, group=group, generation_index=row["generation_index"],
                                       **{field: row[field] for field in FIELDS.values()}))
            for metric in FIELDS:
                vals = values_for(data, name, group, metric)
                bins = edges[(name, group, metric)]
                counts, _ = np.histogram(vals, bins=bins)
                if int(counts.sum()) != len(vals):
                    raise ValueError("Histogram coverage check failed")
                stats.append(dict(model=name, group=group, metric=metric, n=len(vals),
                                  mean_eV=float(vals.mean()), std_ddof0_eV=float(vals.std(ddof=0)),
                                  min_eV=float(vals.min()), max_eV=float(vals.max()),
                                  bin_width_eV=float(bins[1] - bins[0]), bin_count=len(counts)))
                for i, count in enumerate(counts):
                    histogram_rows.append(dict(model=name, group=group, metric=metric,
                        left_eV=float(bins[i]), right_eV=float(bins[i + 1]),
                        right_inclusive=i == len(counts) - 1, count=int(count)))
    # All inputs are fully validated before creating any output files.
    output.mkdir(parents=True, exist_ok=False)
    (output / "individual").mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    files = []
    for group in ("all_success", "top20"):
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        fig.subplots_adjust(left=0.065, right=0.98, bottom=0.10, top=0.83, hspace=0.62, wspace=0.28)
        title = "All successful calculations" if group == "all_success" else "Top 20 target-matched novel unique molecules"
        fig.suptitle(f"XTB/sTDA energy distributions | {title}", fontsize=20, y=0.975)
        fig.text(0.5, 0.925, "HP1KG target: EST = 0 eV     |     STGG target: S1 = 4.0 eV",
                 ha="center", fontsize=13)
        for i, name in enumerate(RUNS):
            for j, metric in enumerate(FIELDS):
                draw_panel(axes[i, j], data, edges, name, group, metric)
                single, single_ax = plt.subplots(figsize=(8, 6))
                draw_panel(single_ax, data, edges, name, group, metric)
                single.suptitle("All calculation successes" if group == "all_success" else "Top 20 target-matched novel unique molecules",
                                fontsize=12, y=0.98)
                single.tight_layout(rect=(0, 0, 1, 0.94), pad=1.6)
                individual = output / "individual" / f"{name}_{group}_{metric}.png"
                single.savefig(individual, dpi=180)
                plt.close(single)
                files.append(str(individual))
        note = ("Failed calculations excluded; DB matches and duplicate graphs retained."
                if group == "all_success" else
                "Selected by target proximity after DB-match removal and graph deduplication; not an unbiased sample. HP1KG EST axis is zoomed.")
        fig.text(0.5, 0.035, f"Uncalibrated calculated energies; SD uses ddof=0. {note}",
                 ha="center", fontsize=10.2)
        overview = output / f"{group}_histograms.png"
        fig.savefig(overview, dpi=180)
        plt.close(fig)
        files.append(str(overview))
    write_csv(output / "statistics.csv", stats)
    write_csv(output / "histogram_bins.csv", histogram_rows)
    write_csv(output / "plotted_values.csv", value_rows)
    for path, before in source_hashes.items():
        if sha256(Path(path)) != before:
            raise RuntimeError(f"Source changed during plotting: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources_sha256=source_hashes,
                    script_sha256=sha256(Path(__file__)), statistics=stats, files=files,
                    notes=["All successes include duplicates and DB matches; top20 are selected novel unique graphs.",
                           "No failed calculation is assigned zero energy.",
                           "Counts in each histogram sum to the corresponding number of successful calculations.",
                           "S1/T1 bins are common at 0.5 eV; full EST bins are common at 0.25 eV.",
                           "Top20 EST uses a zoomed 0.001 eV grid for HP1KG and 0.1 eV bins for STGG.",
                           "Only HP1KG EST and STGG S1 are generation targets."])
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    shutil.copyfile(__file__, output / "plot_script.py")
    print(json.dumps(dict(output=str(output), png_count=len(files), all_counts={
        name: len(data[name]["all_success"]) for name in RUNS}, top20_counts={
        name: len(data[name]["top20"]) for name in RUNS}, histogram_count=len(stats)), indent=2))


if __name__ == "__main__":
    main()
