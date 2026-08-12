import argparse
import csv
import itertools
import math
import os
import re
import statistics


BASE_RULES = ("align", "importance", "stability")
STRUCTURED_RULES = (
    "cproj_importance",
    "middle_h2_h9_cproj_importance",
    "middle_h3_h9_cproj_importance",
    "soft_middle_l025",
    "soft_middle_l050",
    "soft_middle_l100",
    "contiguous_window_l000",
    "contiguous_window_l025",
    "contiguous_window_l050",
    "contiguous_window_l100",
    "fixed_window_h2_h7",
    "band_guarded_importance",
    "probe_as_veto",
)
RULES = BASE_RULES + STRUCTURED_RULES
DEFAULT_RULES = STRUCTURED_RULES
MIB = 1024 * 1024


def parse_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def parse_int(row, key, default=0):
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def release_percent_label(release_frac):
    value = int(round(release_frac * 100))
    return f"release{value:02d}"


def lambda_from_rule(rule):
    if rule.endswith("_l000"):
        return 0.0
    if rule.endswith("_l025"):
        return 0.25
    if rule.endswith("_l050"):
        return 0.50
    if rule.endswith("_l100"):
        return 1.00
    raise ValueError(f"rule has no lambda suffix: {rule}")


def fixed_window_from_rule(rule):
    match = re.fullmatch(r"fixed_window_h(\d+)_h(\d+)", rule)
    if not match:
        raise ValueError(f"rule is not a fixed window: {rule}")
    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        raise ValueError(f"fixed window start must be <= end: {rule}")
    return start, end


def layer_index(name):
    match = re.search(r"transformer\.h\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def is_mlp_c_proj(row):
    return ".mlp.c_proj.weight" in row.get("name", "")


def in_layer_range(row, start, end):
    idx = row.get("_layer_idx", -1)
    return start <= idx <= end


def model_layer_count(rows):
    return max((row.get("_layer_idx", -1) for row in rows), default=-1) + 1


def band_middle_layers(rows):
    count = model_layer_count(rows)
    if count <= 0:
        return set()
    width = max(1, count // 2)
    start = max(0, int(math.floor((count - width) / 2.0)))
    return set(range(start, start + width))


def read_probe_report(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty cheap Muon probe report: {path}")
    for row in rows:
        row["_k_state_full_bytes"] = parse_int(row, "k_state_full_bytes")
        row["_grad_rms_mean"] = parse_float(row, "grad_rms_mean")
        row["_misalignment"] = max(0.0, parse_float(row, "grad_muon_misalignment_mean"))
        row["_instability"] = max(0.0, parse_float(row, "update_instability_mean"))
        row["_layer_idx"] = layer_index(row.get("name", ""))
    return rows


def zscores(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
    std = math.sqrt(variance)
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def add_base_scores(rows, cost_power):
    positive_grads = [row["_grad_rms_mean"] for row in rows if row["_grad_rms_mean"] > 0.0]
    median_grad = statistics.median(positive_grads) if positive_grads else 1.0
    median_grad = max(median_grad, 1e-12)

    scored = []
    for row in rows:
        cost_mib = max(row["_k_state_full_bytes"] / MIB, 1e-12)
        grad_factor = math.log1p(row["_grad_rms_mean"] / median_grad)
        misalignment = row["_misalignment"]
        instability = row["_instability"]

        enriched = dict(row)
        enriched["_align_score"] = misalignment
        enriched["_importance_score"] = grad_factor * misalignment / (cost_mib**cost_power)
        enriched["_stability_score"] = (
            grad_factor * misalignment * (1.0 + instability) / (cost_mib**cost_power)
        )
        enriched["_grad_factor"] = grad_factor
        enriched["_cost_mib"] = cost_mib
        enriched["_keep_score"] = enriched["_importance_score"]
        enriched["_release_score"] = enriched["_importance_score"]
        enriched["_candidate"] = True
        enriched["_rule_details"] = ""
        scored.append(enriched)
    return scored


def set_rule_scores(scored, rule):
    rows = [dict(row) for row in scored]
    if rule == "align":
        for row in rows:
            row["_keep_score"] = row["_align_score"]
            row["_release_score"] = row["_align_score"]
            row["_rule_details"] = "global align score"
        return rows
    if rule == "importance":
        for row in rows:
            row["_keep_score"] = row["_importance_score"]
            row["_release_score"] = row["_importance_score"]
            row["_rule_details"] = "global importance score"
        return rows
    if rule == "stability":
        for row in rows:
            row["_keep_score"] = row["_stability_score"]
            row["_release_score"] = row["_stability_score"]
            row["_rule_details"] = "global stability score"
        return rows

    for row in rows:
        row["_keep_score"] = row["_importance_score"]
        row["_release_score"] = row["_importance_score"]
        row["_rule_details"] = rule
    return rows


def candidate_rows(rows, rule):
    if rule == "cproj_importance":
        return [row for row in rows if is_mlp_c_proj(row)]
    if rule == "middle_h2_h9_cproj_importance":
        return [row for row in rows if is_mlp_c_proj(row) and in_layer_range(row, 2, 9)]
    if rule == "middle_h3_h9_cproj_importance":
        return [row for row in rows if is_mlp_c_proj(row) and in_layer_range(row, 3, 9)]
    if rule.startswith("soft_middle_"):
        return [row for row in rows if is_mlp_c_proj(row)]
    if rule.startswith("contiguous_window_"):
        return [row for row in rows if is_mlp_c_proj(row)]
    if rule.startswith("fixed_window_"):
        start, end = fixed_window_from_rule(rule)
        return [row for row in rows if is_mlp_c_proj(row) and in_layer_range(row, start, end)]
    if rule == "band_guarded_importance":
        return [row for row in rows if is_mlp_c_proj(row)]
    if rule == "probe_as_veto":
        return [row for row in rows if is_mlp_c_proj(row)]
    return list(rows)


def apply_soft_middle_prior(rows, rule):
    lam = lambda_from_rule(rule)
    candidates = candidate_rows(rows, rule)
    layer_count = max(1, model_layer_count(rows))
    center = (layer_count - 1) / 2.0
    score_z = zscores([row["_importance_score"] for row in candidates])
    distance_z = zscores([abs(row["_layer_idx"] - center) for row in candidates])
    by_name = {}
    for row, score_part, distance_part in zip(candidates, score_z, distance_z):
        enriched = dict(row)
        enriched["_release_score"] = score_part + lam * distance_part
        enriched["_rule_details"] = (
            f"soft_middle lambda={lam}; base_z={score_part:.6g}; "
            f"distance_z={distance_part:.6g}"
        )
        by_name[enriched.get("name", "")] = enriched
    return [by_name.get(row.get("name", ""), dict(row, _candidate=False)) for row in rows]


def target_release_bytes(rows, release_frac):
    total_bytes = sum(row["_k_state_full_bytes"] for row in rows)
    return int(round(total_bytes * release_frac)), total_bytes


def target_candidate_count(candidates, target_bytes):
    if not candidates:
        return 0
    ranked_by_size = sorted(candidates, key=lambda row: row["_k_state_full_bytes"], reverse=True)
    best_count = 0
    best_error = abs(target_bytes)
    running = 0
    for idx, row in enumerate(ranked_by_size, start=1):
        running += row["_k_state_full_bytes"]
        error = abs(running - target_bytes)
        if error < best_error:
            best_error = error
            best_count = idx
    return max(1, best_count)


def choose_by_rank(rows, candidates, release_frac):
    target_bytes, total_bytes = target_release_bytes(rows, release_frac)
    ranked_candidates = sorted(
        candidates,
        key=lambda row: (
            row["_release_score"],
            -row["_k_state_full_bytes"],
            row.get("name", ""),
        ),
    )

    released = []
    released_bytes = 0
    for row in ranked_candidates:
        if target_bytes <= 0:
            break
        next_bytes = released_bytes + row["_k_state_full_bytes"]
        if abs(next_bytes - target_bytes) <= abs(released_bytes - target_bytes):
            released.append(row)
            released_bytes = next_bytes
        else:
            break

    released_names = {row.get("name", "") for row in released}
    ranked = ranked_candidates + sorted(
        [row for row in rows if row.get("name", "") not in {item.get("name", "") for item in ranked_candidates}],
        key=lambda row: row.get("name", ""),
    )
    return ranked, released_names, released_bytes, total_bytes, target_bytes


def choose_contiguous_window(rows, rule, release_frac):
    lam = lambda_from_rule(rule)
    target_bytes, total_bytes = target_release_bytes(rows, release_frac)
    candidates = sorted(candidate_rows(rows, rule), key=lambda row: row["_layer_idx"])
    if not candidates:
        ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
            rows,
            [],
            release_frac,
        )
        return ranked, released_names, released_bytes, total_bytes, target_bytes, rows

    layer_count = max(1, model_layer_count(rows))
    model_center = (layer_count - 1) / 2.0
    windows = []
    for start in range(len(candidates)):
        released_bytes = 0
        window = []
        for end in range(start, len(candidates)):
            window.append(candidates[end])
            released_bytes += candidates[end]["_k_state_full_bytes"]
            layers = [row["_layer_idx"] for row in window]
            if layers != list(range(layers[0], layers[-1] + 1)):
                continue
            windows.append(
                {
                    "rows": list(window),
                    "bytes": released_bytes,
                    "mean_importance": sum(row["_importance_score"] for row in window) / len(window),
                    "center_distance": abs((layers[0] + layers[-1]) / 2.0 - model_center),
                    "start": layers[0],
                    "end": layers[-1],
                }
            )

    viable = [window for window in windows if abs(window["bytes"] - target_bytes) == min(abs(w["bytes"] - target_bytes) for w in windows)]
    mean_z = zscores([window["mean_importance"] for window in viable])
    distance_z = zscores([window["center_distance"] for window in viable])
    best = None
    best_key = None
    for window, mean_part, distance_part in zip(viable, mean_z, distance_z):
        score = mean_part + lam * distance_part
        key = (score, window["center_distance"], window["start"])
        window["score"] = score
        window["rule_details"] = (
            f"contiguous h{window['start']}-h{window['end']}; lambda={lam}; "
            f"mean_z={mean_part:.6g}; distance_z={distance_part:.6g}"
        )
        if best_key is None or key < best_key:
            best = window
            best_key = key

    released_names = {row.get("name", "") for row in best["rows"]}
    released_bytes = best["bytes"]
    by_name = {row.get("name", ""): dict(row, _release_score=best["score"], _rule_details=best["rule_details"]) for row in best["rows"]}
    rewritten = [by_name.get(row.get("name", ""), row) for row in rows]
    ranked = best["rows"] + [row for row in candidates if row.get("name", "") not in released_names]
    ranked += sorted([row for row in rows if not is_mlp_c_proj(row)], key=lambda row: row.get("name", ""))
    return ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten


def choose_fixed_window(rows, rule, release_frac):
    start, end = fixed_window_from_rule(rule)
    target_bytes, total_bytes = target_release_bytes(rows, release_frac)
    cproj = sorted(
        [row for row in rows if is_mlp_c_proj(row)],
        key=lambda row: (row["_layer_idx"], row.get("name", "")),
    )
    released = [row for row in cproj if in_layer_range(row, start, end)]
    if not released:
        ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
            rows,
            [],
            release_frac,
        )
        return ranked, released_names, released_bytes, total_bytes, target_bytes, rows

    released_names = {row.get("name", "") for row in released}
    released_bytes = sum(row["_k_state_full_bytes"] for row in released)
    details = f"fixed c_proj window h{start}-h{end}"
    rewritten = []
    for row in rows:
        name = row.get("name", "")
        is_cproj = is_mlp_c_proj(row)
        if name in released_names:
            rewritten.append(
                dict(
                    row,
                    _release_score=0.0,
                    _rule_details=details,
                    _candidate=True,
                )
            )
        elif is_cproj:
            rewritten.append(dict(row, _rule_details=details, _candidate=True))
        else:
            rewritten.append(
                dict(row, _rule_details=f"not a candidate for {rule}", _candidate=False)
            )

    by_name = {row.get("name", ""): row for row in rewritten}
    released_ranked = [by_name[row.get("name", "")] for row in released]
    remaining_cproj = [
        by_name[row.get("name", "")]
        for row in cproj
        if row.get("name", "") not in released_names
    ]
    ranked = released_ranked + remaining_cproj
    ranked += sorted(
        [row for row in rewritten if not is_mlp_c_proj(row)],
        key=lambda row: row.get("name", ""),
    )
    return ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten


def choose_band_guarded(rows, release_frac, min_band_count=4, max_late_count=1):
    target_bytes, total_bytes = target_release_bytes(rows, release_frac)
    candidates = candidate_rows(rows, "band_guarded_importance")
    count = target_candidate_count(candidates, target_bytes)
    band_layers = band_middle_layers(rows)
    late_start = max(band_layers) + 1 if band_layers else model_layer_count(rows)

    best_combo = None
    best_key = None
    for combo in itertools.combinations(candidates, count):
        released_bytes = sum(row["_k_state_full_bytes"] for row in combo)
        band_count = sum(1 for row in combo if row["_layer_idx"] in band_layers)
        late_count = sum(1 for row in combo if row["_layer_idx"] >= late_start)
        if band_count < min_band_count or late_count > max_late_count:
            continue
        score = sum(row["_importance_score"] for row in combo) / max(1, len(combo))
        center = sum(row["_layer_idx"] for row in combo) / max(1, len(combo))
        key = (abs(released_bytes - target_bytes), score, abs(center - (model_layer_count(rows) - 1) / 2.0))
        if best_key is None or key < best_key:
            best_combo = combo
            best_key = key

    if best_combo is None:
        ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
            rows,
            candidates,
            release_frac,
        )
        return ranked, released_names, released_bytes, total_bytes, target_bytes, rows

    released_names = {row.get("name", "") for row in best_combo}
    released_bytes = sum(row["_k_state_full_bytes"] for row in best_combo)
    details = f"band_guarded min_band={min_band_count}; max_late={max_late_count}"
    rewritten = [
        dict(row, _rule_details=details) if row.get("name", "") in released_names else row
        for row in rows
    ]
    ranked = list(best_combo) + [
        row for row in sorted(candidates, key=lambda row: row["_importance_score"])
        if row.get("name", "") not in released_names
    ]
    ranked += sorted([row for row in rows if not is_mlp_c_proj(row)], key=lambda row: row.get("name", ""))
    return ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten


def choose_probe_as_veto(rows, release_frac, veto_margin):
    target_bytes, total_bytes = target_release_bytes(rows, release_frac)
    cproj = {row["_layer_idx"]: row for row in candidate_rows(rows, "probe_as_veto")}
    band_layers = sorted(band_middle_layers(rows))
    selected_layers = set(band_layers)
    if not selected_layers:
        ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
            rows,
            list(cproj.values()),
            release_frac,
        )
        return ranked, released_names, released_bytes, total_bytes, target_bytes, rows

    neighbor_layers = [min(band_layers) - 1, max(band_layers) + 1]
    for _ in range(len(neighbor_layers)):
        selected = [cproj[layer] for layer in selected_layers if layer in cproj]
        outside = [
            cproj[layer]
            for layer in neighbor_layers
            if layer in cproj and layer not in selected_layers
        ]
        if not selected or not outside:
            break
        worst_selected = max(selected, key=lambda row: row["_importance_score"])
        best_outside = min(outside, key=lambda row: row["_importance_score"])
        if best_outside["_importance_score"] + veto_margin < worst_selected["_importance_score"]:
            selected_layers.remove(worst_selected["_layer_idx"])
            selected_layers.add(best_outside["_layer_idx"])
        else:
            break

    released = [cproj[layer] for layer in sorted(selected_layers) if layer in cproj]
    released_names = {row.get("name", "") for row in released}
    released_bytes = sum(row["_k_state_full_bytes"] for row in released)
    details = f"probe_as_veto margin={veto_margin}; base_band=h{min(band_layers)}-h{max(band_layers)}"
    rewritten = [
        dict(row, _rule_details=details) if row.get("name", "") in released_names else row
        for row in rows
    ]
    ranked = released + [
        row for row in sorted(cproj.values(), key=lambda row: row["_importance_score"])
        if row.get("name", "") not in released_names
    ]
    ranked += sorted([row for row in rows if not is_mlp_c_proj(row)], key=lambda row: row.get("name", ""))
    return ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten


def build_mask(rows, rule, release_frac, veto_margin):
    scored = set_rule_scores(rows, rule)
    if rule.startswith("soft_middle_"):
        scored = apply_soft_middle_prior(scored, rule)
        candidates = candidate_rows(scored, rule)
        ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
            scored,
            candidates,
            release_frac,
        )
        return scored, ranked, released_names, released_bytes, total_bytes, target_bytes
    if rule.startswith("contiguous_window_"):
        ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten = choose_contiguous_window(
            scored,
            rule,
            release_frac,
        )
        return rewritten, ranked, released_names, released_bytes, total_bytes, target_bytes
    if rule.startswith("fixed_window_"):
        ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten = choose_fixed_window(
            scored,
            rule,
            release_frac,
        )
        return rewritten, ranked, released_names, released_bytes, total_bytes, target_bytes
    if rule == "band_guarded_importance":
        ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten = choose_band_guarded(
            scored,
            release_frac,
        )
        return rewritten, ranked, released_names, released_bytes, total_bytes, target_bytes
    if rule == "probe_as_veto":
        ranked, released_names, released_bytes, total_bytes, target_bytes, rewritten = choose_probe_as_veto(
            scored,
            release_frac,
            veto_margin,
        )
        return rewritten, ranked, released_names, released_bytes, total_bytes, target_bytes

    candidates = candidate_rows(scored, rule)
    candidate_names = {row.get("name", "") for row in candidates}
    for row in scored:
        row["_candidate"] = row.get("name", "") in candidate_names
        if not row["_candidate"]:
            row["_rule_details"] = f"not a candidate for {rule}"
    ranked, released_names, released_bytes, total_bytes, target_bytes = choose_by_rank(
        scored,
        candidates,
        release_frac,
    )
    return scored, ranked, released_names, released_bytes, total_bytes, target_bytes


def write_mask(path, rows, ranked, released_names, release_frac, rule, released_bytes, total_bytes, target_bytes):
    ranked_names = [row.get("name", "") for row in ranked]
    rank_by_name = {name: rank for rank, name in enumerate(ranked_names, start=1)}
    fieldnames = [
        "seed",
        "dataset",
        "wandb_project",
        "wandb_group",
        "wandb_run_name",
        "optimizer_type",
        "mask_rule",
        "target_release_k_fraction",
        "target_release_k_state_bytes",
        "actual_release_k_state_bytes",
        "actual_release_k_fraction",
        "rank",
        "name",
        "shape",
        "rows",
        "cols",
        "probe_steps",
        "grad_rms_mean",
        "muon_update_rms_mean",
        "grad_muon_cos_mean",
        "grad_muon_misalignment_mean",
        "update_instability_mean",
        "update_instability_count",
        "cheap_keep_score",
        "cheap_release_score",
        "cheap_grad_factor",
        "cheap_cost_mib",
        "rule_details",
        "candidate",
        "k_state_full_bytes",
        "selected",
        "released",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            name = row.get("name", "")
            released = name in released_names
            writer.writerow(
                {
                    "seed": row.get("seed", ""),
                    "dataset": row.get("dataset", ""),
                    "wandb_project": row.get("wandb_project", ""),
                    "wandb_group": row.get("wandb_group", ""),
                    "wandb_run_name": f"{row.get('wandb_run_name', '')}__mask_{rule}",
                    "optimizer_type": "selective_newton_muon",
                    "mask_rule": rule,
                    "target_release_k_fraction": release_frac,
                    "target_release_k_state_bytes": target_bytes,
                    "actual_release_k_state_bytes": released_bytes,
                    "actual_release_k_fraction": released_bytes / total_bytes if total_bytes > 0 else 0.0,
                    "rank": rank_by_name.get(name, len(rank_by_name) + 1),
                    "name": name,
                    "shape": row.get("shape", ""),
                    "rows": row.get("rows", ""),
                    "cols": row.get("cols", ""),
                    "probe_steps": row.get("probe_steps", ""),
                    "grad_rms_mean": row.get("grad_rms_mean", ""),
                    "muon_update_rms_mean": row.get("muon_update_rms_mean", ""),
                    "grad_muon_cos_mean": row.get("grad_muon_cos_mean", ""),
                    "grad_muon_misalignment_mean": row.get("grad_muon_misalignment_mean", ""),
                    "update_instability_mean": row.get("update_instability_mean", ""),
                    "update_instability_count": row.get("update_instability_count", ""),
                    "cheap_keep_score": row.get("_keep_score", ""),
                    "cheap_release_score": row.get("_release_score", ""),
                    "cheap_grad_factor": row.get("_grad_factor", ""),
                    "cheap_cost_mib": row.get("_cost_mib", ""),
                    "rule_details": row.get("_rule_details", ""),
                    "candidate": 1 if row.get("_candidate", True) else 0,
                    "k_state_full_bytes": row.get("k_state_full_bytes", ""),
                    "selected": 0 if released else 1,
                    "released": 1 if released else 0,
                }
            )


def summarize_mask(rule, mask_path, rows, ranked, released_names, release_frac, released_bytes, total_bytes, target_bytes):
    released_ranked = [row for row in ranked if row.get("name", "") in released_names]
    band_layers = band_middle_layers(rows)
    released_cproj = [row for row in released_ranked if is_mlp_c_proj(row)]
    band_overlap = [row for row in released_cproj if row.get("_layer_idx", -1) in band_layers]
    cproj_layers = [row.get("_layer_idx", -1) for row in released_cproj]
    if cproj_layers and cproj_layers == list(range(min(cproj_layers), max(cproj_layers) + 1)):
        cproj_window = f"h{min(cproj_layers)}-h{max(cproj_layers)}"
    else:
        cproj_window = ""
    return {
        "rule": rule,
        "mask_path": mask_path,
        "target_release_k_fraction": release_frac,
        "target_release_k_state_mib": target_bytes / MIB,
        "actual_release_k_state_mib": released_bytes / MIB,
        "actual_release_k_fraction": released_bytes / total_bytes if total_bytes > 0 else 0.0,
        "released_layers": len(released_names),
        "released_cproj_layers": len(released_cproj),
        "band_middle_cproj_overlap": len(band_overlap),
        "cproj_window": cproj_window,
        "released_names": ";".join(row.get("name", "") for row in released_ranked),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--release-frac", type=float, default=0.40)
    parser.add_argument("--cost-power", type=float, default=0.25)
    parser.add_argument("--veto-margin", type=float, default=0.02)
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(DEFAULT_RULES))
    args = parser.parse_args()

    if not 0.0 <= args.release_frac < 1.0:
        raise ValueError("--release-frac must be in [0, 1)")
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.probe_report))
    os.makedirs(output_dir, exist_ok=True)

    rows = read_probe_report(args.probe_report)
    scored_base = add_base_scores(rows, args.cost_power)
    release_label = release_percent_label(args.release_frac)
    summary_rows = []
    for rule in args.rules:
        mask_rows, ranked, released_names, released_bytes, total_bytes, target_bytes = build_mask(
            scored_base,
            rule,
            args.release_frac,
            args.veto_margin,
        )
        mask_path = os.path.join(output_dir, f"cheap_muon_probe_mask_{rule}_{release_label}.csv")
        write_mask(
            mask_path,
            mask_rows,
            ranked,
            released_names,
            args.release_frac,
            rule,
            released_bytes,
            total_bytes,
            target_bytes,
        )
        summary = summarize_mask(
            rule,
            mask_path,
            mask_rows,
            ranked,
            released_names,
            args.release_frac,
            released_bytes,
            total_bytes,
            target_bytes,
        )
        summary_rows.append(summary)
        print(
            f"{rule}: released {released_bytes / MIB:.2f} MiB "
            f"({released_bytes / total_bytes:.2%}); "
            f"c_proj={summary['released_cproj_layers']}; "
            f"band_overlap={summary['band_middle_cproj_overlap']}; "
            + summary["released_names"]
        )

    summary_path = os.path.join(output_dir, f"cheap_muon_probe_mask_summary_{release_label}.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote mask summary to {summary_path}")


if __name__ == "__main__":
    main()
