"""
Stage 4: Rule Extraction & Causal Validation
==============================================
Takes the best SAE run from Stage 3 and:
  1. Reads IF-THEN rules from W_a and W_int weights
  2. Evaluates rule fidelity, coverage, and complexity
  3. Performs concept-level causal interventions (KL divergence)
  4. Performs rule-level causal validation (counterfactual action flips)
  5. Produces a human-readable rule report

Usage:
    python rule_extraction.py \\
        --stage2_dir ./stage2_outputs \\
        --stage3_path ./stage3_outputs/stage3_outputs.pt \\
        --features_path ./stage1_outputs/collected_data.pt \\
        --stage1_path ./stage1_outputs/stage1_outputs.pt \\
        --save_dir ./stage4_outputs
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# 4.1  Rule readout from weights
# ============================================================================

ACTION_NAMES = ["TurnLeft", "TurnRight", "Forward", "Pickup", "Drop", "Toggle", "Done"]


@dataclass
class Rule:
    """A single IF-THEN rule."""
    action: int
    action_name: str
    positive_concepts: List[int]       # concept indices that must be active
    negative_concepts: List[int]       # concept indices that must be inactive
    positive_labels: List[str]         # human-readable names
    negative_labels: List[str]
    weight: float                      # total weight magnitude (importance)
    rule_type: str                     # "single", "interaction", "combined"
    interaction_pair: Optional[Tuple[int, int]] = None  # for interaction rules

    def __str__(self):
        parts = []
        for label in self.positive_labels:
            parts.append(label)
        for label in self.negative_labels:
            parts.append(f"NOT {label}")
        condition = " AND ".join(parts) if parts else "ALWAYS"
        return f"IF {condition} THEN {self.action_name} (w={self.weight:.3f})"

    def to_dict(self):
        return {
            "action": self.action,
            "action_name": self.action_name,
            "positive_concepts": self.positive_concepts,
            "negative_concepts": self.negative_concepts,
            "positive_labels": self.positive_labels,
            "negative_labels": self.negative_labels,
            "weight": self.weight,
            "rule_type": self.rule_type,
            "interaction_pair": list(self.interaction_pair) if self.interaction_pair else None,
        }


def extract_rules(
    action_predictor,
    concept_mapping: torch.Tensor,
    concept_labels: List[str],
    weight_threshold: float = 0.1,
    n_actions: int = 7,
) -> List[Rule]:
    """
    Read rules directly from the action predictor weights.

    Single-concept rules: large |W_a[action, concept_col]|
    Interaction rules: large |W_int[action, pair_idx]|

    Args:
        action_predictor : trained InteractionActionPredictor
        concept_mapping  : (K,) maps consensus concept i -> column in SAE hidden dim
        concept_labels   : (K,) human-readable names
        weight_threshold : minimum |weight| to include a rule
        n_actions        : number of discrete actions

    Returns:
        list of Rule objects, sorted by weight magnitude
    """
    W_a = action_predictor.W_a.weight.data.cpu()  # (n_actions, hidden_dim)
    b_a = action_predictor.W_a.bias.data.cpu()     # (n_actions,)
    K = len(concept_mapping)

    rules = []

    # --- Single-concept rules ---
    for a in range(n_actions):
        for ci in range(K):
            col = concept_mapping[ci].item()
            w = W_a[a, col].item()
            if abs(w) > weight_threshold:
                if w > 0:
                    rules.append(Rule(
                        action=a, action_name=ACTION_NAMES[a],
                        positive_concepts=[ci], negative_concepts=[],
                        positive_labels=[concept_labels[ci]], negative_labels=[],
                        weight=w, rule_type="single",
                    ))
                else:
                    rules.append(Rule(
                        action=a, action_name=ACTION_NAMES[a],
                        positive_concepts=[], negative_concepts=[ci],
                        positive_labels=[], negative_labels=[concept_labels[ci]],
                        weight=abs(w), rule_type="single",
                    ))

    # --- Interaction rules ---
    if action_predictor.W_int is not None:
        W_int = action_predictor.W_int.weight.data.cpu()  # (n_actions, n_pairs)
        pairs = action_predictor.interaction_pairs.cpu()   # (n_pairs, 2)

        # Map SAE hidden dim indices to consensus concept indices
        col_to_concept = {}
        for ci in range(K):
            col_to_concept[concept_mapping[ci].item()] = ci

        for a in range(n_actions):
            for pi in range(len(pairs)):
                w = W_int[a, pi].item()
                if abs(w) > weight_threshold:
                    col_i, col_j = pairs[pi, 0].item(), pairs[pi, 1].item()
                    # Check if both columns map to consensus concepts
                    ci = col_to_concept.get(col_i)
                    cj = col_to_concept.get(col_j)

                    if ci is not None and cj is not None:
                        labels_pos = [concept_labels[ci], concept_labels[cj]]
                        concepts_pos = [ci, cj]
                    elif ci is not None:
                        labels_pos = [concept_labels[ci], f"hidden_{col_j}"]
                        concepts_pos = [ci]
                    elif cj is not None:
                        labels_pos = [f"hidden_{col_i}", concept_labels[cj]]
                        concepts_pos = [cj]
                    else:
                        labels_pos = [f"hidden_{col_i}", f"hidden_{col_j}"]
                        concepts_pos = []

                    if w > 0:
                        rules.append(Rule(
                            action=a, action_name=ACTION_NAMES[a],
                            positive_concepts=concepts_pos, negative_concepts=[],
                            positive_labels=labels_pos, negative_labels=[],
                            weight=w, rule_type="interaction",
                            interaction_pair=(col_i, col_j),
                        ))
                    else:
                        # Negative interaction: penalize when both are active
                        rules.append(Rule(
                            action=a, action_name=ACTION_NAMES[a],
                            positive_concepts=concepts_pos, negative_concepts=[],
                            positive_labels=labels_pos, negative_labels=[],
                            weight=abs(w), rule_type="interaction_negative",
                            interaction_pair=(col_i, col_j),
                        ))

    # Sort by weight
    rules.sort(key=lambda r: r.weight, reverse=True)

    return rules


# ============================================================================
# 4.2  Rule evaluation (fidelity, coverage, complexity)
# ============================================================================

def evaluate_rules(
    rules: List[Rule],
    model,
    action_predictor,
    features: torch.Tensor,
    actions: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    concept_mapping: torch.Tensor,
    n_actions: int = 7,
) -> dict:
    """
    Evaluate rule set quality.

    Fidelity: agreement between rules and actual policy
    Coverage: fraction of states where at least one rule fires
    Complexity: number of rules, mean rule length
    """
    model.eval()
    features_norm = (features - feat_mean) / feat_std

    with torch.no_grad():
        _, z_sparse, _ = model(features_norm)
        action_logits = action_predictor(z_sparse)
        pred_actions = action_logits.argmax(dim=-1)

    N = len(features)

    # --- Action predictor fidelity (full model, not just rules) ---
    full_fidelity = (pred_actions == actions).float().mean().item()

    # --- Per-action fidelity ---
    per_action = {}
    for a in range(n_actions):
        mask = actions == a
        if mask.sum() > 0:
            acc = (pred_actions[mask] == a).float().mean().item()
            per_action[ACTION_NAMES[a]] = {
                "accuracy": acc,
                "count": int(mask.sum().item()),
            }

    # --- Rule-based evaluation ---
    # Binarize concept activations for rule matching
    # Use consensus concept columns only
    K = len(concept_mapping)
    concept_active = torch.zeros(N, K, dtype=torch.bool)
    for ci in range(K):
        col = concept_mapping[ci].item()
        concept_active[:, ci] = z_sparse[:, col] > 0

    # Match rules to states
    rule_predictions = torch.full((N,), -1, dtype=torch.long)  # -1 = no rule fires
    rule_fired = torch.zeros(N, dtype=torch.bool)

    # Apply rules in priority order (sorted by weight)
    for rule in rules:
        # Check conditions
        match = torch.ones(N, dtype=torch.bool)
        for ci in rule.positive_concepts:
            if ci < K:
                match &= concept_active[:, ci]
        for ci in rule.negative_concepts:
            if ci < K:
                match &= ~concept_active[:, ci]

        # Apply to states not yet covered
        new_matches = match & ~rule_fired
        rule_predictions[new_matches] = rule.action
        rule_fired |= match

    coverage = rule_fired.float().mean().item()

    # Fidelity among covered states
    covered_mask = rule_predictions >= 0
    if covered_mask.sum() > 0:
        rule_fidelity = (rule_predictions[covered_mask] == actions[covered_mask]).float().mean().item()
    else:
        rule_fidelity = 0.0

    # Complexity
    n_rules = len(rules)
    mean_length = np.mean([
        len(r.positive_concepts) + len(r.negative_concepts) for r in rules
    ]) if rules else 0.0

    # Interaction order histogram
    order_hist = {}
    for r in rules:
        order = len(r.positive_concepts) + len(r.negative_concepts)
        order_hist[order] = order_hist.get(order, 0) + 1

    result = {
        "full_model_fidelity": full_fidelity,
        "rule_fidelity": rule_fidelity,
        "coverage": coverage,
        "n_rules": n_rules,
        "mean_rule_length": mean_length,
        "interaction_order_histogram": order_hist,
        "per_action": per_action,
    }

    print(f"\n{'='*60}")
    print(f"RULE EVALUATION")
    print(f"{'='*60}")
    print(f"  Full model fidelity : {full_fidelity*100:.2f}%")
    print(f"  Rule-based fidelity : {rule_fidelity*100:.2f}%")
    print(f"  Coverage            : {coverage*100:.1f}%")
    print(f"  Number of rules     : {n_rules}")
    print(f"  Mean rule length    : {mean_length:.1f}")
    print(f"  Interaction orders  : {order_hist}")
    print(f"\n  Per-action accuracy:")
    for name, data in per_action.items():
        print(f"    {name:10s}: {data['accuracy']*100:5.1f}% ({data['count']} samples)")

    return result


# ============================================================================
# 4.3  Concept-level causal interventions
# ============================================================================

def causal_interventions(
    model,
    action_predictor,
    features: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    concept_mapping: torch.Tensor,
    concept_labels: List[str],
    n_sample: int = 2000,
) -> dict:
    """
    For each consensus concept, measure causal effect via intervention.

    Protocol:
      1. Find states where concept c_i is inactive
      2. Activate c_i (set z[col] = median activation of active states)
      3. Measure KL divergence between original and intervened action distributions

    Returns:
        dict with per-concept KL divergence and most-affected action
    """
    model.eval()
    K = len(concept_mapping)
    features_norm = (features[:n_sample] - feat_mean) / feat_std

    with torch.no_grad():
        _, z_sparse, _ = model(features_norm)
        original_logits = action_predictor(z_sparse)
        original_probs = F.softmax(original_logits, dim=-1)

    results = []

    for ci in range(K):
        col = concept_mapping[ci].item()

        with torch.no_grad():
            # Find states where this concept is inactive
            inactive_mask = z_sparse[:, col] <= 0
            if inactive_mask.sum() < 10:
                results.append({
                    "concept": ci, "label": concept_labels[ci],
                    "kl_div": 0.0, "most_affected_action": "N/A",
                    "n_inactive": int(inactive_mask.sum().item()),
                })
                continue

            # Compute median activation among active states
            active_vals = z_sparse[~inactive_mask, col]
            if len(active_vals) > 0:
                activation_val = active_vals.median().item()
            else:
                activation_val = 1.0

            # Intervene: activate the concept
            z_intervened = z_sparse.clone()
            z_intervened[inactive_mask, col] = activation_val

            # Get new action distribution
            intervened_logits = action_predictor(z_intervened)
            intervened_probs = F.softmax(intervened_logits, dim=-1)

            # KL divergence (only for inactive states)
            kl = F.kl_div(
                intervened_probs[inactive_mask].log().clamp(min=-100),
                original_probs[inactive_mask],
                reduction="batchmean",
                log_target=False,
            ).item()

            # Most affected action
            prob_diff = (intervened_probs[inactive_mask] - original_probs[inactive_mask]).mean(dim=0)
            most_increased = prob_diff.argmax().item()
            most_decreased = prob_diff.argmin().item()

            results.append({
                "concept": ci,
                "label": concept_labels[ci],
                "kl_div": abs(kl),
                "most_increased_action": ACTION_NAMES[most_increased],
                "most_decreased_action": ACTION_NAMES[most_decreased],
                "prob_shift": prob_diff.tolist(),
                "n_inactive": int(inactive_mask.sum().item()),
            })

    # Sort by KL divergence
    results.sort(key=lambda r: r["kl_div"], reverse=True)

    print(f"\n{'='*60}")
    print(f"CAUSAL INTERVENTIONS ({K} concepts)")
    print(f"{'='*60}")
    for r in results:
        print(f"  C{r['concept']:02d} ({r['label']:20s}): "
              f"KL={r['kl_div']:.4f}  "
              f"+{r.get('most_increased_action', 'N/A'):10s}  "
              f"-{r.get('most_decreased_action', 'N/A'):10s}  "
              f"(n_inactive={r['n_inactive']})")

    return {"concept_interventions": results}


# ============================================================================
# 4.4  Rule-level causal validation
# ============================================================================

def validate_rules_causally(
    rules: List[Rule],
    model,
    action_predictor,
    features: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    concept_mapping: torch.Tensor,
    n_sample: int = 2000,
) -> dict:
    """
    For each rule, find states where the antecedent is *almost* satisfied
    (one concept missing), activate the missing concept, and check if the
    action flips to the rule's predicted action.

    Returns:
        dict with per-rule causal validation rate
    """
    model.eval()
    K = len(concept_mapping)
    features_norm = (features[:n_sample] - feat_mean) / feat_std
    N = len(features_norm)

    with torch.no_grad():
        _, z_sparse, _ = model(features_norm)

    # Binarize
    concept_active = torch.zeros(N, K, dtype=torch.bool)
    for ci in range(K):
        col = concept_mapping[ci].item()
        concept_active[:, ci] = z_sparse[:, col] > 0

    rule_validations = []

    for ri, rule in enumerate(rules):
        if not rule.positive_concepts or rule.rule_type.startswith("interaction"):
            # Skip interaction rules and rules with no positive concepts
            rule_validations.append({
                "rule_idx": ri, "rule_str": str(rule),
                "validation_rate": None, "n_candidates": 0,
                "reason": "skipped (no single-concept counterfactual)"
            })
            continue

        pos = [c for c in rule.positive_concepts if c < K]
        neg = [c for c in rule.negative_concepts if c < K]

        if len(pos) == 0:
            rule_validations.append({
                "rule_idx": ri, "rule_str": str(rule),
                "validation_rate": None, "n_candidates": 0,
                "reason": "no consensus positive concepts"
            })
            continue

        # For each positive concept, try removing it (find states where
        # all OTHER conditions are met but this one is missing)
        total_candidates = 0
        total_flipped = 0

        for missing_ci in pos:
            # States where all conditions met EXCEPT missing_ci
            others_met = torch.ones(N, dtype=torch.bool)
            for ci in pos:
                if ci != missing_ci:
                    others_met &= concept_active[:, ci]
            for ci in neg:
                others_met &= ~concept_active[:, ci]
            # AND missing concept is inactive
            almost = others_met & ~concept_active[:, missing_ci]

            if almost.sum() < 1:
                continue

            # Original action
            with torch.no_grad():
                original_logits = action_predictor(z_sparse[almost])
                original_actions = original_logits.argmax(dim=-1)

                # Intervene: activate missing concept
                z_int = z_sparse[almost].clone()
                col = concept_mapping[missing_ci].item()
                active_vals = z_sparse[z_sparse[:, col] > 0, col]
                act_val = active_vals.median().item() if len(active_vals) > 0 else 1.0
                z_int[:, col] = act_val

                int_logits = action_predictor(z_int)
                int_actions = int_logits.argmax(dim=-1)

                # Check if action flipped to rule's predicted action
                flipped = (int_actions == rule.action) & (original_actions != rule.action)
                total_candidates += int(almost.sum().item())
                total_flipped += int(flipped.sum().item())

        validation_rate = total_flipped / max(total_candidates, 1)
        rule_validations.append({
            "rule_idx": ri,
            "rule_str": str(rule),
            "validation_rate": validation_rate,
            "n_candidates": total_candidates,
            "n_flipped": total_flipped,
        })

    # Summary
    valid_rules = [rv for rv in rule_validations if rv["validation_rate"] is not None]
    if valid_rules:
        rates = [rv["validation_rate"] for rv in valid_rules]
        mean_rate = np.mean(rates)
        n_validated = sum(1 for r in rates if r > 0.1)
    else:
        mean_rate = 0.0
        n_validated = 0

    print(f"\n{'='*60}")
    print(f"RULE-LEVEL CAUSAL VALIDATION")
    print(f"{'='*60}")
    print(f"  Rules tested: {len(valid_rules)}/{len(rules)}")
    print(f"  Mean validation rate: {mean_rate*100:.1f}%")
    print(f"  Rules validated (>10%): {n_validated}")
    for rv in rule_validations[:20]:  # Show top 20
        if rv["validation_rate"] is not None:
            print(f"    [{rv['rule_idx']:2d}] {rv['rule_str']}")
            print(f"         rate={rv['validation_rate']*100:.1f}% "
                  f"(n={rv['n_candidates']}, flipped={rv.get('n_flipped', 0)})")
        else:
            print(f"    [{rv['rule_idx']:2d}] {rv['rule_str']}  — {rv.get('reason', 'skipped')}")

    return {
        "rule_validations": rule_validations,
        "mean_validation_rate": mean_rate,
        "n_validated": n_validated,
    }


# ============================================================================
# 4.5  Visualization
# ============================================================================

def plot_stage4_diagnostics(
    rules: List[Rule],
    eval_result: dict,
    causal_result: dict,
    validation_result: dict,
    save_dir: str,
):
    """Generate Stage 4 diagnostic plots."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Stage 4: Rule Extraction & Causal Validation",
                 fontsize=14, fontweight="bold")

    # 1. Rule weights by action
    ax = axes[0]
    action_weights = {}
    for r in rules:
        a = r.action_name
        if a not in action_weights:
            action_weights[a] = []
        action_weights[a].append(r.weight)

    if action_weights:
        names = list(action_weights.keys())
        positions = range(len(names))
        for i, name in enumerate(names):
            ws = action_weights[name]
            ax.scatter([i] * len(ws), ws, alpha=0.6, s=30)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rule weight")
    ax.set_title("Rule Weights by Action")
    ax.grid(True, alpha=0.3)

    # 2. Causal KL divergence per concept
    ax = axes[1]
    interventions = causal_result.get("concept_interventions", [])
    if interventions:
        labels = [r["label"][:15] for r in interventions]
        kls = [r["kl_div"] for r in interventions]
        colors = ["green" if kl > 0.1 else "orange" if kl > 0.01 else "red" for kl in kls]
        ax.barh(range(len(labels)), kls, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("KL Divergence")
    ax.set_title("Causal Effect per Concept")
    ax.grid(True, alpha=0.3)

    # 3. Rule validation rates
    ax = axes[2]
    valid_rules = [rv for rv in validation_result.get("rule_validations", [])
                   if rv["validation_rate"] is not None and rv["n_candidates"] > 0]
    if valid_rules:
        valid_rules_sorted = sorted(valid_rules, key=lambda x: x["validation_rate"], reverse=True)
        labels = [f"R{rv['rule_idx']}" for rv in valid_rules_sorted[:20]]
        rates = [rv["validation_rate"] * 100 for rv in valid_rules_sorted[:20]]
        colors = ["green" if r > 10 else "orange" if r > 1 else "red" for r in rates]
        ax.barh(range(len(labels)), rates, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Validation Rate (%)")
    ax.set_title("Rule Causal Validation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "stage4_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Diagnostic plot saved: {path}")


# ============================================================================
# 4.6  Human-readable rule report
# ============================================================================

def generate_rule_report(
    rules: List[Rule],
    eval_result: dict,
    causal_result: dict,
    validation_result: dict,
    stage3_data: dict,
    save_path: str,
):
    """Generate a human-readable text report of extracted rules."""
    lines = []
    lines.append("=" * 70)
    lines.append("  SCORE: Rule Extraction Report")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    K_consensus = len(stage3_data.get("concept_labels", []))
    lines.append(f"  Consensus concepts  : {K_consensus}")
    lines.append(f"  Total rules         : {eval_result['n_rules']}")
    lines.append(f"  Model fidelity      : {eval_result['full_model_fidelity']*100:.2f}%")
    lines.append(f"  Rule fidelity       : {eval_result['rule_fidelity']*100:.2f}%")
    lines.append(f"  Coverage            : {eval_result['coverage']*100:.1f}%")
    lines.append(f"  Mean rule length    : {eval_result['mean_rule_length']:.1f}")
    cka = stage3_data.get("mean_cka")
    if cka is not None:
        lines.append(f"  Mean CKA stability  : {cka:.4f}")
    lines.append("")

    # Concept dictionary
    lines.append("CONCEPT DICTIONARY")
    lines.append("-" * 40)
    concept_labels = stage3_data.get("concept_labels", [])
    mono_scores = stage3_data.get("mono_scores", torch.zeros(0))
    best_nmi = stage3_data.get("best_nmi", torch.zeros(0))
    for i in range(K_consensus):
        mono = mono_scores[i].item() if i < len(mono_scores) else 0
        nmi = best_nmi[i].item() if i < len(best_nmi) else 0
        mono_tag = "monosemantic" if mono > 2.0 else "polysemantic" if mono > 0 else "ungrounded"
        lines.append(f"  C{i:02d}: {concept_labels[i]:25s}  NMI={nmi:.3f}  Mono={mono:.2f}  [{mono_tag}]")
    lines.append("")

    # Rules by action
    lines.append("RULES BY ACTION")
    lines.append("-" * 40)
    for a in range(7):
        action_rules = [r for r in rules if r.action == a]
        if not action_rules:
            continue
        lines.append(f"\n  {ACTION_NAMES[a]} ({len(action_rules)} rules):")
        for r in action_rules[:10]:  # Max 10 per action
            lines.append(f"    {r}")
    lines.append("")

    # Causal effects
    lines.append("CAUSAL EFFECTS")
    lines.append("-" * 40)
    interventions = causal_result.get("concept_interventions", [])
    for r in interventions:
        lines.append(
            f"  C{r['concept']:02d} ({r['label']:20s}): "
            f"KL={r['kl_div']:.4f}  "
            f"+{r.get('most_increased_action', 'N/A')}  "
            f"-{r.get('most_decreased_action', 'N/A')}"
        )
    lines.append("")

    # Causal validation
    lines.append("CAUSAL VALIDATION")
    lines.append("-" * 40)
    lines.append(f"  Mean validation rate: {validation_result['mean_validation_rate']*100:.1f}%")
    lines.append(f"  Rules validated (>10%): {validation_result['n_validated']}")
    lines.append("")

    # Per-action accuracy
    lines.append("PER-ACTION ACCURACY")
    lines.append("-" * 40)
    for name, data in eval_result.get("per_action", {}).items():
        lines.append(f"  {name:10s}: {data['accuracy']*100:5.1f}% ({data['count']} samples)")

    report = "\n".join(lines)

    with open(save_path, "w") as f:
        f.write(report)
    print(f"\n  Rule report saved: {save_path}")

    # Also print
    print(f"\n{report}")

    return report


# ============================================================================
# 4.7  Save outputs
# ============================================================================

def save_stage4_outputs(
    rules: List[Rule],
    eval_result: dict,
    causal_result: dict,
    validation_result: dict,
    save_dir: str,
):
    """Save all Stage 4 outputs."""
    os.makedirs(save_dir, exist_ok=True)

    # Rules as JSON
    rules_json = [r.to_dict() for r in rules]
    with open(os.path.join(save_dir, "rules.json"), "w") as f:
        json.dump(rules_json, f, indent=2)

    # Evaluation
    eval_json = {k: v for k, v in eval_result.items()}
    with open(os.path.join(save_dir, "evaluation.json"), "w") as f:
        json.dump(eval_json, f, indent=2)

    # Causal
    causal_json = {}
    for k, v in causal_result.items():
        causal_json[k] = v
    with open(os.path.join(save_dir, "causal_interventions.json"), "w") as f:
        json.dump(causal_json, f, indent=2)

    # Validation
    val_json = {
        "mean_validation_rate": validation_result["mean_validation_rate"],
        "n_validated": validation_result["n_validated"],
        "rule_validations": [
            {k: v for k, v in rv.items() if k != "rule_str"}
            for rv in validation_result["rule_validations"]
        ],
    }
    with open(os.path.join(save_dir, "causal_validation.json"), "w") as f:
        json.dump(val_json, f, indent=2)

    print(f"  Stage 4 outputs saved to {save_dir}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_stage4(
    stage2_dir: str,
    stage3_data: dict,
    features: torch.Tensor,
    actions: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    save_dir: str = "./stage4_outputs",
    weight_threshold: float = 0.1,
    n_sample_causal: int = 2000,
) -> dict:
    """Run the full Stage 4 pipeline."""
    from sparse_concept_autoencoder import load_run

    print(f"\n{'#'*60}")
    print(f"  STAGE 4: RULE EXTRACTION & CAUSAL VALIDATION")
    print(f"{'#'*60}")

    # Load best run
    best_run_dir = stage3_data["best_run_dir"]
    print(f"\n  Loading best run from {best_run_dir}...")
    run_data = load_run(best_run_dir)
    model = run_data["model"]
    action_predictor = run_data["action_predictor"]

    concept_mapping = stage3_data["concept_mapping"]
    concept_labels = stage3_data["concept_labels"]
    K = len(concept_labels)

    print(f"  Consensus concepts: {K}")
    print(f"  Concept labels: {concept_labels}")

    # 4.1 Extract rules
    print(f"\n  Extracting rules (threshold={weight_threshold})...")
    rules = extract_rules(
        action_predictor, concept_mapping, concept_labels,
        weight_threshold=weight_threshold,
    )
    print(f"  Found {len(rules)} rules")

    # Print top rules
    print(f"\n  Top 20 rules:")
    for i, r in enumerate(rules[:20]):
        print(f"    {i+1:2d}. {r}")

    # 4.2 Evaluate
    eval_result = evaluate_rules(
        rules, model, action_predictor,
        features, actions, feat_mean, feat_std,
        concept_mapping,
    )

    # 4.3 Causal interventions
    causal_result = causal_interventions(
        model, action_predictor,
        features, feat_mean, feat_std,
        concept_mapping, concept_labels,
        n_sample=n_sample_causal,
    )

    # 4.4 Rule-level causal validation
    validation_result = validate_rules_causally(
        rules, model, action_predictor,
        features, feat_mean, feat_std,
        concept_mapping,
        n_sample=n_sample_causal,
    )

    # 4.5 Plots
    plot_stage4_diagnostics(
        rules, eval_result, causal_result, validation_result, save_dir,
    )

    # 4.6 Report
    generate_rule_report(
        rules, eval_result, causal_result, validation_result,
        stage3_data,
        os.path.join(save_dir, "rule_report.txt"),
    )

    # 4.7 Save
    save_stage4_outputs(rules, eval_result, causal_result, validation_result, save_dir)

    print(f"\n{'='*60}")
    print(f"STAGE 4 COMPLETE")
    print(f"{'='*60}")

    return {
        "rules": rules,
        "eval": eval_result,
        "causal": causal_result,
        "validation": validation_result,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Rule Extraction & Causal Validation")

    parser.add_argument("--stage2_dir", type=str, required=True)
    parser.add_argument("--stage3_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--stage1_path", type=str, required=True)

    parser.add_argument("--weight_threshold", type=float, default=0.1,
                        help="Min |weight| for rule inclusion")
    parser.add_argument("--n_sample_causal", type=int, default=2000,
                        help="Samples for causal interventions")
    parser.add_argument("--save_dir", type=str, default="./stage4_outputs")

    args = parser.parse_args()

    # Load data
    raw = torch.load(args.features_path, map_location="cpu", weights_only=False)
    features = raw["features"]
    actions = raw["actions"]

    stage1_data = torch.load(args.stage1_path, map_location="cpu", weights_only=False)
    feat_mean = stage1_data["feature_mean"]
    feat_std = stage1_data["feature_std"]

    stage3_data = torch.load(args.stage3_path, map_location="cpu", weights_only=False)

    run_stage4(
        stage2_dir=args.stage2_dir,
        stage3_data=stage3_data,
        features=features,
        actions=actions,
        feat_mean=feat_mean,
        feat_std=feat_std,
        save_dir=args.save_dir,
        weight_threshold=args.weight_threshold,
        n_sample_causal=args.n_sample_causal,
    )


if __name__ == "__main__":
    main()