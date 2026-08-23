"""Phase 3 verdicts, evaluated against prereg Revision 6.

Both decision rules were frozen before this file existed. Nothing here chooses
a threshold; it only reports which frozen branch the data lands in, including
the branches that end the project.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from dllmfert.degeneracy import (auc, distinctness, partial_spearman,
                                 relative_cutoff, spearman)

# Frozen in prereg Revision 6.
P1_SUPPORT_DISTINCT = 0.8
P1_SUPPORT_PF = -0.8
P1_KILL_PF = -0.3
P2_BUDGETS = (0.10, 0.25, 0.50)
P2_MARGIN = 0.05
P2_DEGENERATE_PCT = 20.0
TOKEN_SHINGLE_K = 8


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def p1_verdict(rows: list[dict]) -> dict:
    """Does suppressing repetition lower the *measured* parallel_factor?

    Correlations are computed over per-penalty means, which is what
    "monotonically" in the prereg refers to; the row-level figures are reported
    alongside so a strong mean-level trend resting on noise is visible.
    """
    out = {}
    by_arm = defaultdict(list)
    for r in rows:
        if r.get("error") or r.get("parallel_factor") is None:
            continue
        by_arm[r["arm"]].append(r)

    for arm, rs in sorted(by_arm.items()):
        by_pen = defaultdict(list)
        for r in rs:
            by_pen[r.get("repetition_penalty", 1.0)].append(r)
        pens = sorted(by_pen)
        if len(pens) < 3:
            out[arm] = {"decision": "INCOMPLETE",
                        "reason": f"{len(pens)} penalty settings, need >= 3"}
            continue

        cells = []
        for p in pens:
            rr = by_pen[p]
            cells.append({
                "penalty": p,
                "n": len(rr),
                "distinct": _mean([distinctness(r.get("text") or "") for r in rr]),
                "pf": _mean([r["parallel_factor"] for r in rr]),
                "content_len": _mean([r.get("tokens_generated") for r in rr]),
                "acc": _mean([1.0 if r.get("acc") else 0.0 for r in rr]),
            })

        rho_distinct = spearman([c["penalty"] for c in cells],
                                [c["distinct"] for c in cells])
        rho_pf = spearman([c["penalty"] for c in cells], [c["pf"] for c in cells])

        row_pen = [r.get("repetition_penalty", 1.0) for r in rs]
        row_pf = [r["parallel_factor"] for r in rs]
        row_len = [float(r.get("tokens_generated") or 0) for r in rs]
        rho_pf_rows = spearman(row_pen, row_pf)
        rho_pf_ctrl = partial_spearman(row_pen, row_pf, row_len)

        # A perfectly flat pf gives no rank variance and so a nan correlation.
        # That is the *clearest* form of the kill condition, not an undecided
        # one: suppressing repetition moved the measured parallel_factor not at
        # all. Letting nan fall through to UNDECIDED would hide the one result
        # that ends the project.
        if math.isnan(rho_pf) or rho_pf > P1_KILL_PF:
            decision = "KILL_K3"
        elif rho_distinct >= P1_SUPPORT_DISTINCT and rho_pf <= P1_SUPPORT_PF:
            # The control decides whether this is the mechanism or the length.
            decision = ("SUPPORTED" if rho_pf_ctrl <= P1_KILL_PF
                        else "CONFOUNDED_WITH_LENGTH")
        else:
            decision = "UNDECIDED"

        out[arm] = {
            "decision": decision,
            "spearman_distinct_vs_penalty": rho_distinct,
            "spearman_pf_vs_penalty": rho_pf,
            "spearman_pf_vs_penalty_rowlevel": rho_pf_rows,
            "spearman_pf_vs_penalty_controlling_length": rho_pf_ctrl,
            "cells": cells,
        }
    return out


def _token_max_repeat(ids: list[int], k: int = TOKEN_SHINGLE_K) -> int:
    """Max repeat over token k-grams.

    Token-level rather than character-level on purpose: the comparison in P2 is
    between two signals *inside one model*, so cross-tokeniser comparability is
    not needed, and this keeps the detector free of a tokeniser at analysis
    time -- which is what an online detector would actually have.
    """
    if len(ids) < k:
        return 0 if not ids else 1
    grams = [tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)]
    return max(Counter(grams).values())


def p2_verdict(rows: list[dict]) -> dict:
    """Does parallel_factor flag a degenerating generation earlier than the
    decoded text does?"""
    out = {}
    by_arm = defaultdict(list)
    for r in rows:
        if (r.get("error") or not r.get("commit_step")
                or not r.get("content_tokens") or r.get("text") is None):
            continue
        by_arm[r["arm"]].append(r)

    for arm, rs in sorted(by_arm.items()):
        if len(rs) < 20:
            out[arm] = {"decision": "INCOMPLETE",
                        "reason": f"{len(rs)} usable rows, need >= 20"}
            continue

        dis = [distinctness(r.get("text") or "") for r in rs]
        cut = relative_cutoff(dis, P2_DEGENERATE_PCT)
        labels = [d <= cut for d in dis]

        budgets = {}
        for b in P2_BUDGETS:
            sig_pf, sig_rep = [], []
            for r in rs:
                steps = r["commit_step"]
                total = max(steps) + 1 if steps else 1
                upto = max(0, math.ceil(b * total) - 1)
                idx = [i for i, s in enumerate(steps) if 0 <= s <= upto]
                n_steps = upto + 1
                sig_pf.append(len(idx) / n_steps)
                toks = [r["content_tokens"][i] for i in idx]
                sig_rep.append(float(_token_max_repeat(toks)))
            a_pf = auc(sig_pf, labels)
            a_rep = auc(sig_rep, labels)
            budgets[f"{int(b * 100)}%"] = {
                "auc_parallel_factor": a_pf,
                "auc_ngram_repeat": a_rep,
                "margin": (a_pf - a_rep
                           if not (math.isnan(a_pf) or math.isnan(a_rep))
                           else None),
            }

        early = [budgets["10%"]["margin"], budgets["25%"]["margin"]]
        if any(m is None for m in early):
            decision = "INCOMPLETE"
        elif all(m >= P2_MARGIN for m in early):
            decision = "M1_SUPPORTED"
        elif all(m <= 0 for m in early):
            decision = "M1_DROPPED"
        else:
            decision = "UNDECIDED"

        out[arm] = {"decision": decision, "n": len(rs),
                    "degenerate_cutoff_distinct": cut,
                    "n_degenerate": sum(labels), "budgets": budgets}
    return out
