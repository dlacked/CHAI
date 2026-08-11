import numpy as np
from scipy.optimize import linear_sum_assignment

# Shared context for the three post-processes below: server.py /tooth_predict's per-tooth argmax
# has no way to know two teeth in the same quadrant claimed the same last digit (anatomically
# impossible - a quadrant has at most one of each). All three take the same (probs, group_keys)
# shape and only differ in how they resolve that conflict once found - see each docstring for
# the trade-off. A held-out test comparison (see resolve_quadrant_duplicates) is what replaced
# the ResNet+ViT arch-transformer refinement step with resolve_quadrant_duplicates in production.


def correct_mirrors_by_digit_occurrence(raw_pred, mirrors):
    """Refines the client's geometry-only quadrant guess (`mirrors`, from js/geometry.js
    computeArchMirrorFlags) using ResNet's own raw per-tooth digit predictions, before any
    quadrant-grouped post-process (resolve_quadrant_duplicates etc.) runs on it.

    A full arch's last digit runs 1..6 with each digit appearing at most once per quadrant (so
    at most twice total, once per side). Whichever arch-order position hits a given digit FIRST
    is unambiguously the non-mirrored side and the SECOND occurrence is the mirrored side - no
    geometry needed at all for that tooth, and completely unaffected by a missing tooth
    elsewhere in the arch. This is the fix for a failure mode the PCA-only geometric guess can't
    avoid: a missing tooth skews the PCA-estimated midline, and the tooth most exposed to that
    skew is exactly the one closest to the true midline (a digit-"1" tooth) - which is precisely
    the tooth this occurrence check disambiguates for free whenever its pair is still present.

    Only a digit that appears exactly ONCE (its own partner tooth is genuinely missing, so there
    is no second occurrence to disambiguate against) falls back to the geometric guess - that
    case genuinely has no digit-based signal to use instead. Three or more occurrences of the
    same digit in one arch shouldn't happen (a real quadrant has at most one of each) and is
    left on the geometric guess too, rather than picking which two of three "count" as the pair.

    raw_pred: (n,) array/list of 0-indexed digit predictions (ResNet's own argmax, before any
    post-process), one per tooth, in left-to-right arch order. mirrors: length-n list of bools,
    the client's geometry-only guess. Returns a corrected length-n list of bools.
    """
    n = len(raw_pred)
    corrected = list(mirrors)
    positions_by_digit = {}
    for i in range(n):
        positions_by_digit.setdefault(int(raw_pred[i]), []).append(i)

    for positions in positions_by_digit.values():
        if len(positions) == 2:
            first, second = positions
            corrected[first] = False
            corrected[second] = True

    return corrected


def _group_ranges(group_keys, n):
    """Contiguous runs of equal keys -> [(start, end), ...] half-open ranges. Shared by all three
    resolvers below; the only thing they disagree on is what to do inside each range."""
    groups = []
    start = 0
    for i in range(1, n):
        if group_keys[i] != group_keys[start]:
            groups.append((start, i))
            start = i
    groups.append((start, n))
    return groups


def resolve_quadrant_duplicates(probs, group_keys):
    """Per-quadrant Hungarian assignment: within each contiguous run of teeth sharing the same
    `group_keys` value (one quadrant), forces a unique last-digit per tooth by maximizing total
    log-probability, instead of each tooth's independent argmax.

    Compared against a stricter DP (resolve_quadrant_monotonic) that also enforces the teeth's
    digits strictly increase along the arch: that version assumes the arch order is always
    anatomically correct, which it only is ~99% of the time, and the ~1% miss cascades into
    breaking neighboring already-correct predictions. This weaker "no duplicates" version doesn't
    depend on ordering being right at all, so it has no such failure mode - on a held-out test
    set it cut upper-jaw error count by ~77% with zero regressions, vs. the stricter DP making
    things worse than doing nothing.

    probs: (n, 6) softmax probabilities, one row per tooth, already in left-to-right arch order.
    group_keys: length-n sequence of hashable keys - contiguous runs of equal keys are treated as
    one quadrant (e.g. js/geometry.js's per-tooth `mirror` boolean in production, or the GT FDI
    tens digit during evaluation - either works, since the point is only "which teeth share a
    quadrant", not what the key itself means).

    Returns an (n,) array of 0-indexed digit predictions (0 = last-digit "1", ..., 5 = "6").
    """
    n = probs.shape[0]
    final = probs.argmax(axis=1).copy()
    if n == 0:
        return final

    for a, b in _group_ranges(group_keys, n):
        size = b - a
        # Nothing to deduplicate; and >6 teeth can't fit the 6 last-digit slots a real quadrant
        # has, so over-detection past that point is left as independent argmax rather than
        # forcing a nonsensical assignment.
        if size <= 1 or size > 6:
            continue
        block_preds = final[a:b]
        if len(set(block_preds.tolist())) == size:
            continue  # already unique, no conflict to resolve
        cost = -np.log(np.clip(probs[a:b], 1e-8, 1.0))
        row_ind, col_ind = linear_sum_assignment(cost)
        resolved = block_preds.copy()
        resolved[row_ind] = col_ind
        final[a:b] = resolved

    return final


def resolve_quadrant_monotonic(probs, group_keys):
    """Per-quadrant weighted-DP assignment: like resolve_quadrant_duplicates, but additionally
    forces the assigned digits to strictly increase away from the midline (the real anatomical
    constraint - FDI digits run 1..6 outward in every quadrant), not just be duplicate-free.
    The first group encountered is assumed to run posterior-to-midline (digit descends as
    position increases - true for whichever quadrant a Held-Karp arch order reaches first) and
    every group after it midline-to-posterior (digit ascends); solved as a small O(6) longest-
    weighted-increasing-subsequence DP per group (reverse the row order first for a descending
    group, solve ascending, then un-reverse).

    On a held-out test this ends up *worse* than the weaker duplicate-only version despite being
    the "more correct" constraint on paper: Held-Karp's arch ordering is only right ~99% of the
    time, and forcing strict monotonicity on top of a wrong order doesn't just miss that one
    tooth, it can cascade and break neighboring teeth that were already correct. Kept here as a
    comparison point, not because it won.

    probs, group_keys: see resolve_quadrant_duplicates. Returns an (n,) array of 0-indexed digit
    predictions.
    """
    n = probs.shape[0]
    final = probs.argmax(axis=1).copy()
    if n == 0:
        return final

    NEG = -1e18
    ranges = _group_ranges(group_keys, n)
    # The first of two groups runs posterior-to-midline (descending); the second, midline-to-
    # posterior (ascending). A lone group (no quadrant boundary crossed in this arch) defaults
    # to ascending - matches the convention CANONICAL_ORDER's second half always uses.
    for group_idx, (a, b) in enumerate(ranges):
        size = b - a
        if size <= 1 or size > 6:
            continue
        ascending = group_idx > 0 or len(ranges) == 1
        block = probs[a:b]
        logp = np.log(np.clip(block, 1e-8, 1.0))
        if not ascending:
            logp = logp[::-1]

        dp = np.full((size, 6), NEG)
        choice = np.full((size, 6), -1, dtype=int)
        dp[0, :] = logp[0, :]
        for i in range(1, size):
            # running_best must reflect max(dp[i-1][d']) over d' < d (strictly less) at the
            # point dp[i][d] is computed, so dp[i-1][d] itself has to be folded in AFTER, not
            # before, each iteration's assignment - otherwise d'==d could feed into dp[i][d]
            # and silently downgrade "strictly increasing" to "non-decreasing".
            running_best = NEG
            running_best_d = -1
            for d in range(6):
                if running_best > NEG:
                    dp[i, d] = running_best + logp[i, d]
                    choice[i, d] = running_best_d
                if dp[i - 1, d] > running_best:
                    running_best = dp[i - 1, d]
                    running_best_d = d

        best_d = int(np.argmax(dp[size - 1]))
        assign = [0] * size
        d = best_d
        for i in range(size - 1, -1, -1):
            assign[i] = d
            d = choice[i, d]
        if not ascending:
            assign = assign[::-1]
        final[a:b] = assign

    return final


def resolve_quadrant_greedy(probs, group_keys):
    """Per-quadrant greedy assignment: processes the teeth in a quadrant most-confident-first
    (by each tooth's own top softmax probability), each claiming its highest-ranked digit that
    an earlier, more-confident tooth hasn't already taken. Same goal as
    resolve_quadrant_duplicates (no duplicate digit within a quadrant) but a local greedy
    heuristic instead of a globally-optimal Hungarian assignment - cheaper (no linear_sum_
    assignment call) and a useful independent comparison point: if it tracks the Hungarian
    result closely, global optimality wasn't buying much on this data; if it lags, the
    optimality was doing real work.

    probs, group_keys: see resolve_quadrant_duplicates. Returns an (n,) array of 0-indexed digit
    predictions.
    """
    n = probs.shape[0]
    final = probs.argmax(axis=1).copy()
    if n == 0:
        return final

    for a, b in _group_ranges(group_keys, n):
        size = b - a
        if size <= 1 or size > 6:
            continue
        block_preds = final[a:b]
        if len(set(block_preds.tolist())) == size:
            continue  # already unique, no conflict to resolve

        block = probs[a:b]
        order = sorted(range(size), key=lambda i: -block[i].max())
        taken = set()
        resolved = block_preds.copy()
        for i in order:
            ranked_digits = np.argsort(-block[i])
            for d in ranked_digits:
                if d not in taken:
                    resolved[i] = d
                    taken.add(int(d))
                    break
        final[a:b] = resolved

    return final
