"""
Problem loaders and random instance generators.

Supported:
    - G-set Max-Cut:           load_gset(path)
    - TSPLIB TSP (EUC_2D/ATT): load_tsplib(path, B=1.0)
    - Semi-prime factoring:     build_factoring_problem(M)
    - Erdos-Renyi Max-Cut:      random_er_maxcut(n, p, sigma, seed)
    - Sherrington-Kirkpatrick:  random_sk(n, seed)

Note on TSP scaling
-------------------
The n^2-spin QUBO formulation of Section 3.1.4 is correct but highly
inefficient for SA-class solvers: N=n^2 spins with O(n^3) nonzero J
entries makes the feasibility subspace exponentially small. In
practice, SA finds optimal tours on tiny instances (n <= 15) within
minutes, on moderate instances (n <= 30) with difficulty, and
essentially never on berlin52 or larger within reasonable time. This
is a well-known limitation of the QUBO encoding (see Lucas 2014,
Front. Phys.), not a bug in the solver. For thesis benchmarking we
restrict TSP to small instances where the encoding is still
instructive.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from isim import Problem, preprocess


# ===========================================================================
# Max-Cut: G-set loader and random generators
# ===========================================================================

def load_gset(path, name: str = None, normalize: bool = False) -> Problem:
    """Load a G-set instance.

    G-set format (plain text):
        first line: n_vertices n_edges
        subsequent lines: i j w   (1-indexed)
    """
    path = Path(path)
    name = name or path.stem
    with open(path, "r") as f:
        tokens = f.read().split()
    it = iter(tokens)
    n = int(next(it))
    m = int(next(it))
    rows = np.empty(2 * m, dtype=np.int64)
    cols = np.empty(2 * m, dtype=np.int64)
    vals = np.empty(2 * m, dtype=np.float64)
    edge_sum = 0.0
    k = 0
    for _ in range(m):
        i = int(next(it)) - 1
        j = int(next(it)) - 1
        w = float(next(it))
        # J_ij = -w/2 (stored symmetrically)
        rows[k] = i; cols[k] = j; vals[k] = -0.5 * w; k += 1
        rows[k] = j; cols[k] = i; vals[k] = -0.5 * w; k += 1
        edge_sum += w
    J = sp.coo_matrix((vals[:k], (rows[:k], cols[:k])), shape=(n, n)).tocsr()
    h = np.zeros(n, dtype=np.float64)
    J, h = preprocess(J, h, normalize=normalize)
    prob = Problem(name=name, n=n, J=J, h=h)
    prob.meta.update({
        "kind": "maxcut",
        "edge_sum": edge_sum,
        "raw_n": n,
        "raw_m": m,
    })
    return prob


def random_er_maxcut(n: int, p: float = 0.1, sigma: float = 1.0,
                     seed: int = 0, name: str = None,
                     normalize: bool = False) -> Problem:
    """Erdos-Renyi Max-Cut: each of n*(n-1)/2 edges present with prob p,
    weight drawn from N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    tri_i, tri_j = np.triu_indices(n, k=1)
    mask = rng.random(len(tri_i)) < p
    src = tri_i[mask]
    dst = tri_j[mask]
    w = rng.normal(scale=sigma, size=mask.sum())
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([-0.5 * w, -0.5 * w])
    J = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    h = np.zeros(n, dtype=np.float64)
    J, h = preprocess(J, h, normalize=normalize)
    name = name or f"ER_n{n}_p{p:g}_s{seed}"
    prob = Problem(name=name, n=n, J=J, h=h)
    prob.meta.update({
        "kind": "maxcut",
        "edge_sum": float(w.sum()),
        "er_params": {"n": n, "p": p, "sigma": sigma, "seed": seed},
    })
    return prob


def random_sk(n: int, seed: int = 0, name: str = None,
              normalize: bool = False) -> Problem:
    """Sherrington-Kirkpatrick spin glass: all-to-all J_ij ~ N(0, 1/sqrt(n))."""
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=1.0 / np.sqrt(n), size=(n, n))
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    J = sp.csr_matrix(A)
    h = np.zeros(n, dtype=np.float64)
    J, h = preprocess(J, h, normalize=normalize)
    name = name or f"SK_n{n}_s{seed}"
    prob = Problem(name=name, n=n, J=J, h=h)
    prob.meta.update({
        "kind": "sk",
        "sk_params": {"n": n, "seed": seed},
    })
    return prob


# ===========================================================================
# Traveling Salesman Problem (TSPLIB)
# ===========================================================================

# Guard threshold. The n^2-spin encoding produces N = n^2 variables with
# extremely low feasibility density; for n > this threshold, expect very
# poor SA convergence and multi-hour run times. Explicit opt-in required.
TSP_DEFAULT_MAX_N = 20


def _geo_to_radians(x: np.ndarray) -> np.ndarray:
    """TSPLIB95 GEO coordinate conversion. Input is in DDD.MM format;
    C `(int)` truncates toward zero (so np.trunc, not np.floor) to
    keep the sign of minutes consistent for negative coordinates."""
    PI = 3.141592
    deg = np.trunc(x)
    mn = x - deg
    return PI * (deg + 5.0 * mn / 3.0) / 180.0


def _parse_explicit_edge_weights(block: str, n: int, fmt: str) -> np.ndarray:
    """Decode an EDGE_WEIGHT_SECTION block for EDGE_WEIGHT_TYPE=EXPLICIT.
    Supports every variant listed in the TSPLIB95 specification."""
    tokens = block.split()
    vals = [float(t) for t in tokens]
    D = np.zeros((n, n), dtype=np.float64)
    fmt = fmt.upper()
    if fmt == "FULL_MATRIX":
        expected = n * n
        if len(vals) < expected:
            raise ValueError(f"FULL_MATRIX: expected {expected} values, "
                             f"got {len(vals)}")
        D = np.array(vals[:expected], dtype=np.float64).reshape(n, n)
    elif fmt == "LOWER_DIAG_ROW":
        k = 0
        for i in range(n):
            for j in range(i + 1):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "UPPER_DIAG_ROW":
        k = 0
        for i in range(n):
            for j in range(i, n):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "LOWER_ROW":
        k = 0
        for i in range(1, n):
            for j in range(i):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "UPPER_ROW":
        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "LOWER_DIAG_COL":
        k = 0
        for j in range(n):
            for i in range(j, n):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "UPPER_DIAG_COL":
        k = 0
        for j in range(n):
            for i in range(j + 1):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "LOWER_COL":
        k = 0
        for j in range(n - 1):
            for i in range(j + 1, n):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    elif fmt == "UPPER_COL":
        k = 0
        for j in range(1, n):
            for i in range(j):
                D[i, j] = vals[k]; D[j, i] = vals[k]; k += 1
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_FORMAT {fmt} not supported")
    np.fill_diagonal(D, 0.0)
    return D


def _parse_tsplib(path):
    """Parse a TSPLIB instance and return (name, distance_matrix).

    Supported EDGE_WEIGHT_TYPE:
        EUC_2D, CEIL_2D, ATT, MAX_2D, MAN_2D, GEO (from NODE_COORD_SECTION)
        EXPLICIT (from EDGE_WEIGHT_SECTION) with any of the
        FULL_MATRIX / LOWER_DIAG_ROW / UPPER_DIAG_ROW / LOWER_ROW /
        UPPER_ROW / LOWER_DIAG_COL / UPPER_DIAG_COL / LOWER_COL /
        UPPER_COL formats.
    """
    path = Path(path)
    text = path.read_text()
    name_match = re.search(r"NAME\s*:\s*(\S+)", text)
    name = name_match.group(1) if name_match else path.stem
    # Some TSPLIB mirrors store the filename (including a '.tsp'
    # extension) in the NAME header rather than the canonical instance
    # identifier. Strip the extension so downstream lookups against
    # canonical tables (e.g. published optimal tour lengths keyed by
    # 'ulysses16', not 'ulysses16.tsp') succeed.
    for ext in (".tsp", ".TSP", ".atsp", ".ATSP"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    dim_match = re.search(r"DIMENSION\s*:\s*(\d+)", text)
    if not dim_match:
        raise ValueError("TSPLIB file missing DIMENSION")
    n = int(dim_match.group(1))
    ew_match = re.search(r"EDGE_WEIGHT_TYPE\s*:\s*(\S+)", text)
    ew_type = ew_match.group(1).upper() if ew_match else "EUC_2D"

    # Explicit distance matrix branch
    if ew_type == "EXPLICIT":
        fmt_match = re.search(r"EDGE_WEIGHT_FORMAT\s*:\s*(\S+)", text)
        if not fmt_match:
            raise ValueError("EXPLICIT instance missing EDGE_WEIGHT_FORMAT")
        fmt = fmt_match.group(1)
        if "EDGE_WEIGHT_SECTION" not in text:
            raise ValueError("EXPLICIT instance missing EDGE_WEIGHT_SECTION")
        block = text.split("EDGE_WEIGHT_SECTION", 1)[1]
        block = re.split(r"EOF|DISPLAY_DATA_SECTION|NODE_COORD_SECTION",
                         block)[0]
        D = _parse_explicit_edge_weights(block, n, fmt)
        return name, D

    # Coordinate-based branch
    if "NODE_COORD_SECTION" not in text:
        raise NotImplementedError(
            f"EDGE_WEIGHT_TYPE {ew_type} requires NODE_COORD_SECTION")
    coord_block = text.split("NODE_COORD_SECTION", 1)[1]
    coord_block = re.split(r"EOF|DISPLAY_DATA_SECTION|EDGE_WEIGHT_SECTION",
                           coord_block)[0]
    coords = np.zeros((n, 2), dtype=np.float64)
    read = 0
    for line in coord_block.splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        coords[read, 0] = float(parts[1])
        coords[read, 1] = float(parts[2])
        read += 1
        if read >= n:
            break
    if read != n:
        raise ValueError(f"Expected {n} coords, parsed {read}")

    if ew_type in ("EUC_2D", "CEIL_2D", "ATT"):
        diff = coords[:, None, :] - coords[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        if ew_type == "EUC_2D":
            D = np.round(r)
        elif ew_type == "ATT":
            # Pseudo-Euclidean per TSPLIB95 spec: r = sqrt((dx^2+dy^2)/10)
            # then t = nint(r); if t < r use t+1, else use t. Simplified
            # to ceil for monotonic coordinates (matches reference impl).
            D = np.ceil(r / np.sqrt(10.0))
        elif ew_type == "CEIL_2D":
            D = np.ceil(r)
    elif ew_type == "MAX_2D":
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.round(np.max(np.abs(diff), axis=-1))
    elif ew_type == "MAN_2D":
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.round(np.sum(np.abs(diff), axis=-1))
    elif ew_type == "GEO":
        # Great-circle distance on a sphere of radius RRR km. The
        # reference C code applies a `+1.0` fudge and truncates toward
        # zero; arccos argument is clamped for numerical safety.
        RRR = 6378.388
        lat = _geo_to_radians(coords[:, 0])
        lon = _geo_to_radians(coords[:, 1])
        q1 = np.cos(lon[:, None] - lon[None, :])
        q2 = np.cos(lat[:, None] - lat[None, :])
        q3 = np.cos(lat[:, None] + lat[None, :])
        arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
        arg = np.clip(arg, -1.0, 1.0)
        D = np.trunc(RRR * np.arccos(arg) + 1.0)
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_TYPE {ew_type} not supported")

    D = D.astype(np.float64)
    np.fill_diagonal(D, 0.0)
    return name, D


def build_tsp_problem(D: np.ndarray, B: float = 1.0, A: float = None,
                      name: str = "TSP",
                      allow_large: bool = False) -> Problem:
    """Build the TSP Ising formulation (Section 3.1.4).

    Variables x_{v,j} in {0, 1} indexed as (v * n + j).
    Path runs position 0 -> 1 -> ... -> n-1 -> 0 (cyclic).
    Ising spin s = 2x - 1.
    Penalty A must satisfy A > B * max(d) for ground-state feasibility.

    For n > TSP_DEFAULT_MAX_N (20), raises ValueError unless
    allow_large=True. The n^2-spin QUBO encoding is known to converge
    extremely slowly under plain SA for larger n; this guard prevents
    accidental multi-hour runs.
    """
    n = int(D.shape[0])
    N = n * n
    if n > TSP_DEFAULT_MAX_N and not allow_large:
        raise ValueError(
            f"TSP n={n} exceeds default guard (n<={TSP_DEFAULT_MAX_N}). "
            f"The n^2-spin QUBO encoding converges very slowly under SA "
            f"for large n. Pass allow_large=True to proceed (expect "
            f"hour-scale runs and poor convergence).")
    D = np.asarray(D, dtype=np.float64)
    if A is None:
        A = float(2.0 * B * D.max())

    # Build QUBO as COO and convert once, vectorized.
    # Variable index: idx(v, j) = v * n + j
    rows = []
    cols = []
    vals = []

    # Row constraints: (1 - sum_j x_{v,j})^2 = 1 - 2 sum_j x_{v,j}
    #                                          + (sum_j x_{v,j})^2
    # (sum x_{v,j})^2 = sum_j x_{v,j} + 2 sum_{j<k} x_{v,j} x_{v,k}
    # So contribution: -A on each diagonal, +2A on each within-row pair
    # Vectorized:
    for v in range(n):
        idxs = np.arange(v * n, (v + 1) * n)
        # Diagonal: -A * n terms
        rows.append(idxs)
        cols.append(idxs)
        vals.append(np.full(n, -A))
        # Within-row upper triangle pairs
        pi, pj = np.triu_indices(n, k=1)
        a = idxs[pi]
        b = idxs[pj]
        rows.append(a); cols.append(b); vals.append(np.full(len(a), A))
        rows.append(b); cols.append(a); vals.append(np.full(len(a), A))

    # Column constraints (same structure on columns j)
    for j in range(n):
        idxs = np.arange(j, n * n, n)  # indices (v, j) for v=0..n-1
        rows.append(idxs); cols.append(idxs); vals.append(np.full(n, -A))
        pi, pj = np.triu_indices(n, k=1)
        a = idxs[pi]; b = idxs[pj]
        rows.append(a); cols.append(b); vals.append(np.full(len(a), A))
        rows.append(b); cols.append(a); vals.append(np.full(len(a), A))

    # Cost term B * sum_{u!=v} d_{uv} sum_j x_{u,j} x_{v,j+1}
    # Vectorize over j for each (u, v) pair
    j_arr = np.arange(n)
    jp1 = (j_arr + 1) % n
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            w = 0.5 * B * D[u, v]
            a = u * n + j_arr
            b = v * n + jp1
            rows.append(a); cols.append(b); vals.append(np.full(n, w))
            rows.append(b); cols.append(a); vals.append(np.full(n, w))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    Q_coo = sp.coo_matrix((vals, (rows, cols)), shape=(N, N))
    Q = Q_coo.toarray()  # dense is O(n^4) memory; for n<=20, N<=400, manageable

    J, h = _qubo_to_ising(Q)
    J_sparse = sp.csr_matrix(J)
    J_sparse, h = preprocess(J_sparse, h, normalize=False)
    prob = Problem(name=name, n=N, J=J_sparse, h=h)
    prob.meta.update({
        "kind": "tsp",
        "n_cities": n,
        "D": D,
        "A": A,
        "B": B,
    })
    return prob


def load_tsplib(path, B: float = 1.0, A: float = None,
                allow_large: bool = False) -> Problem:
    name, D = _parse_tsplib(path)
    return build_tsp_problem(D, B=B, A=A, name=name, allow_large=allow_large)


def decode_tour(s: np.ndarray, n_cities: int):
    """Decode Ising state to a tour. Returns (tour, feasible_bool).
    Feasibility = each row and each column has exactly one +1."""
    x = ((np.asarray(s, dtype=int) + 1) // 2).reshape(n_cities, n_cities)
    row_ok = np.all(x.sum(axis=1) == 1)
    col_ok = np.all(x.sum(axis=0) == 1)
    feasible = bool(row_ok and col_ok)
    tour = np.argmax(x, axis=0).astype(int)
    return tour, feasible


def tour_length(tour, D) -> float:
    tour = np.asarray(tour, dtype=int)
    n = len(tour)
    return float(sum(D[tour[j], tour[(j + 1) % n]] for j in range(n)))


# ===========================================================================
# Integer factoring (semi-primes, odd p and q)
# ===========================================================================

def suggest_factoring_bp_bq(M: int):
    """Return the minimal (bp, bq) bit budget whose encoding exactly
    covers the factorization of a semiprime M = p*q with p <= q, both
    odd. Uses O(sqrt(M)) trial division. Raises ValueError if M is not
    a valid odd semiprime with both factors >= 3."""
    if M < 9 or M % 2 == 0:
        raise ValueError("M must be odd and >= 9")
    for p in range(3, int(np.sqrt(M)) + 1, 2):
        if M % p == 0:
            q = M // p
            bp = max(2, int(np.ceil(np.log2(p + 1))))
            bq = max(2, int(np.ceil(np.log2(q + 1))))
            return bp, bq
    raise ValueError(f"M={M} has no odd factor in [3, sqrt(M)] "
                     f"(either prime or a power of 2 times an odd prime)")


def build_factoring_problem(M: int, bp: int = None, bq: int = None,
                            name: str = None,
                            penalty: float = None) -> Problem:
    """Factoring M = p * q with p, q odd via multiplication-table encoding.

    Variables (for p_0 = q_0 = 1 fixed):
        p_1, ..., p_{bp-1}           : high bits of p
        q_1, ..., q_{bq-1}           : high bits of q
        z_{i,j} = p_i * q_j          : auxiliary bit-product variables

    Hamiltonian:
        H = (M - pq)^2
          + C * sum_{i,j>=1} (p_i q_j - 2 p_i z_{ij} - 2 q_j z_{ij} + 3 z_{ij})

    Bit-budget allocation
    ---------------------
    If bp and bq are None (default), both are auto-sized via trial
    division so the true factorization fits exactly. This is
    appropriate for benchmarking: the encoding depends on M but the
    solver still discovers the factors without being told them.
    Auto-sizing avoids the common pitfall where the default balanced
    allocation (bp ~ bq ~ bM/2) is too tight to represent unbalanced
    factorizations — e.g., 51 = 3*17 needs bq >= 5 because 2^4 - 1 = 15
    < 17, and a balanced bq = 4 makes success probability
    identically zero regardless of sweep count.

    To simulate the unknown-factor setting or fix a specific allocation
    for ablation, pass bp and bq explicitly. In that case the caller
    is responsible for ensuring 2^bp - 1 >= p and 2^bq - 1 >= q for
    the true factorization (otherwise the QUBO ground state no longer
    corresponds to pq = M).
    """
    if M < 9 or M % 2 == 0:
        raise ValueError("M must be odd and >= 9")
    bM = int(np.ceil(np.log2(M + 1)))

    if bp is None or bq is None:
        try:
            bp_auto, bq_auto = suggest_factoring_bp_bq(M)
            if bp is None:
                bp = bp_auto
            if bq is None:
                bq = bq_auto
        except ValueError:
            # Fall back to the balanced allocation when no odd factor
            # pair is found (e.g., M is prime). This path is not
            # expected for semiprime benchmark targets.
            if bp is None:
                bp = max(2, (bM + 1) // 2)
            if bq is None:
                bq = max(2, bM - bp + 1)

    n_p = bp - 1
    n_q = bq - 1
    n_z = n_p * n_q
    N = n_p + n_q + n_z

    def ip(i): return i - 1
    def iq(j): return n_p + (j - 1)
    def iz(i, j): return n_p + n_q + (i - 1) * n_q + (j - 1)

    Q = np.zeros((N, N), dtype=np.float64)

    # (1) (M - pq)^2 = (T - L(v))^2 where T = M-1 and L is linear in v
    c = np.zeros(N)
    for i in range(1, bp):
        c[ip(i)] = 2 ** i
    for j in range(1, bq):
        c[iq(j)] = 2 ** j
    for i in range(1, bp):
        for j in range(1, bq):
            c[iz(i, j)] = 2 ** (i + j)
    T = M - 1
    for a in range(N):
        Q[a, a] += c[a] * (c[a] - 2 * T)
    nonzero = np.where(c != 0)[0]
    for i_a in range(len(nonzero)):
        a = nonzero[i_a]
        for i_b in range(i_a + 1, len(nonzero)):
            b = nonzero[i_b]
            val = 2.0 * c[a] * c[b]
            Q[a, b] += val
            Q[b, a] += val

    # (2) Penalty enforces z_{i,j} = p_i * q_j
    if penalty is None:
        penalty = float(M * M) + 1.0
    for i in range(1, bp):
        for j in range(1, bq):
            pi = ip(i); qj = iq(j); zij = iz(i, j)
            Q[pi, qj] += penalty
            Q[qj, pi] += penalty
            Q[pi, zij] += -2.0 * penalty
            Q[zij, pi] += -2.0 * penalty
            Q[qj, zij] += -2.0 * penalty
            Q[zij, qj] += -2.0 * penalty
            Q[zij, zij] += 3.0 * penalty

    J, h = _qubo_to_ising(Q)
    J_sparse = sp.csr_matrix(J)
    J_sparse, h = preprocess(J_sparse, h, normalize=False)
    name = name or f"factor_M{M}"
    prob = Problem(name=name, n=N, J=J_sparse, h=h)
    prob.meta.update({
        "kind": "factoring",
        "M": int(M),
        "bp": int(bp),
        "bq": int(bq),
        "penalty": float(penalty),
    })
    return prob


def decode_factors(s: np.ndarray, problem: Problem):
    """Extract (p, q) from an Ising state for a factoring problem.
    Returns (p, q, constraints_satisfied)."""
    meta = problem.meta
    bp, bq = int(meta["bp"]), int(meta["bq"])
    n_p = bp - 1
    n_q = bq - 1
    x = ((np.asarray(s, dtype=int) + 1) // 2)
    p_val = 1
    q_val = 1
    for i in range(1, bp):
        p_val += (2 ** i) * int(x[i - 1])
    for j in range(1, bq):
        q_val += (2 ** j) * int(x[n_p + j - 1])
    ok = True
    for i in range(1, bp):
        for j in range(1, bq):
            pi = int(x[i - 1])
            qj = int(x[n_p + j - 1])
            zij = int(x[n_p + n_q + (i - 1) * n_q + (j - 1)])
            if zij != pi * qj:
                ok = False
                break
        if not ok:
            break
    return int(p_val), int(q_val), bool(ok)


# ===========================================================================
# Shared QUBO to Ising helper
# ===========================================================================

def _qubo_to_ising(Q: np.ndarray):
    """Convert symmetric QUBO (with linear coefficients on diagonal) to
    Ising form H(s) = -(1/2) s^T J s - h^T s. With s = 2x - 1:
        J = -(1/4) * (Q - diag(Q))
        h = -(1/4) * (Q_off @ 1) - (1/2) * diag(Q)
    """
    Q = np.asarray(Q, dtype=np.float64)
    Q_off = Q - np.diag(np.diag(Q))
    J = -0.25 * Q_off
    h = -(0.25 * Q_off.sum(axis=1) + 0.5 * np.diag(Q))
    return J, h
