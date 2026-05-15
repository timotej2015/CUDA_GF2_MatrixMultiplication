# GF(2) Binary Matrix Space Diameter Search (CUDA Optimized)

High-performance CUDA kernels built to search for the **diameter of a 9×9 matrix space over GF(2)**, developed to support an academic research paper.

> **tl;dr for engineers:** Custom GPU kernels for binary matrix multiplication over GF(2), integrated into a Python pipeline via CuPy. Three iterative filtration phases reduce a search space of 2³⁶ candidates down to a tractable set using increasingly aggressive pruning — including block matmul, inverse consistency checks, and binary tree lookup on the GPU.

---

## 🔬 Research Context

The goal was to explore the structural properties of 9×9 binary matrices over GF(2). The search space is far too large for CPU-based enumeration, so the entire filtration pipeline was built to run on GPU — using custom CUDA kernels rather than general-purpose libraries, which are optimized for floating-point and not GF(2) arithmetic.

## 🛠 Key Technical Decisions

- **Custom GF(2) kernels** — standard libraries (cuBLAS etc.) don't support binary arithmetic, so all matrix multiplication uses XOR/AND logic written from scratch in CUDA.
- **Filtration pushed into kernels** — rather than returning raw products to the CPU for evaluation, the pass/fail logic runs inside the kernel, minimizing GPU↔CPU data transfer.
- **Sliding window minibatching** — R-pair iteration is windowed to avoid exceeding GPU thread limits while still processing large candidate sets.
- **Binary tree on GPU** — Phase 3 moves the membership lookup into the kernel via a prebuilt binary search tree over the variable sub-block of the candidate matrices.

---

## 📁 Project Evolution

The project went through three phases, each narrowing the candidate set further.

---

### Phase 1 — Broad Candidate Scan

**Files:** `v1.py` + `kernels/gf2_matmul.cu`

Enumerates all 2³⁶ possible 9×9 block-structured matrices `X` (parameterized by four free 3×3 sub-blocks `a1..a4` over GF(2)) and tests them against a precomputed set of left- and right-multipliers `Λ₁`, `Λ₂`.

A matrix `X` survives if, for **every** `(Λ₁, Λ₂)` pair, the block matrix product

```
Λ₁ @ X @ Λ₂  (mod 2)
```

is non-zero. A single zero product disqualifies `X`.

**Kernel (`matmul_all_batches`):** Each thread handles one `(Λ₁, X, Λ₂)` triple. The product is a block matrix multiplication where each scalar "element" is itself a 3×3 GF(2) matrix. If the result is the zero block matrix, the thread writes `0` to `OUTPUT[x_index]` — a constant-value write repeated by multiple threads, which is safe under CUDA's memory model. Surviving matrices are written to a Wolfram-compatible output file.

---

### Phase 2 — Inverse Consistency Check

**Files:** `v2.py` + `kernels/gf2_matmul_1b.cu`

Takes the Phase 1 survivors (now as flat 9×9 matrices `S` with precomputed GF(2) inverses `S⁻¹`) and a batch of multipliers `R`. For each pair `(R₁, R₂)`, it checks whether the following commutativity-type condition holds:

```
S⁻¹ @ R₁ @ S @ R₂  ==  R₂ @ S⁻¹ @ R₁ @ S  (mod 2)
```

If it holds for **any** pair, `S` is rejected. A valid candidate must break this symmetry for all `(R₁, R₂)`.

**Kernel (`matmul_all_in_minibatch`):** Each thread computes both four-matrix products for its assigned `(R₁, R₂)` pair and compares them element-by-element. If equal, the shared flag `CURRENT_S_IS_GOOD` is set to `0` (constant-value write, safe under data races). The outer Python loop processes one `S` at a time; `R` pairs are windowed into minibatches.

---

### Phase 3 — Binary Tree Search & Mask Pruning

**Files:** `v3.py` + `kernels/gf2_matmul_2.cu`

The most optimized phase. For each remaining candidate `CURRENT_S` and every `(R₁, R₂)` pair, the kernel computes:

```
R₁ @ CURRENT_S @ R₂  (mod 2)
```

It then immediately checks whether this product exists anywhere in `S_BATCH` by traversing a binary search tree built over the 6×6 variable sub-block (rows 3–8, cols 3–8) of each candidate. If a match is found at index `k`, `S_BATCH[k]` is eliminated from the mask and the eliminating triple `(R₁, S, R₂)` is recorded.

**Kernel (`matmul_all_in_minibatch`):** Before tree traversal, the kernel checks the product's fixed sub-block against a hardcoded reference matrix — an early-exit that avoids tree lookup for the vast majority of products. On a hit, the thread writes `0` to `S_BATCH_MASK[k]` and flags itself in `ELIMINATED_PRODUCTS_MASK` so the CPU can retrieve the eliminating trios after the kernel finishes.

After all candidates are processed, survivors are saved with a freshly built binary tree into a new `.npz` for the next iteration.

---

## 📊 Data Flow

```
2^36 X candidates
    └─(v1: non-zero block matmul filter)──► good_x_matrices.wl
        └─(build S + S⁻¹ pairs)───────────► GOODINVERSES.npz
            └─(v2: inverse consistency)────► S_matrices_output.txt
                └─(build binary tree)──────► S_BATCH_BINARY_TREE.npz
                    └─(v3: tree pruning)───► filtered_data.npz
                                             eliminated_products.txt
```

---

## 📜 License & Attribution

This project is licensed under the **MIT License**.

**Author: Timotej Kuzma** — developed for research purposes. If you find this implementation useful for similar mathematical searches, please provide attribution.
