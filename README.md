# GF(2) Binary Matrix Diameter Search (CUDA Optimized)

This repository contains high-performance CUDA kernels developed to search for the **diameter of a 9x9 matrix space over GF(2)**. The code was specifically designed to support computational research for an academic paper.

## 🔬 Research Context
The goal was to explore the structural properties of 9x9 binary matrices. Due to the massive search space, standard CPU implementations were insufficient. This project focuses on the computational engine that made the data filtration and diameter search feasible through GPU acceleration.

## 🛠 Technical Implementation & Optimization
- **Search Space Filtration:** Each iteration of the code (v1, v2, v3) introduced more aggressive heuristics and bitwise filters to prune the search space.
- **GF(2) Arithmetic on GPU:** Custom CUDA kernels for binary operations (XOR/AND) to bypass the limitations of floating-point optimized libraries.
- **Performance Scaling:** Optimized for handling the combinatorial complexity of 9x9 matrix transformations.
- **Python-CUDA Pipeline:** Integrated via `CuPy`'s `RawModule` to combine Python's flexibility for research logic with CUDA's raw power for the heavy lifting.

## 📁 Project Structure & Evolution

The project evolved through three main iterations, each significantly narrowing down the search space for the 9x9 matrix diameter.

### Phase 1: Initial Batch Processing
- **Files:** `src/v1.py` & `src/kernels/gf2_matmul.cu`
- **Logic:** Basic implementation of binary matrix multiplication on GPU. Used for broad scanning of the matrix space to identify potential candidates.

### Phase 2: Inverse Consistency Check
- **Files:** `src/v2.py` & `src/kernels/gf2_matmul_1b.cu`
- **Logic:** Introduced simultaneous checking of the matrix and its inverse. This version added a filtration layer that eliminated matrices which didn't meet the research-specific symmetry criteria in GF(2).

### Phase 3: Binary Tree Search & Advanced Pruning
- **Files:** `src/v3.py` & `src/kernels/gf2_matmul_2.cu`
- **Logic:** The most optimized version. 
  - **CUDA:** Implemented `search_binary_tree` directly on the GPU to verify matrices against a pre-calculated batch mask.
  - **Performance:** Significant reduction in GPU-CPU data transfer by moving the filtration logic into the kernel itself.
  - **Goal:** Final filtration for the 9x9 matrix diameter research.

## 📊 Project Scope
Note: This repository contains only the **computational source code** and kernels. Experimental input data and resulting research datasets are excluded.

## 📜 License & Attribution
This project is licensed under the **MIT License**.

**Author: Timotej Kuzma** *Developed for research purposes. If you find this implementation useful for similar mathematical searches, please provide attribution.*
