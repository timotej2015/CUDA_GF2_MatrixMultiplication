# GF(2) Binary Matrix Diameter Search (CUDA Optimized)

This repository contains high-performance CUDA kernels developed to search for the **diameter of a 9x9 matrix space over GF(2)**. The code was specifically designed to support computational research for an academic paper.

## 🔬 Research Context
The goal was to explore the structural properties of 9x9 binary matrices. Due to the massive search space, standard CPU implementations were insufficient. This project focuses on the computational engine that made the data filtration and diameter search feasible through GPU acceleration.

## 🛠 Technical Implementation & Optimization
- **Search Space Filtration:** Each iteration of the code (v1, v2, etc.) introduced more aggressive heuristics and bitwise filters to prune the search space.
- **GF(2) Arithmetic on GPU:** Custom CUDA kernels for binary operations (XOR/AND) to bypass the limitations of floating-point optimized libraries.
- **Performance Scaling:** Optimized for handling the combinatorial complexity of 9x9 matrix transformations.
- **Python-CUDA Pipeline:** Integrated via `CuPy`'s `RawModule` to combine Python's flexibility for research logic with CUDA's raw power for the heavy lifting.

## 📊 Project Scope
Note: This repository contains only the **computational source code** and kernels. Experimental input data and resulting research datasets are excluded.

## 📜 License & Attribution
This project is licensed under the **MIT License**.

**Author: Timotej Kuzma** *Developed for research purposes. If you find this implementation useful for similar mathematical searches, please provide attribution.*
