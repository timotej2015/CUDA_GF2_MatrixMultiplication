import cupy as cp
from pathlib import Path
import time
import numpy as np
import re

BASE_DIR = Path(__file__).parent
cuda_code = (BASE_DIR / "kernels" / "gf2_matmul_1b.cu").read_text()

module = cp.RawModule(code=cuda_code)
kernel = module.get_function("matmul_all_in_minibatch")

def EvaluateS_GPU(CURRENT_S, CURRENT_S_INV, R_BATCH, r1_batch_window_length=100, r2_batch_window_length=100, blockDim=32):
    # Dvojna zanka čez R_BATCH v oknih, da ne bo preveč niti za GPU
    R_BATCH_LENGTH = R_BATCH.shape[0]
    current_S_isAGoodOne = cp.array([1], dtype=cp.int8)

    for r1_start in range(0, R_BATCH_LENGTH, r1_batch_window_length):
        r1_end = min(r1_start + r1_batch_window_length, R_BATCH_LENGTH)
        R1_MINIBATCH = R_BATCH[r1_start:r1_end]
        R1_MINIBATCH_LEN = r1_end - r1_start

        for r2_start in range(0, R_BATCH_LENGTH, r2_batch_window_length):
            r2_end = min(r2_start + r2_batch_window_length, R_BATCH_LENGTH)
            R2_MINIBATCH = R_BATCH[r2_start:r2_end]
            R2_MINIBATCH_LEN = r2_end - r2_start

            NUMBER_OF_THREADS = R1_MINIBATCH_LEN * R2_MINIBATCH_LEN
            NUMBER_OF_BLOCKS = (NUMBER_OF_THREADS + blockDim - 1) // blockDim   # isto kot ceil(NUMBER_OF_THREADS / blockdim)

            kernel((NUMBER_OF_BLOCKS,), (blockDim,), (CURRENT_S, CURRENT_S_INV, R1_MINIBATCH, R2_MINIBATCH, current_S_isAGoodOne, R1_MINIBATCH_LEN, R2_MINIBATCH_LEN))
            cp.cuda.runtime.deviceSynchronize()


    if(current_S_isAGoodOne[0] == 0):
        return False
    else:
        return True


def Matmul9x9_CPU(A, B, C, D):
    # Prvi produkt: A @ B
    temp1 = cp.zeros((9, 9), dtype=cp.uint8)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                temp1[i, j] += A[i, k] * B[k, j]
                temp1 %= 2

    # Drugi produkt: (A @ B) @ C
    temp2 = cp.zeros((9, 9), dtype=cp.uint8)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                temp2[i, j] += temp1[i, k] * C[k, j]
                temp2 %= 2

    # Tretji produkt: ((A @ B) @ C) @ D
    result = cp.zeros((9, 9), dtype=cp.uint8)
    for i in range(9):
        for j in range(9):
            for k in range(9):
                result[i, j] += temp2[i, k] * D[k, j]
                result %= 2

    return result







def load_all_batches(
    goodinverse_npz_path,
    rk_npz_path,
    use_gpu=True
):
    goodinverse_npz_path = Path(goodinverse_npz_path)
    rk_npz_path = Path(rk_npz_path)

    # --- GOOD INVERSES (S, S_INV)
    gi_data = np.load(goodinverse_npz_path)

    if "S_BATCH" not in gi_data or "S_INV_BATCH" not in gi_data:
        raise ValueError("GOODINVERSES npz must contain S_BATCH and S_INV_BATCH")

    S_BATCH_np = gi_data["S_BATCH"]
    S_INV_BATCH_np = gi_data["S_INV_BATCH"]

    # --- R BATCH
    rk_data = np.load(rk_npz_path)

    if "RK" not in rk_data:
        raise ValueError("RK npz must contain array named 'RK'")

    R_BATCH_np = rk_data["RK"]

    # --- sanity checks
    if S_BATCH_np.shape != S_INV_BATCH_np.shape:
        raise ValueError("S_BATCH and S_INV_BATCH must have identical shapes")

    if S_BATCH_np.shape[1:] != (9, 9):
        raise ValueError("S_BATCH matrices must be 9x9")

    if R_BATCH_np.shape[1:] != (9, 9):
        raise ValueError("R_BATCH matrices must be 9x9")

    # --- move to GPU if requested
    if use_gpu:
        S_BATCH = cp.asarray(S_BATCH_np)
        S_INVERSE_BATCH = cp.asarray(S_INV_BATCH_np)
        R_BATCH = cp.asarray(R_BATCH_np)
    else:
        S_BATCH = S_BATCH_np
        S_INVERSE_BATCH = S_INV_BATCH_np
        R_BATCH = R_BATCH_np

    print("✔ All batches loaded successfully")
    print(f"  S_BATCH shape: {S_BATCH.shape}")
    print(f"  S_INVERSE_BATCH shape: {S_INVERSE_BATCH.shape}")
    print(f"  R_BATCH shape: {R_BATCH.shape}")

    return S_BATCH, S_INVERSE_BATCH, R_BATCH

def load_S_and_SINV_from_txt(
    S_txt_path,
    S_INV_txt_path,
    use_gpu=True
):
    """
    Prebere datoteki z 9x9 matrikami v obliki:
    [
      [ ... ],
      ...
      [ ... ]
    ],
    
    in vrne S_BATCH in S_INV_BATCH
    """

    def parse_txt(path):
        path = Path(path)
        text = path.read_text()

        # Najdi vse bloke [ [ ... ] ]
        matrix_blocks = re.findall(
            r"\[\s*(?:\[\s*[01,\s]+\]\s*,?\s*){9}\]",
            text,
            flags=re.MULTILINE
        )

        matrices = []

        for block in matrix_blocks:
            rows = re.findall(r"\[\s*([01,\s]+)\s*\]", block)
            if len(rows) != 9:
                raise ValueError("Found non-9x9 matrix")

            mat = np.zeros((9, 9), dtype=np.uint8)
            for i, row in enumerate(rows):
                mat[i] = np.fromstring(row, sep=",", dtype=np.uint8)

            matrices.append(mat)

        return np.stack(matrices, axis=0)

    # --- parse both files
    S_np = parse_txt(S_txt_path)
    S_INV_np = parse_txt(S_INV_txt_path)

    # --- sanity check
    if S_np.shape != S_INV_np.shape:
        raise ValueError(
            f"Mismatch: S_BATCH {S_np.shape}, S_INV_BATCH {S_INV_np.shape}"
        )

    print("✔ TXT files loaded")
    print(f"  Matrices found: {S_np.shape[0]}")
    print(f"  Shape: {S_np.shape}")

    # --- move to GPU if requested
    if use_gpu:
        return cp.asarray(S_np), cp.asarray(S_INV_np)
    else:
        return S_np, S_INV_np



def append_9x9_to_txt(matrix, txt_path):
    """
    Append a single 9x9 CuPy matrix to a text file in the format:
    [ [ 0, 0, 0, ..., 0 ],
      ...
      [ 0, 0, 0, ..., 0 ] ],
    
    Parametri:
    - matrix: cp.ndarray 9x9
    - txt_path: Path ali str
    """
    txt_path = Path(txt_path)
    
    # Pretvori CuPy matriko v Numpy, če še ni
    if isinstance(matrix, cp.ndarray):
        mat = cp.asnumpy(matrix)
    else:
        mat = matrix

    if mat.shape != (9,9):
        raise ValueError("Matrix must be 9x9")
    
    # Odpri datoteko v append načinu
    with open(txt_path, "a") as f:
        f.write("[\n")
        for i, row in enumerate(mat):
            row_str = ", ".join(str(int(x)) for x in row)
            if i < 8:
                f.write(f"[ {row_str} ],\n")
            else:
                f.write(f"[ {row_str} ]\n")
        f.write("],\n\n")  # Zaključi posamezno matriko s vejico in novo vrstico






# --- GLAVNA LOGIKA z izpisom napredka

# Naloži vsebino datotek v RAM
BASE_DIR = Path(__file__).parent

S_BATCH, S_INVERSE_BATCH, R_BATCH = load_all_batches(
    BASE_DIR / "InputFiles" / "GOODINVERSESALL.npz",
    BASE_DIR / "InputFiles" / "RK1.npz",
    use_gpu=True
)

# Naloži tmp datoteke, kjer so matrike že močno filtrirane
S_BATCH, S_INVERSE_BATCH = load_S_and_SINV_from_txt(
    BASE_DIR / "InputFiles" / "S_matrices_tmp.txt",
    BASE_DIR / "InputFiles" / "S_INV_matrices_tmp.txt",
    use_gpu=True
)


# Poti za shranjevanje
outputS_filepath = Path(BASE_DIR / "S_matrices_output.txt")
outputS_INV_filepath = Path(BASE_DIR / "S_INV_matrices_output.txt")







# V R_BATCH ohrani samo prvih 10% matrik
#num_keep = max(1, int(0.10 * R_BATCH.shape[0]))  # vsaj 1 element, če slučajno manj kot 10
#R_BATCH = R_BATCH[:num_keep]




# Parametri za performance tuning
blockDim = 256
r1BatchWindowLength = 1000
r2BatchWindowLength = 1000

S_BATCH_LENGTH = S_BATCH.shape[0]
start_time = time.time()
found_count = 0  # števec za dobre matrike

for current_S_index in range(S_BATCH_LENGTH):
    CURRENT_S = S_BATCH[current_S_index]
    CURRENT_S_INV = S_INVERSE_BATCH[current_S_index]

    if (EvaluateS_GPU(CURRENT_S, CURRENT_S_INV, R_BATCH, r1BatchWindowLength, r2BatchWindowLength, blockDim) == True):
        append_9x9_to_txt(CURRENT_S, outputS_filepath)
        append_9x9_to_txt(CURRENT_S_INV, outputS_INV_filepath)
        found_count += 1
        print(CURRENT_S)

    # Izpis napredka
    elapsed_time = time.time() - start_time
    avg_time_per_S = elapsed_time / (current_S_index + 1)
    remaining_S = S_BATCH_LENGTH - (current_S_index + 1)
    eta_seconds = remaining_S * avg_time_per_S
    percent_done = (current_S_index + 1) / S_BATCH_LENGTH * 100
    percent_found = found_count / (current_S_index + 1) * 100  # procent najdenih

    if current_S_index % 10 == 0:  # izpis na vsakih 10 S
        print(
            f"[S index: {current_S_index+1}/{S_BATCH_LENGTH}] "
            f"{percent_done:.4f}% done | "
            f"Elapsed: {int(elapsed_time//3600):02d}:"
            f"{int((elapsed_time%3600)//60):02d}:"
            f"{int(elapsed_time%60):02d} | "
            f"ETA: {int(eta_seconds//3600):02d}:"
            f"{int((eta_seconds%3600)//60):02d}:"
            f"{int(eta_seconds%60):02d} | "
            f"Found: {found_count} matrices ({percent_found:.2f}%)"
        )