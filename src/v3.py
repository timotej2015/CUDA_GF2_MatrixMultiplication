import cupy as cp
from pathlib import Path
import time
import numpy as np
import ast

BASE_DIR = Path(__file__).parent
cuda_code = (BASE_DIR / "kernels" / "gf2_matmul_2.cu").read_text()

module = cp.RawModule(code=cuda_code)
kernel = module.get_function("matmul_all_in_minibatch")



def EvaluateS_GPU(CURRENT_S, R_BATCH, S_BATCH_BINARY_TREE, S_BATCH_MASK, eliminatedProducts,
                    r1_batch_window_length=100, r2_batch_window_length=100, blockDim=32):
    # Dvojna zanka čez R_BATCH v oknih, da ne bo preveč niti za GPU
    R_BATCH_LENGTH = R_BATCH.shape[0]

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

            gpu_eliminated_products_mask = cp.zeros((NUMBER_OF_THREADS), dtype=cp.uint8) # vsaka nit, ki izloči S, tu postavi svoj indeks na 1

            kernel((NUMBER_OF_BLOCKS,), (blockDim,), (CURRENT_S, S_BATCH_BINARY_TREE, S_BATCH_MASK, R1_MINIBATCH, R2_MINIBATCH, 
                                                      gpu_eliminated_products_mask, R1_MINIBATCH_LEN, R2_MINIBATCH_LEN))
            cp.cuda.runtime.deviceSynchronize()


            # ✅ Po kernelu najdi thread-e, ki so postavili 1
            threads_eliminated = cp.where(gpu_eliminated_products_mask == 1)[0]

            # Nadaljujemo samo, če smo dejansko našli kakšen produkt
            if len(threads_eliminated) > 0:
                # Izračun indeksov za celoten batch hkrati
                r2_indices = threads_eliminated % R2_MINIBATCH_LEN
                r1_indices = (threads_eliminated // R2_MINIBATCH_LEN) % R1_MINIBATCH_LEN

                # Izberi prave matrike z advanced indexing
                R1_matrices = R1_MINIBATCH[r1_indices]
                R2_matrices = R2_MINIBATCH[r2_indices]
                
                # CURRENT_S je isti za vse, samo navidezno ga razmnožimo za ujemanje dimenzij
                S_matrices = cp.broadcast_to(CURRENT_S, (len(threads_eliminated), 9, 9))

                # Zloži matrike v shape (N, 3, 9, 9)
                trios_gpu = cp.stack([R1_matrices, S_matrices, R2_matrices], axis=1)

                # Prenesi na CPU
                trios_cpu = cp.asnumpy(trios_gpu)

                eliminatedProducts.extend(trios_cpu)



def load_S_batch(goodinverse_npz_path, use_gpu=True):
    goodinverse_npz_path = Path(goodinverse_npz_path)

    data = np.load(goodinverse_npz_path)

    if "S_BATCH" not in data or "S_BATCH_BINARY_TREE" not in data:
        raise ValueError("NPZ must contain S_BATCH and S_BATCH_BINARY_TREE")

    S_BATCH_np = data["S_BATCH"].astype(np.uint8)
    BINARY_TREE_np = data["S_BATCH_BINARY_TREE"].astype(np.int32)

    if use_gpu:
        S_BATCH = cp.asarray(S_BATCH_np)
        BINARY_TREE = cp.asarray(BINARY_TREE_np)
    else:
        S_BATCH = S_BATCH_np
        BINARY_TREE = BINARY_TREE_np

    print("✔ Data loaded successfully")
    print(f"  S_BATCH shape: {S_BATCH.shape}")
    print(f"  BINARY_TREE shape: {BINARY_TREE.shape}")

    return S_BATCH, BINARY_TREE



def createBinaryTreeArray_CPU(S_BATCH, row_offset=3, col_offset=3, size=6):
    N = len(S_BATCH)
    max_nodes = N * size * size + 1

    tree = np.full((max_nodes, 3), -1, dtype=np.int32)

    next_free = 1  # root je 0
    
    for idx, S in enumerate(S_BATCH):

        current_idx = 0

        for i in range(size):
            for j in range(size):
                bit = int(S[row_offset + i][col_offset + j])

                col = 0 if bit == 0 else 1

                if tree[current_idx, col] == -1:
                    tree[current_idx, col] = next_free
                    next_free += 1

                current_idx = tree[current_idx, col]

        tree[current_idx, 2] = idx


    # odrežemo neuporabljen del
    tree = tree[:next_free]

    return tree


def searchBinaryTreeArray_CPU(tree, S, row_offset=3, col_offset=3, size=6):
    current_idx = 0
    for i in range(size):
        for j in range(size):
            bit = int(S[row_offset + i][col_offset + j])
            if bit == 0:
                if tree[current_idx, 0] == -1:
                    return -1
                current_idx = tree[current_idx, 0]
            else:
                if tree[current_idx, 1] == -1:
                    return -1
                current_idx = tree[current_idx, 1]
    return tree[current_idx, 2]


def filterAndSaveAll(S_BATCH, S_BATCH_MASK, R_BATCH):
    print("Ustvarjanje filtrirane NPZ datoteke ...")
    
    # Prenos maske na CPU, če je na GPU
    mask_cpu = cp.asnumpy(S_BATCH_MASK) if isinstance(S_BATCH_MASK, cp.ndarray) else S_BATCH_MASK

    # Boolean maska: True = ohranimo, False = izločimo
    mask_bool = mask_cpu == 1

    # Prenos S_BATCH in R_BATCH na CPU, če so na GPU
    S_BATCH_cpu = cp.asnumpy(S_BATCH) if isinstance(S_BATCH, cp.ndarray) else S_BATCH
    R_BATCH_cpu = cp.asnumpy(R_BATCH) if isinstance(R_BATCH, cp.ndarray) else R_BATCH

    # Filtriramo S_BATCH
    S_BATCH_filtered = S_BATCH_cpu[mask_bool]

    print(f"Izbranih {S_BATCH_filtered.shape[0]} matrik iz {S_BATCH_cpu.shape[0]} glede na masko.")

    # Izgradnja novega drevesa samo iz filtriranih matrik
    print("Izgradnja novega dvojiškega drevesa iz filtriranih matrik ...")
    S_BATCH_BINARY_TREE_cpu = createBinaryTreeArray_CPU(S_BATCH_filtered)

    # Shranimo filtrirano S_BATCH, novo drevo in R_BATCH v .npz
    output_file = BASE_DIR / "filtered_data.npz"
    np.savez(
        output_file,
        S_BATCH=S_BATCH_filtered,
        S_BATCH_BINARY_TREE=S_BATCH_BINARY_TREE_cpu,
        R_BATCH=R_BATCH_cpu
    )

    print(f"✔ Shranjenih {S_BATCH_filtered.shape[0]} matrik in novo drevo v {output_file}")


def load_R_batch_from_txt(filename, use_gpu=True):
    """
    Robustno naloži R matrike iz txt datoteke, ki vsebuje Python seznam 9x9 matrik.
    Deluje tudi, če so prelomi vrstic in zamiki.

    Args:
        txt_datoteka (str ali Path): Pot do txt datoteke.
        uporabi_gpu (bool): Če True, vrne CuPy array, drugače NumPy array.

    Returns:
        cp.ndarray ali np.ndarray: Batch matrik oblike (N, 9, 9), dtype uint8.
    """
    filename = Path(filename)

    # Preberi celotno vsebino
    with open(filename, "r") as f:
        fileContent = f.read()

    # Odstrani začetne / končne presledke
    fileContent = fileContent.strip()

    # Ker je txt podobno Python seznamu, ga lahko pretvorimo v JSON (robustno)
    # Zamenjaj 'single quotes' s 'double quotes', da je veljaven JSON
    fileContent_json = fileContent.replace("'", '"')

    # Naloži kot JSON
    import json
    try:
        R_list = json.loads(fileContent_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Napaka pri pretvorbi txt v seznam: {e}")

    # Pretvori v NumPy array
    R_batch_np = np.array(R_list, dtype=np.uint8)
    if use_gpu:
        R_batch = cp.asarray(R_batch_np)
    else:
        R_batch = R_batch_np

    print(f"✔ R batch naložen iz '{filename}': oblika = {R_batch.shape}")
    return R_batch


def save_trios_to_txt(eliminatedProducts, txt_path):
    """
    Shrani seznam trio matrik v txt datoteko v formatu:
    [ [R1 matrika],
      [S matrika],
      [R2 matrika] ],
    
    Parametri:
    - eliminatedProducts: seznam array-ov oblike (3, 9, 9)
    - txt_path: Path ali str
    """
    txt_path = Path(txt_path)
    
    with open(txt_path, "w") as f:
        for trio in eliminatedProducts:  # vsak trio je (3,9,9)
            trio_np = cp.asnumpy(trio) if isinstance(trio, cp.ndarray) else trio
            f.write("[\n")
            for mat_idx, mat in enumerate(trio_np):  # mat shape = (9,9)
                f.write("[\n")
                for i, row in enumerate(mat):
                    row_str = ", ".join(str(int(x)) for x in row)
                    if i < 8:
                        f.write(f"[ {row_str} ],\n")
                    else:
                        f.write(f"[ {row_str} ]\n")
                if mat_idx < 2:
                    f.write("],\n")  # loči matrike v trio
                else:
                    f.write("]\n")  # zadnja matrika v triu
            f.write("],\n\n")  # ločimo posamezen trio

    print(f"✔ Vse trio matrike zapisane v {txt_path}")



S_BATCH, S_BATCH_BINARY_TREE = load_S_batch(BASE_DIR / "InputFiles" / "S_BATCH_BINARY_TREE.npz")
S_BATCH_MASK = cp.ones((S_BATCH.shape[0]), dtype=cp.uint8)

R_BATCH = load_R_batch_from_txt(BASE_DIR / "InputFiles" / "r_batch.txt", use_gpu=True)



eliminatedProducts = [] # seznam R1 in R2, katerih produkt je bil v S_BATCH, polni se med izvajanjem programa


# Parametri za performance tuning
blockDim = 256
r1BatchWindowLength = 1000
r2BatchWindowLength = 1000

S_BATCH_LENGTH = S_BATCH.shape[0]
start_time = time.time()

for current_S_index in range(S_BATCH_LENGTH):
    # Izpis napredka
    elapsed_time = time.time() - start_time
    avg_time_per_S = elapsed_time / (current_S_index + 1)
    remaining_S = S_BATCH_LENGTH - (current_S_index + 1)
    eta_seconds = remaining_S * avg_time_per_S
    percent_done = (current_S_index + 1) / S_BATCH_LENGTH * 100

    if current_S_index % 100 == 0:  # izpis na vsakih 100 S
        print(
            f"[S index: {current_S_index+1}/{S_BATCH_LENGTH}] "
            f"{percent_done:.4f}% done | "
            f"Elapsed: {int(elapsed_time//3600):02d}:"
            f"{int((elapsed_time%3600)//60):02d}:"
            f"{int(elapsed_time%60):02d} | "
            f"ETA: {int(eta_seconds//3600):02d}:"
            f"{int((eta_seconds%3600)//60):02d}:"
            f"{int(eta_seconds%60):02d}"
        )


    # obdelava na GPU
    if(S_BATCH_MASK[current_S_index] == 0):
        continue


    CURRENT_S = S_BATCH[current_S_index]
    EvaluateS_GPU(CURRENT_S, R_BATCH, S_BATCH_BINARY_TREE, S_BATCH_MASK, eliminatedProducts, r1BatchWindowLength, r2BatchWindowLength, blockDim)


print("S_BATCH je bil v celoti obdelan.")
filterAndSaveAll(S_BATCH, S_BATCH_MASK, R_BATCH)
save_trios_to_txt(eliminatedProducts, BASE_DIR / "eliminated_products.txt")