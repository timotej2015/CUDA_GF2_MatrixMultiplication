import cupy as cp
from pathlib import Path
import math
import time
import numpy as np
import itertools

cuda_code = Path("kernels/gf2_matmul.cu").read_text()

module = cp.RawModule(code=cuda_code)
kernel = module.get_function("matmul_all_batches")


# Osnovne 3x3 matrike
I3 = cp.array([[1,0,0],
               [0,1,0],
               [0,0,1]], dtype=cp.uint8)

ZERO3 = cp.zeros((3,3), dtype=cp.uint8)

# Konstantna matrika C
C = cp.array([[0,0,1],
              [1,0,1],
              [0,1,0]], dtype=cp.uint8)

# Konstantna matrika U
U = cp.array([[1,1,1],
              [1,1,0],
              [0,1,0]], dtype=cp.uint8)




class XBatchGenerator4D:
    def __init__(self, batch_size=1000, resume_index=0):
        # Konstantne 3x3 matrike
        self.U2 = (U @ U) % 2

        self.batch_size = batch_size
        self.total_bits = 36  # a1..a4

        self.resume_index = resume_index
        self.current_index = resume_index

        self.total_combinations = 2 ** self.total_bits
        self.total_batches = math.ceil(self.total_combinations / batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.total_combinations:
            raise StopIteration

        # določi indeks konca batcha in velikost batcha (zadnji batch morda ne bo poln)
        end_index = min(self.current_index + self.batch_size, self.total_combinations)
        actual_batch_size = end_index - self.current_index

        # vse kombinacije v tem batchu kot uint64
        idx = cp.arange(self.current_index, end_index, dtype=cp.uint64)[:, None]

        # bitni masiv: (batch_size, 36)
        bits = ((idx >> cp.arange(self.total_bits, dtype=cp.uint64)) & 1).astype(cp.uint8)

        # razdelimo na a1..a4
        a1 = bits[:, 0:9].reshape(actual_batch_size, 3,3)
        a2 = bits[:, 9:18].reshape(actual_batch_size, 3,3)
        a3 = bits[:, 18:27].reshape(actual_batch_size, 3,3)
        a4 = bits[:, 27:36].reshape(actual_batch_size, 3,3)

        # batch X: (batch_size, 3,3,3,3)
        X_batch = cp.empty((actual_batch_size, 3,3,3,3), dtype=cp.uint8)

        X_batch[:,0,0] = I3
        X_batch[:,0,1] = U
        X_batch[:,0,2] = self.U2
        X_batch[:,1,0] = U
        X_batch[:,1,1] = a1
        X_batch[:,1,2] = a2
        X_batch[:,2,0] = self.U2
        X_batch[:,2,1] = a3
        X_batch[:,2,2] = a4

        self.current_index += actual_batch_size

        return X_batch
    



def FillLambdaBatches():
    def build_lambda(p1, p2, p3):
        Lambda1 = cp.empty((1,3,3,3), dtype=cp.uint8)  # 1 vrstica, 3 stolpci, vsak element 3x3
        Lambda2 = cp.empty((3,1,3,3), dtype=cp.uint8)  # 3 vrstice, 1 stolpec, vsak element 3x3
        
        # Lambda1
        Lambda1[0,0] = p1
        Lambda1[0,1] = p2
        Lambda1[0,2] = p3
        
        # Lambda2 → rotacija vrstic v stolpce
        Lambda2[0,0] = p1
        Lambda2[1,0] = p2
        Lambda2[2,0] = p3
        
        return Lambda1, Lambda2
    

    
    # Ustvarjanje vseh p
    C2 = (C @ C) % 2
    
    k_combinations = cp.array(list(itertools.product([0,1], repeat=3)), dtype=cp.uint8) # seznam vseh permutacij treh
    
    p_list = cp.empty((8,3,3), dtype=cp.uint8)
    
    for i, (k1,k2,k3) in enumerate(k_combinations):
        p_list[i] = (k1*I3 + k2*C + k3*C2) % 2

    
    #Ustvarjanje vseh lambd
    Lambda1_list = []
    Lambda2_list = []

    # 1️⃣ Prva izvedba: vse kombinacije p1, p2
    for i1, i2 in itertools.product(range(len(p_list)), repeat=2):   # eleganten zapis nested for loopa z i1 in i2
        L1, L2 = build_lambda(I3, p_list[i1], p_list[i2])
        Lambda1_list.append(L1)
        Lambda2_list.append(L2)

    # 2️⃣ Druga izvedba: vse p3
    for i3 in range(len(p_list)):
        L1, L2 = build_lambda(ZERO3, I3, p_list[i3])
        Lambda1_list.append(L1)
        Lambda2_list.append(L2)

    # 3️⃣ Tretja izvedba: konstanta
    L1, L2 = build_lambda(ZERO3, ZERO3, I3)
    Lambda1_list.append(L1)
    Lambda2_list.append(L2)


    # Stack v GPU tensor
    Lambda1_set = cp.stack(Lambda1_list, axis=0)  # shape = (N,1,3,3)
    Lambda2_set = cp.stack(Lambda2_list, axis=0)  # shape = (N,3,1,3)

    return Lambda1_set, Lambda2_set




def FindAllGoodXMatricesInBatch(la1Batch, xBatch, la2Batch, blockdim=256): 
    # Metoda vrne seznam vseh X iz xBatch, katerih produkti množenja z množicami iz la1Batch in la2Batch nikoli niso 0

    LA1_BATCH_LEN = la1Batch.shape[0]
    X_BATCH_LEN   = xBatch.shape[0]
    LA2_BATCH_LEN = la2Batch.shape[0]

    #gpu_matmul_output = cp.zeros((X_BATCH_LEN, LA1_BATCH_LEN, LA2_BATCH_LEN, 3, 3), dtype=cp.uint8) # če želiš, da se celotni rezultati množenja shranijo
    gpu_matmul_output = cp.ones((X_BATCH_LEN), dtype=cp.uint8)

    NUMBER_OF_THREADS = LA1_BATCH_LEN * X_BATCH_LEN * LA2_BATCH_LEN
    NUMBER_OF_BLOCKS = (NUMBER_OF_THREADS + blockdim - 1) // blockdim   # isto kot ceil(NUMBER_OF_THREADS / blockdim)

    """
    Kernel za vse možne kombinacije matrik iz batchev izračuna la1@X@la2 v GF(2).
    Če nek X v xBatch pri kakšnem matmulu da ničelen rezultat, popravi istoležen element v gpu_matmul_output iz 1 na 0.
    """
    kernel((NUMBER_OF_BLOCKS,), (blockdim,), (la1Batch, xBatch, la2Batch, gpu_matmul_output, LA1_BATCH_LEN, X_BATCH_LEN, LA2_BATCH_LEN))
    cp.cuda.runtime.deviceSynchronize()

    # ustvari array dobrih X glede na rezultate kernela
    mask = gpu_matmul_output.astype(cp.bool_)
    good_x_matrices = xBatch[mask]


    return good_x_matrices



def AppendToWolframFile(good_X_matrices, filename="good_x_matrices.wl"):
    if good_X_matrices is None or len(good_X_matrices) == 0:
        return
    
    print(good_X_matrices)

    if isinstance(good_X_matrices, cp.ndarray):
        good_X_matrices = cp.asnumpy(good_X_matrices)

    file_path = Path(filename)
    is_new_file = not file_path.exists()

    with open(filename, "a") as f:
        if is_new_file:
            f.write("GoodXMatrices = {\n")
        else:
            f.write("\n")  # ločimo nove matrike od prejšnjih

        for idx, mat in enumerate(good_X_matrices):
            f.write("    FromBlockMat[")
            f.write(np.array2string(mat, separator=", ", max_line_width=np.inf).replace("\n", ""))
            f.write("]")
            f.write(",\n")


def SmallMatrixMatmul_CPU(A, B, C):
    # Matmul treh 3x3 matrik: A @ B @ C
    result = cp.zeros((3,3), dtype=A.dtype)
    for i in range(3):
        for j in range(3):
            s = 0
            for k in range(3):
                # seštevanje po vmesnem indeksu k
                for l in range(3):
                    s += A[i,k] * B[k,l] * C[l,j]
                    s %= 2
            result[i,j] = s
    return result

def BigMatrixMatmul_CPU(A, B, C):
    # Matmul A (1x3x3x3) @ B (3x3x3x3) @ C (3x1x3x3) -> 3x3
    result = cp.zeros((3,3), dtype=A.dtype)
    for i in range(3):      # dimenzija, ki se množi
        for j in range(3):  # notranji element B
            result += SmallMatrixMatmul_CPU(A[0,i], B[i,j], C[j,0])
            result %= 2
    return result





X_BATCH_LENGTH = 1000
la1Batch, la2Batch = FillLambdaBatches()
gen = XBatchGenerator4D(batch_size=X_BATCH_LENGTH, resume_index=0)


batch_number = 1
# --- Čas začetka celotnega programa ---
program_start_time = time.time()

try:
    while True:
        X_batch = next(gen)  # (B,3,3,3,3)

        # --- Povprečni čas na batch od začetka programa ---
        total_elapsed_time = time.time() - program_start_time
        avg_time_per_batch = total_elapsed_time / batch_number

        # --- Napredek ---
        percent_done = (batch_number / gen.total_batches) * 100
        remaining_batches = gen.total_batches - batch_number

        # --- ETA ---
        eta_seconds = avg_time_per_batch * remaining_batches    

        # --- Poišči želene X ---
        goodXfromBatch = FindAllGoodXMatricesInBatch(la1Batch, X_batch, la2Batch)
        AppendToWolframFile(goodXfromBatch)

        # --- Izpis napredka
        if batch_number % 500 == 0:
            print(
                f"[Batch {batch_number}/{gen.total_batches}] "
                f"{percent_done:.6f}% done | "
                f"Elapsed: "
                f"{int(total_elapsed_time//3600):02d}:"
                f"{int((total_elapsed_time%3600)//60):02d}:"
                f"{int(total_elapsed_time%60):02d} | "
                f"ETA: "
                f"{int(eta_seconds//3600):02d}:"
                f"{int((eta_seconds%3600)//60):02d}:"
                f"{int(eta_seconds%60):02d}"
            )

        batch_number += 1

except StopIteration:
    total_time = time.time() - program_start_time
    print("Vsi batch-i so bili obdelani.")