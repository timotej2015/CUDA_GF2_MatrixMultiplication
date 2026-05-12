__device__ __forceinline__
unsigned char* get_9x9_matrix_element_from_batch(
    // Vrne pointer na batch[startPosition][i1][i2] iz batcha z 9x9 matrikami
    // Za S[i1][i2] poslji startPosition = 0

    unsigned char* BATCH_OF_9x9_MATRICES,
    int startPosition, int i1, int i2
){
    // Izracun offseta za obliko (N, 9, 9)
    // Stride dim 2 (zadnja) = 1
    // Stride dim 1 = 9 = 9
    // Stride dim 0 = 9 * 9 = 81

    int offset =
        startPosition * 81 +
        i1 * 9  +
        i2;

    return &BATCH_OF_9x9_MATRICES[offset];
}

__device__ __forceinline__
void calculate_matmul(
    // Funkcija izracuna matmul stirih matrik 9x9 (((A @ B) @ C) @ D)
    unsigned char A[9][9],
    unsigned char B[9][9],
    unsigned char C[9][9],
    unsigned char D[9][9],
    unsigned char matmulResult[9][9]    // Izhodna matrika, vse mora biti inicializirano na 0
){
    // Prvi produkt: A @ B
    unsigned char temp1[9][9] = {0};
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            for(int k=0; k<9; k++){
                temp1[i][j] += (A[i][k]) * (B[k][j]);
                temp1[i][j] %= 2;
            }
        }
    }

    // Drugi produkt: (A @ B) @ C
    unsigned char temp2[9][9] = {0};
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            for(int k=0; k<9; k++){
                temp2[i][j] += (temp1[i][k]) * (C[k][j]);
                temp2[i][j] %= 2;
            }
        }
    }

    // Tretji produkt: ((A @ B) @ C) @ D
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            for(int k=0; k<9; k++){
                matmulResult[i][j] += (temp2[i][k]) * (D[k][j]);
                matmulResult[i][j] %= 2;
            }
        }
    }
}

__device__ __forceinline__
void evaluate_S_matrix(
    // Metoda ovrednoti kombinacijo matrik, ki jih obdeluje thread in zapise rezultat

    unsigned char* CURRENT_S,
    unsigned char* CURRENT_S_INV,
    unsigned char* R1_MINIBATCH,
    unsigned char* R2_MINIBATCH,
    unsigned char* CURRENT_S_IS_GOOD,
    int thread_id, int Current_R1_StartPosition, int Current_R2_StartPosition
){
    unsigned char S[9][9] = {0};
    unsigned char S_INV[9][9] = {0};
    unsigned char R1[9][9] = {0};
    unsigned char R2[9][9] = {0};

    // Napolni matrike s podatki
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            S[i][j] = *get_9x9_matrix_element_from_batch(CURRENT_S, 0, i, j);
            S_INV[i][j] = *get_9x9_matrix_element_from_batch(CURRENT_S_INV, 0, i, j);
            R1[i][j] = *get_9x9_matrix_element_from_batch(R1_MINIBATCH, Current_R1_StartPosition, i, j);
            R2[i][j] = *get_9x9_matrix_element_from_batch(R2_MINIBATCH, Current_R2_StartPosition, i, j);
        }
    }

    unsigned char levaStranEnacbe[9][9] = {0};  // S^(-1) * R_1 * S * R_2
    calculate_matmul(S_INV, R1, S, R2, levaStranEnacbe);

    unsigned char desnaStranEnacbe[9][9] = {0};  // R_2 * S^(-1) * R_1 * S
    calculate_matmul(R2, S_INV, R1, S, desnaStranEnacbe);



    // Ce je levaStranEnacbe == desnaStranEnacbe, to ni dober S
    bool isEqual = true;
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            if(levaStranEnacbe[i][j] != desnaStranEnacbe[i][j]){
                isEqual = false;
            }
        }
    }

    if(isEqual){
        CURRENT_S_IS_GOOD[0] = 0;   // data race safe: multiple writers, same value (0)
    }
}


extern "C" __global__
void matmul_all_in_minibatch(
    unsigned char* CURRENT_S,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* CURRENT_S_INV,     // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* R1_MINIBATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* R2_MINIBATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* CURRENT_S_IS_GOOD,      // Pazi: To mora ustrezati dtype-u v Pythonu!
    int R1_MINIBATCH_LEN, int R2_MINIBATCH_LEN
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= R1_MINIBATCH_LEN * R2_MINIBATCH_LEN) return; // Ker se bo tudi zadnji blok zapolnil z blockDim nitmi, nekatere bodo odvec


    // Threadi sledijo tej logiki:
    /*
    thread_id = 0
    for each R1 in R1_MINIBATCH:
        for each R2 in R2_MINIBATCH:
            if(levastranenacbe == desnastranenacbe):
                CURRENT_S_IS_GOOD = 0
            threadId++
    */

    int Current_R2_StartPosition = thread_id % R2_MINIBATCH_LEN;
    int Current_R1_StartPosition = (thread_id / R2_MINIBATCH_LEN) % R1_MINIBATCH_LEN;

    evaluate_S_matrix(CURRENT_S, CURRENT_S_INV, R1_MINIBATCH, R2_MINIBATCH, CURRENT_S_IS_GOOD, thread_id, Current_R1_StartPosition, Current_R2_StartPosition);
}