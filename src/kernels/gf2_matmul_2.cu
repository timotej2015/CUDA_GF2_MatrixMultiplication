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
int search_binary_tree(
    // Funkcija poisce matriko v dvojiskem drevesu in vrne njen index oziroma -1, ce je ne najde
    unsigned char lookupMatrix[9][9],
    int* S_BATCH_BINARY_TREE
){
    int row_offset = 3;
    int col_offset = 3;
    int size = 6;
    int current_idx = 0;

    // najprej preveri fiksni del
    const int referenceMatrix[9][9] = {
        { 1, 0, 0, 1, 1, 1, 0, 1, 1 },
        { 0, 1, 0, 1, 1, 0, 0, 0, 1 },
        { 0, 0, 1, 0, 1, 0, 1, 1, 0 },
        { 1, 1, 1,-1,-1,-1,-1,-1,-1 },
        { 1, 1, 0,-1,-1,-1,-1,-1,-1 },
        { 0, 1, 0,-1,-1,-1,-1,-1,-1 },
        { 0, 1, 1,-1,-1,-1,-1,-1,-1 },
        { 0, 0, 1,-1,-1,-1,-1,-1,-1 },
        { 1, 1, 0,-1,-1,-1,-1,-1,-1 }
    };

    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            if(referenceMatrix[i][j] == -1) continue; // ignoriraj variabilni del
            if(lookupMatrix[i][j] != referenceMatrix[i][j]) return -1; // ne ujema se
        }
    }


    // ce je fiksni del v redu, poisci variabilen del v drevesu
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            int bit = lookupMatrix[row_offset + i][col_offset + j];
            if(bit == 0){
                if(S_BATCH_BINARY_TREE[current_idx*3] == -1){
                    return -1;
                }
                current_idx = S_BATCH_BINARY_TREE[current_idx*3];
            }
            else{
                if(S_BATCH_BINARY_TREE[current_idx*3 + 1] == -1){
                    return -1;
                }
                current_idx = S_BATCH_BINARY_TREE[current_idx*3 + 1];
            }
        }
    }
    return S_BATCH_BINARY_TREE[current_idx*3 + 2];
}

__device__ __forceinline__
void calculate_matmul(
    // Funkcija izracuna matmul treh matrik 9x9 ((A @ B) @ C)
    unsigned char A[9][9],
    unsigned char B[9][9],
    unsigned char C[9][9],
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
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            for(int k=0; k<9; k++){
                matmulResult[i][j] += (temp1[i][k]) * (C[k][j]);
                matmulResult[i][j] %= 2;
            }
        }
    }
}

__device__ __forceinline__
void evaluate_S_matrix(
    // Metoda ovrednoti kombinacijo matrik, ki jih obdeluje thread in spremeni masko, ce so pogoji izpolnjeni

    unsigned char* CURRENT_S,
    unsigned char* R1_MINIBATCH,
    unsigned char* R2_MINIBATCH,
    int* S_BATCH_BINARY_TREE, 
    unsigned char* S_BATCH_MASK,
    unsigned char* ELIMINATED_PRODUCTS_MASK,
    int thread_id, int Current_R1_StartPosition, int Current_R2_StartPosition
){
    unsigned char S[9][9] = {0};
    unsigned char R1[9][9] = {0};
    unsigned char R2[9][9] = {0};

    // Napolni matrike s podatki
    for(int i=0; i<9; i++){
        for(int j=0; j<9; j++){
            S[i][j] = *get_9x9_matrix_element_from_batch(CURRENT_S, 0, i, j);
            R1[i][j] = *get_9x9_matrix_element_from_batch(R1_MINIBATCH, Current_R1_StartPosition, i, j);
            R2[i][j] = *get_9x9_matrix_element_from_batch(R2_MINIBATCH, Current_R2_StartPosition, i, j);
        }
    }

    unsigned char rezultatMnozenja[9][9] = {0};  // R_1 * S * R_2
    calculate_matmul(R1, S, R2, rezultatMnozenja);
    int matrixExistsOnIndex = search_binary_tree(rezultatMnozenja, S_BATCH_BINARY_TREE);

    if(matrixExistsOnIndex != -1){
        S_BATCH_MASK[matrixExistsOnIndex] = 0;  // data race safe: multiple writers, same value (0)

        ELIMINATED_PRODUCTS_MASK[thread_id] = 1;
    }
}


extern "C" __global__
void matmul_all_in_minibatch(
    unsigned char* CURRENT_S,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    int* S_BATCH_BINARY_TREE,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* S_BATCH_MASK,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* R1_MINIBATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* R2_MINIBATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* ELIMINATED_PRODUCTS_MASK,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    int R1_MINIBATCH_LEN, int R2_MINIBATCH_LEN
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= R1_MINIBATCH_LEN * R2_MINIBATCH_LEN) return; // Ker se bo tudi zadnji blok zapolnil z blockDim nitmi, nekatere bodo odvec


    // Threadi sledijo tej logiki:
    /*
    thread_id = 0
    for each R1 in R1_MINIBATCH:
        for each R2 in R2_MINIBATCH:
            (izracunaj produkt R1*S*R2 in poglej, ce se nahaja v mnozici S_BATCH)
            threadId++
    */

    int Current_R2_StartPosition = thread_id % R2_MINIBATCH_LEN;
    int Current_R1_StartPosition = (thread_id / R2_MINIBATCH_LEN) % R1_MINIBATCH_LEN;

    evaluate_S_matrix(CURRENT_S, R1_MINIBATCH, R2_MINIBATCH, S_BATCH_BINARY_TREE, S_BATCH_MASK, ELIMINATED_PRODUCTS_MASK,
        thread_id, Current_R1_StartPosition, Current_R2_StartPosition);
}