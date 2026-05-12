__device__ __forceinline__
unsigned char* get_x_batch_element(
    // Vrne pointer na X_BATCH[startPosition][i1][i2][i3][i4]

    unsigned char* X_BATCH,
    int startPosition, int i1, int i2, int i3, int i4
){
    // Izracun offseta za obliko (N, 3, 3, 3, 3)
    // Stride dim 4 (zadnja) = 1
    // Stride dim 3 = 3
    // Stride dim 2 = 3 * 3 = 9
    // Stride dim 1 = 3 * 3 * 3 = 27
    // Stride dim 0 = 3 * 3 * 3 * 3 = 81

    int offset =
        startPosition * 81 +
        i1 * 27  +
        i2 * 9  +
        i3 * 3  +
        i4;

    return &X_BATCH[offset];
}

__device__ __forceinline__
unsigned char* get_la1_batch_element(
    // Vrne pointer na LA1_BATCH[startPosition][i1][i2][i3][i4]

    unsigned char* LA2_BATCH,
    int startPosition, int i1, int i2, int i3, int i4
){
    // Izracun offseta za obliko (N, 1, 3, 3, 3)
    // Stride dim 4 (zadnja) = 1
    // Stride dim 3 = 3
    // Stride dim 2 = 3 * 3 = 9
    // Stride dim 1 = 3 * 3 * 3 = 27
    // Stride dim 0 = 1 * 3 * 3 * 3 = 27

    int offset =
        startPosition * 27 +
        i1 * 27  +
        i2 * 9  +
        i3 * 3  +
        i4;

    return &LA2_BATCH[offset];
}

__device__ __forceinline__
unsigned char* get_la2_batch_element(
    // Vrne pointer na LA2_BATCH[startPosition][i1][i2][i3][i4]

    unsigned char* LA2_BATCH,
    int startPosition, int i1, int i2, int i3, int i4
){
    // Izracun offseta za obliko (N, 3, 1, 3, 3)
    // Stride dim 4 (zadnja) = 1
    // Stride dim 3 = 3
    // Stride dim 2 = 3 * 3 = 9
    // Stride dim 1 = 1 * 3 * 3 = 9
    // Stride dim 0 = 3 * 1 * 3 * 3 = 27

    int offset =
        startPosition * 27 +
        i1 * 9  +
        i2 * 9  +
        i3 * 3  +
        i4;

    return &LA2_BATCH[offset];
}


__device__ __forceinline__
void SmallMatrix_Matmul(
    // Matmul treh 3x3 matrik: A @ B @ C
    unsigned char* A[3][3],
    unsigned char* B[3][3],
    unsigned char* C[3][3],
    unsigned char bigMatrixTmpResult[3][3]
){
    unsigned char result[3][3] = {0};

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            int s=0;
            for(int k=0; k<3; k++){
                for(int l=0; l<3; l++){
                    unsigned char a = *A[i][k];
                    unsigned char b = *B[k][l];
                    unsigned char c = *C[l][j];

                    s += a*b*c;
                    s %= 2;
                }
            }
            result[i][j] = s;
        }
    }

    // ta logika je v CPU implementaciji v BigMatrix_Matmul
    /* result[0,0] += SmallMatrixMatmul(A[0,i], B[i,j], C[j,0])
    result[0,0] %= 2 */

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            bigMatrixTmpResult[i][j] += result[i][j];
            bigMatrixTmpResult[i][j] %= 2;
        }
    }
}

__device__ __forceinline__
void BigMatrix_Matmul(
    // Matmul A (1x3x3x3) @ B (3x3x3x3) @ C (3x1x3x3) -> 1x1x3x3

    unsigned char* LA1_BATCH,
    unsigned char* X_BATCH,
    unsigned char* LA2_BATCH,
    unsigned char* OUTPUT,
    int thread_id, int la1_StartPosition, int x_StartPosition, int la2_StartPosition
){
    unsigned char* A[3][3] = {nullptr}; //LA1[0,i]
    unsigned char* B[3][3] = {nullptr}; //B[i,j]
    unsigned char* C[3][3] = {nullptr}; //C[j,0]

    unsigned char result[3][3] = {0};

    for(int i = 0; i<3; i++){
        for(int j = 0; j<3; j++){
            // sestavi 3x3 matrike A, B, C in jih poslji v SmallMatrixMatmul()
            for(int k=0; k<3; k++){
                for(int l=0; l<3; l++){
                    A[k][l] = get_la1_batch_element(LA1_BATCH, la1_StartPosition, 0, i, k, l);
                    B[k][l] = get_x_batch_element(X_BATCH, x_StartPosition, i, j, k, l);
                    C[k][l] = get_la2_batch_element(LA2_BATCH, la2_StartPosition, j, 0, k, l);
                }
            }

            SmallMatrix_Matmul(A, B, C, result);
        }
    }

    //ce hoces shranit cele 3x3 matrike, ki so rezultat matmula
    /*
    // shrani rezultat v OUTPUT, pomagaj si z threadid
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            int outputWritePosition = thread_id * 9 + i * 3 + j;
            OUTPUT[outputWritePosition] = result[i][j];
        }
    }
    */

    //ce je reultat matmula matrika samih nicel, spremeni OUTPUT na mestu, ki predstavlja X na 0 (privzeto so v pythonu vsi nastavljeni na 1)
    bool isAGoodMatrix = false;
    for(int i = 0; i<3; i++){
        if(isAGoodMatrix){
            break;
        }
        for(int j=0; j<3; j++){
            if(result[i][j] == 1){
                isAGoodMatrix = true;
                break;
            }
        }
    }

    if(!isAGoodMatrix){
        OUTPUT[x_StartPosition] = 0; // data race safe: multiple writers, same value (0)
    }
}

extern "C" __global__
void matmul_all_batches(
    unsigned char* LA1_BATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* X_BATCH,     // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* LA2_BATCH,   // Pazi: To mora ustrezati dtype-u v Pythonu!
    unsigned char* OUTPUT,      // Pazi: To mora ustrezati dtype-u v Pythonu!
    int LA1_BATCH_LEN, int X_BATCH_LEN, int LA2_BATCH_LEN
){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= LA1_BATCH_LEN * X_BATCH_LEN * LA2_BATCH_LEN) return; // Ker se bo tudi zadnji blok zapolnil s blockDim nitmi, ki so odvec

    // Threadi sledijo tej logiki:
    /*
    thread_id = 0
    for each x in X_BATCH:
        for each la1 in LA1_BATCH:
            for each la2 in LA2_BATCH:
                // OUTPUT[x][la1][la2] = rezultat     to je ce hoces shranit rezultate mnozenja
                if(rezultat == matrika samih nicel):
                    OUTPUT[x] = 1
                threadId++
    */

    int Current_LA2_StartPosition = thread_id % LA2_BATCH_LEN;
    int Current_LA1_StartPosition = (thread_id / LA2_BATCH_LEN) % LA1_BATCH_LEN;
    int Current_X_StartPosition = thread_id / (LA2_BATCH_LEN * LA1_BATCH_LEN);


    BigMatrix_Matmul(LA1_BATCH, X_BATCH, LA2_BATCH, OUTPUT, thread_id, Current_LA1_StartPosition, Current_X_StartPosition, Current_LA2_StartPosition);
}