// This version is for the "dec.probs" shader tag
// The input tensor has a size [ 51865, 3 ], a very long tensor with just 3 rows.
// Despite the shader only runs on 3 GPU cores, large count of threads helps substantially, this shader is about 50% faster.
#define THREADS 1024

#include "softMax.hlsl"