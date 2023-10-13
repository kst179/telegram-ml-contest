#include "matrix.h"

typedef struct GRU {
    // Weights
    Matrix embeddings;
    Matrix weights_h;
    Matrix bias_h;
    Matrix weights_classifier;
    Matrix bias_classifier;

    // Preallocated matrices for calculations
    Matrix hidden_state;
    Matrix logits;
    Matrix rzn;

    // Sumatrices (views)
    Matrix r;
    Matrix z;
    Matrix n;
    Matrix rz;

    int hidden_dim;
    int embedding_size;
    int num_classes;
} GRU;

GRU* createGRU(const char* path);
Matrix getLastStateGRU(GRU* gru, int* tokens, int num_tokens);
int predictGRU(GRU* gru, int* tokens, int num_tokens);
void freeGRU(GRU** gru);

void saveEmbedGRU(GRU* gru, const char* path);
GRU* loadEmbedGRU();