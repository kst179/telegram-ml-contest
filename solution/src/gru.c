#include <stdio.h>

#include "matrix.h"
#include "gru.h"
#include "embed_gru.h"


GRU* createGRU(const char* path) {
    int hidden_dim;
    int embedding_size;
    int num_classes;

    FILE* file = fopen(path, "rb");
    
    fread(&hidden_dim, sizeof(hidden_dim), 1, file);
    fread(&embedding_size, sizeof(embedding_size), 1, file);
    fread(&num_classes, sizeof(embedding_size), 1, file);

    GRU* gru = malloc(sizeof(GRU));
    gru->hidden_dim = hidden_dim;
    gru->embedding_size = embedding_size;
    gru->num_classes = num_classes;

    gru->embeddings = createMatrix(embedding_size, hidden_dim * 3);
    gru->weights_h = createMatrix(hidden_dim * 3, hidden_dim);
    gru->bias_h = createMatrix(1, hidden_dim * 3);
    gru->weights_classifier = createMatrix(num_classes, hidden_dim);
    gru->bias_classifier = createMatrix(1, num_classes);

    fread(gru->embeddings.data, sizeof(float), embedding_size * hidden_dim * 3, file);
    fread(gru->weights_h.data, sizeof(float), hidden_dim * hidden_dim * 3, file);
    fread(gru->bias_h.data, sizeof(float), hidden_dim * 3, file);
    fread(gru->weights_classifier.data, sizeof(float), num_classes * hidden_dim, file);
    fread(gru->bias_classifier.data, sizeof(float), num_classes, file);

    fclose(file);

    gru->hidden_state = createMatrix(1, hidden_dim);
    gru->logits = createMatrix(1, num_classes);
    gru->rzn = createMatrix(1, 3 * hidden_dim);

    gru->r = submatrix(gru->rzn, 0, 1, 0, hidden_dim);
    gru->z = submatrix(gru->rzn, 0, 1, hidden_dim, hidden_dim * 2);
    gru->n = submatrix(gru->rzn, 0, 1, hidden_dim * 2, hidden_dim * 3);
    gru->rz = submatrix(gru->rzn, 0, 1, 0, hidden_dim * 2);

    return gru;
}

Matrix getLastStateGRU(GRU* gru, int* tokens, int num_tokens) {
    matFillZeros(gru->hidden_state);

    for (int i = 0; i < num_tokens; i++) {
        int token_id = tokens[i];
        Matrix rz_i = submatrix(gru->embeddings, token_id, token_id + 1, 0, gru->hidden_dim * 2);
        Matrix n_i = submatrix(gru->embeddings, token_id, token_id + 1, gru->hidden_dim * 2, gru->hidden_dim * 3);

        // rzn = bias_h + weights_h @ hidden_state
        matCopy(gru->bias_h, gru->rzn);
        matVecProduct(gru->weights_h, gru->hidden_state, gru->rzn);

        // rz = sigmoid(rz_i + rz)
        matSum(rz_i, gru->rz, gru->rz);
        matInplaceSigmoid(gru->rz);

        // n = n_i + r * n
        matHProduct(gru->r, gru->n, gru->n);
        matSum(n_i, gru->n, gru->n);
        matInplaceTanh(gru->n);
        matSlerp(gru->n, gru->hidden_state, gru->z, gru->hidden_state);
    }

    return gru->hidden_state;
}

int predictGRU(GRU* gru, int* tokens, int num_tokens) {
    Matrix last_state = getLastStateGRU(gru, tokens, num_tokens);

    matCopy(gru->bias_classifier, gru->logits);
    matVecProduct(gru->weights_classifier, last_state, gru->logits);

    return vecArgmax(gru->logits);
}

void freeGRU(GRU** gru) {
    freeMatrix((*gru)->embeddings);
    freeMatrix((*gru)->weights_h);
    freeMatrix((*gru)->bias_h);
    freeMatrix((*gru)->weights_classifier);
    freeMatrix((*gru)->bias_classifier);

    freeMatrix((*gru)->hidden_state);
    freeMatrix((*gru)->logits);
    freeMatrix((*gru)->rzn);
    
    free(*gru);
    *gru = NULL;
}

void saveEmbedGRU(GRU* gru, const char* path) {
    FILE* file = fopen(path, "w");

    GRU gru_copy = *gru;
    
    size_t offset = sizeof(GRU);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    gru_copy.embeddings.data = (float*)offset; 
    offset += matSizeBytes(gru->embeddings);
    
    gru_copy.weights_h.data = (float*)offset; 
    offset += matSizeBytes(gru->weights_h);
    
    gru_copy.bias_h.data = (float*)offset; 
    offset += matSizeBytes(gru->bias_h);
  
    gru_copy.weights_classifier.data = (float*)offset;
    offset += matSizeBytes(gru->weights_classifier);

    gru_copy.bias_classifier.data = (float*)offset;
    offset += matSizeBytes(gru->bias_classifier);
    
    gru_copy.hidden_state.data = (float*)offset;
    offset += matSizeBytes(gru->hidden_state);

    gru_copy.logits.data = (float*)offset;
    offset += matSizeBytes(gru->logits);
    
    gru_copy.rzn.data = (float*)offset;
    offset += matSizeBytes(gru->rzn);
    
    gru_copy.r = submatrix(gru_copy.rzn, 0, 1, 0, gru->hidden_dim);
    gru_copy.z = submatrix(gru_copy.rzn, 0, 1, gru->hidden_dim, gru->hidden_dim * 2);
    gru_copy.n = submatrix(gru_copy.rzn, 0, 1, gru->hidden_dim * 2, gru->hidden_dim * 3);
    gru_copy.rz = submatrix(gru_copy.rzn, 0, 1, 0, gru->hidden_dim * 2);

    offset = sizeof(GRU);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    fprintf(file, "#ifndef EMBEDDED_GRU\n#define EMBEDDED_GRU\n\n unsigned int GRU_DATA[] __attribute__((aligned(32))) = {\n");

    int j = 1;
    int k = 0;
    for(k = 0; k < sizeof(gru_copy) / 4; k++) {
        fprintf(file, "0x%08X,", ((unsigned int*)&gru_copy)[k]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    for (; k < offset / 4; k++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    
    for(int i = 0; i < matSize(gru->embeddings); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->embeddings.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->weights_h); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->weights_h.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->bias_h); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->bias_h.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->weights_classifier); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->weights_classifier.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->bias_classifier); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)gru->bias_classifier.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->hidden_state); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->logits); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(gru->rzn); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    fprintf(file, "0x%08X\n};\n\n#endif\n", 0u);
    fclose(file);
}

GRU* loadEmbedGRU() {
#ifdef EMBEDDED_GRU

    GRU* gru = (GRU*)&GRU_DATA[0];
    gru->embeddings.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->embeddings.data);
    gru->weights_h.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->weights_h.data);
    gru->bias_h.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->bias_h.data);
    gru->weights_classifier.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->weights_classifier.data);
    gru->bias_classifier.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->bias_classifier.data);
    gru->hidden_state.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->hidden_state.data);
    gru->logits.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->logits.data);
    gru->rzn.data = (float*)((unsigned char*)GRU_DATA + (size_t)gru->rzn.data);

    gru->r = submatrix(gru->rzn, 0, 1, 0, gru->hidden_dim);
    gru->z = submatrix(gru->rzn, 0, 1, gru->hidden_dim, gru->hidden_dim * 2);
    gru->n = submatrix(gru->rzn, 0, 1, gru->hidden_dim * 2, gru->hidden_dim * 3);
    gru->rz = submatrix(gru->rzn, 0, 1, 0, gru->hidden_dim * 2);

    return gru;
    
#else

    return NULL;
    
#endif
}
