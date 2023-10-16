#include <stdio.h>

#include "matrix.h"
#include "svc.h"

#ifdef EMBED_WEIGHTS
#include "embed_svc.h"
#endif

SVC* createDefaultSVC() {
    const char* path = "resources/svc_coefs.bin";
    return createSVC(path);
}

SVC* createSVC(const char* path) {
    FILE* file = fopen(path, "rb");
    
    int num_classes;
    int num_tokens;
    int gru_hid_dim;

    fread(&num_classes, sizeof(num_classes), 1, file);
    fread(&num_tokens, sizeof(num_tokens), 1, file);
    fread(&gru_hid_dim, sizeof(gru_hid_dim), 1, file);

    SVC* svc = malloc(sizeof(SVC));

    svc->num_classes = num_classes;
    svc->num_tokens = num_tokens;
    svc->gru_hid_dim = gru_hid_dim;

    svc->token_weights = createMatrix(num_tokens, num_classes);
    svc->gru_features_weights = createMatrix(num_classes, gru_hid_dim);
    svc->bias = createMatrix(1, num_classes);

    svc->scores = createMatrix(1, num_classes);

    fread(svc->token_weights.data, sizeof(float), matSize(svc->token_weights), file);
    fread(svc->gru_features_weights.data, sizeof(float), matSize(svc->gru_features_weights), file);
    fread(svc->bias.data, sizeof(float), matSize(svc->bias), file);
    fclose(file);

    return svc;
}

int predictSVC(SVC* svc, int* tokens, int num_tokens, Matrix gru_last_state) {
    matFillZeros(svc->scores);

    for (int i = 0; i < num_tokens; i++) {
        int token_id = tokens[i];
        Matrix row = submatrix(svc->token_weights, token_id, token_id+1, 0, svc->num_classes);
        
        matSum(row, svc->scores, svc->scores);
    }

    matInplaceScalarProd(svc->scores, 1.0f / (float)num_tokens);

    matVecProduct(svc->gru_features_weights, gru_last_state, svc->scores);
    matSum(svc->bias, svc->scores, svc->scores);

    return vecArgmax(svc->scores);
}

void freeSVC(SVC** svc) {
    freeMatrix((*svc)->scores);
    freeMatrix((*svc)->token_weights);
    freeMatrix((*svc)->gru_features_weights);
    freeMatrix((*svc)->bias);
    free(*svc);
    (*svc) = NULL;
}

void saveEmbedSVC(SVC* svc, const char* path) {
    FILE* file = fopen(path, "w");

    fprintf(file, "#ifndef EMBEDDED_SVC\n#define EMBEDDED_SVC\n\n unsigned int SVC_DATA[] __attribute__((aligned(32))) = {\n");

    SVC svc_copy = *svc;
    
    size_t offset = sizeof(SVC);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    svc_copy.token_weights.data = (float*)offset; 
    offset += matSizeBytes(svc->token_weights);

    svc_copy.gru_features_weights.data = (float*)offset; 
    offset += matSizeBytes(svc->gru_features_weights);

    svc_copy.bias.data = (float*)offset;
    offset += matSizeBytes(svc->bias);

    svc_copy.scores.data = (float*)offset;
    offset += matSizeBytes(svc->scores);

    offset = sizeof(SVC);
    offset = (offset + 31) / 32 * 32; // 32 bytes aligned

    int j = 1;
    int k = 0;
    for(k = 0; k < sizeof(svc_copy) / 4; k++) {
        fprintf(file, "0x%08X,", ((unsigned int*)&svc_copy)[k]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    for (; k < offset / 4; k++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(svc->token_weights); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)svc->token_weights.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(svc->gru_features_weights); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)svc->gru_features_weights.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(svc->bias); i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)svc->bias.data)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    for(int i = 0; i < matSize(svc->scores); i++) {
        fprintf(file, "0x%08X,", 0u);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }
    
    fprintf(file, "0x%08X\n};\n\n#endif\n", 0u);
    fclose(file);
}

SVC* loadEmbedSVC() {
#ifdef EMBEDDED_SVC

    SVC* svc = (SVC*)SVC_DATA;
    svc->token_weights.data = (float*)((unsigned char*)SVC_DATA + (size_t)svc->token_weights.data);
    svc->gru_features_weights.data = (float*)((unsigned char*)SVC_DATA + (size_t)svc->gru_features_weights.data);
    svc->bias.data = (float*)((unsigned char*)SVC_DATA + (size_t)svc->bias.data);
    svc->scores.data = (float*)((unsigned char*)SVC_DATA + (size_t)svc->scores.data);

    return svc;

#else

    return NULL;

#endif
}