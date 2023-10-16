#include <stdio.h>
#include <math.h>

#include "matrix.h"

typedef struct SVC {
    Matrix token_weights;           // num_tokens x num_classes
    Matrix gru_features_weights;    // num_classes x gru_hid_dim
    Matrix bias;                    // 1 x num_classes

    Matrix scores;                  // 1 x num_classes

    int num_classes;
    int num_tokens;
    int gru_hid_dim;
} SVC;

SVC* createDefaultSVC();
SVC* createSVC(const char* path);
int predictSVC(SVC* svc, int* tokens, int num_tokens, Matrix gru_last_state);
void freeSVC(SVC** svc);

void saveEmbedSVC(SVC* svc, const char* path);
SVC* loadEmbedSVC();