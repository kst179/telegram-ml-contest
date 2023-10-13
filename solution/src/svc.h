#include <stdio.h>
#include <math.h>

#include "matrix.h"

typedef struct SVC {
    Matrix token_weights;           // num_tokens x num_classes
    Matrix gru_features_weights;    // num_classes x gru_hid_dim

    Matrix scores;                  // num_classes

    int num_classes;
    int num_tokens;
    int gru_hid_dim;
} SVC;

SVC* createDefaultSVC();
SVC* createSVC(const char* path);
int predictSVC(SVC* classifier, int* tokens, int num_tokens, Matrix gru_last_state);
void freeSVC(SVC** classifier);

void saveEmbedSVC(SVC* svc, const char* path);
SVC* loadEmbedSVC();