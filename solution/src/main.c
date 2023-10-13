#include "tokenizer.h"
#include "svc.h"
#include "lang_names.h"
#include "gru.h"

int main() {
    FILE* file = fopen("input.txt", "r");

    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* string = malloc(sizeof(char) * (size + 1));
    fread(string, sizeof(char), size, file);
    string[size-1] = 0;
    fclose(file);

    clock_t start, end;
    double cpu_time_used;
    time_t wall_time_start, wall_time_end;

    start = clock();
    time(&wall_time_start);

    Tokenizer* tokenizer = loadEmbedTokenizer();
    GRU* gru = loadEmbedGRU();
    SVC* svc = loadEmbedSVC();

    int n_repeats = 10;

    int prediction;

    for (int i = 0; i < n_repeats; ++i) {
        int num_tokens;
        int* tokens;

        tokenize(tokenizer, string, &num_tokens, &tokens);

        Matrix gru_last_state = getLastStateGRU(gru, tokens, num_tokens);
        prediction = predictSVC(svc, tokens, num_tokens, gru_last_state);
        free(tokens);
    }

    end = clock();
    time(&wall_time_end);

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    double wall_time_used = difftime(wall_time_end, wall_time_start);
    
    printf("Prediction: %s\n", lang_names[prediction]);
    printf("CPU time: %f ms\n", cpu_time_used  * 1000 / n_repeats);
    printf("Wall time: %f ms\n", wall_time_used * 1000 / n_repeats);

    free(string);

    return 0;
}
