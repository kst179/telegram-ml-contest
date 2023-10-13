#include "tglang.h"

#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "gru.h"
#include "svc.h"

enum TglangLanguage tglang_detect_programming_language(const char *text) {
    static Tokenizer* tokenizer = NULL;
    static GRU* gru = NULL;
    static SVC* svc = NULL;

    if (tokenizer == NULL) { tokenizer = loadEmbedTokenizer(); }
    if (gru == NULL) { gru = loadEmbedGRU(); }
    if (svc == NULL) { svc = loadEmbedSVC(); }

    int num_tokens;
    int* tokens;

    tokenize(tokenizer, text, &num_tokens, &tokens);

    Matrix gru_last_state = getLastStateGRU(gru, tokens, num_tokens);
    int prediction = predictSVC(svc, tokens, num_tokens, gru_last_state);
    
    free(tokens);

    return (enum TglangLanguage)prediction;
}
