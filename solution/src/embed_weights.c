#include "tokenizer.h"
#include "svc.h"
#include "lang_names.h"
#include "gru.h"

int main() {
    GRU* gru = createGRU("resources/gru_weights.bin");
    SVC* svc = createSVC("resources/svc_weights.bin");
    Tokenizer* tokenizer = createTokenizer("resources/tokenizer_vocab.bin");

    saveEmbedGRU(gru, "src/embed_gru.h");
    saveEmbedSVC(svc, "src/embed_svc.h");
    saveEmbedTokenizer(tokenizer, "src/embed_tokenizer.h");

    freeGRU(&gru);
    freeSVC(&svc);
    freeTokenizer(&tokenizer);
    
    return 0;
}
