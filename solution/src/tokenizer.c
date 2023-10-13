#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tokenizer.h"
#include "embed_tokenizer.h"


void createNode(PrefixTree** node) {
    *node = malloc(sizeof(PrefixTree));
    for (int i = 0; i < 256; ++i) {
        (*node)->children[i] = NULL;
    }
    (*node)->token_id = -1;
}

void buildPrefixTree(PrefixTree** root, int num_tokens, char** tokens) {
    createNode(root);
    
    for (int token_id = 0; token_id < num_tokens; ++token_id) {
        PrefixTree *node = *root;
        char *token = tokens[token_id];
        
        for (int j = 0; token[j] != 0; ++j) {
            unsigned char byte = token[j];

            if (node->children[byte] == NULL) {
                createNode(&node->children[byte]);
            }
            node = node->children[byte];
        }
        node->token_id = token_id;
    }
}

void freePrefixTree(PrefixTree** root) {
    if (*root == NULL) {
        return;
    }

    for (int i = 0; i < 256; ++i) {
        freePrefixTree(&(*root)->children[i]);
    }

    free(*root);
    *root = NULL;
}

void initTokenizer(Tokenizer** tokenizer_p, const char* vocab_path) {
    *tokenizer_p = malloc(sizeof(Tokenizer));
    Tokenizer* tokenizer = *tokenizer_p;

    tokenizer->string = NULL;
    tokenizer->char_idx = 0;
    tokenizer->unk_token_id = 3;
    tokenizer->offset = 0;

    FILE *file;

    file = fopen(vocab_path, "rb");

    int num_tokens;
    int total_len;

    fread(&num_tokens, sizeof(num_tokens), 1, file);
    fread(&total_len, sizeof(total_len), 1, file);

    char** tokens = malloc(num_tokens * sizeof(char*));
    char* data = malloc(total_len * sizeof(char));

    fread(tokens, sizeof(char*), num_tokens, file);
    fread(data, sizeof(char), total_len, file);

    fclose(file);

    for (int i = 0; i < num_tokens; ++i) {
        tokens[i] = (long long int)tokens[i] + &data[0];
    }

    tokenizer->num_tokens = num_tokens;
    // tokenizer->data = data;
    // tokenizer->tokens = tokens;

    buildPrefixTree(&tokenizer->prefix_tree, num_tokens, tokens);
}

void freeTokenizer(Tokenizer** tokenizer) {
    freePrefixTree(&(*tokenizer)->prefix_tree);
    // free((*tokenizer)->tokens);
    // free((*tokenizer)->data);
    free(*tokenizer);
    *tokenizer = NULL;
}

void setTokenizeString(Tokenizer* tokenizer, const char* string) {
    tokenizer->string = string;
    tokenizer->char_idx = 0;
}

PrefixTree* getTokenizerNode(Tokenizer* tokenizer, PrefixTree* node) {
    return (PrefixTree*)((char*)node + tokenizer->offset);
}

int nextToken(Tokenizer* tokenizer) {
    PrefixTree* node = getTokenizerNode(tokenizer, tokenizer->prefix_tree);
    tokenizer->last_node = node;

    if (tokenizer->string[tokenizer->char_idx] == 0) {
        return -1;
    }

    int last_char_idx = tokenizer->char_idx;
    PrefixTree* last_node = node;

    for (; tokenizer->string[tokenizer->char_idx] != 0; tokenizer->char_idx++) {
        unsigned char byte = tokenizer->string[tokenizer->char_idx];

        if (node->token_id != -1) {
            last_char_idx = tokenizer->char_idx;
            last_node = node;
        }

        if (node->children[byte] == NULL) {
            if (last_node == getTokenizerNode(tokenizer, tokenizer->prefix_tree)) {
                tokenizer->char_idx++;
                return tokenizer->unk_token_id;
            }

            tokenizer->char_idx = last_char_idx;
            return last_node->token_id;
        }

        node = getTokenizerNode(tokenizer, node->children[byte]);
    }
    
    if (node->token_id != -1) {
        return node->token_id;
    }

    return tokenizer->unk_token_id;
}

void tokenize(Tokenizer* tokenizer, const char* string, int* num_tokens, int** tokens) {
    setTokenizeString(tokenizer, string);
    
    *tokens = malloc(sizeof(int) * (strlen(string) + 1));

    *num_tokens = 0;
    while (1) {
        int token_id = nextToken(tokenizer);
    
        if (token_id == -1) {
            return;
        }

        (*tokens)[(*num_tokens)++] = token_id;
    }
}

Tokenizer* createDefaultTokenizer() {
    const char* vocab_path = "resources/tokenizer_vocab.bin";
    return createTokenizer(vocab_path);
}

Tokenizer* createTokenizer(const char* vocab_path) {
    Tokenizer* tokenizer;
    initTokenizer(&tokenizer, vocab_path);
    return tokenizer;
}

#define QUEUE_SIZE 1<<15
PrefixTree* nodes[QUEUE_SIZE];

void saveEmbedTokenizer(Tokenizer* tokenizer, const char* path) {
    int head = 0;
    int tail = 0;
    
    size_t offset = sizeof(Tokenizer);

    Tokenizer tok_copy = *tokenizer;
    tok_copy.string = NULL;
    tok_copy.last_node = NULL;

    tok_copy.prefix_tree = (PrefixTree*)offset;
    nodes[tail++] = tokenizer->prefix_tree;
    offset += sizeof(PrefixTree);

    FILE* file = fopen(path, "w");
    fprintf(file, "#ifndef EMBEDDED_TOKENIZER\n#define EMBEDDED_TOKENIZER\n\nunsigned int TOKENIZER_DATA[] = {\n");

    int j = 1;
    for(int i = 0; i < sizeof(Tokenizer) / 4; i++) {
        fprintf(file, "0x%08X,", ((unsigned int*)&tok_copy)[i]);
        if (j++ % 16 == 0) { fprintf(file, "\n"); }
    }

    while (head != tail) {
        PrefixTree* node = nodes[head++];
        head %= QUEUE_SIZE;

        for (int i = 0; i < 256; ++i) {
            if (node->children[i] != NULL) {
                nodes[tail++] = node->children[i];
                tail %= QUEUE_SIZE;

                node->children[i] = (PrefixTree*)offset;
                offset += sizeof(PrefixTree);
            }
        }

        for(int i = 0; i < sizeof(PrefixTree) / 4; i++) {
            fprintf(file, "0x%08X,", ((unsigned int*)node)[i]);
            if (j++ % 16 == 0) { fprintf(file, "\n"); }
        }
    }

    fprintf(file, "0x%08X\n};\n\n#endif\n", 0u);
    fclose(file);
}

Tokenizer* loadEmbedTokenizer() {
#ifdef EMBEDDED_TOKENIZER
    
    Tokenizer* tokenizer = (Tokenizer*)TOKENIZER_DATA;
    tokenizer->offset = (size_t)TOKENIZER_DATA;

    return tokenizer;

#else

    return NULL;

#endif
}