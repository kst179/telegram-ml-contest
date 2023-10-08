#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct PrefixTree {
    struct PrefixTree *children[256];
    int token_id;
    int num_children;
} PrefixTree;

void createNode(PrefixTree** node) {
    *node = malloc(sizeof(PrefixTree));
    for (int i = 0; i < 256; ++i) {
        (*node)->children[i] = NULL;
    }
    (*node)->num_children = 0;
    (*node)->token_id = -1;
}

void buildPrefixTree(PrefixTree** root, int num_tokens, char** tokens) {
    createNode(root);
    
    for (int token_id = 0; token_id < num_tokens; ++token_id) {
        PrefixTree *node = *root;
        char *token = tokens[token_id];
        
        for (int j = 0; token[j] != 0; ++j) {
            char byte = token[j];

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

typedef struct Tokenizer{
    PrefixTree* prefix_tree;
    char** tokens;
    char* data;
    char* string;
    int num_tokens;
    int char_idx;
} Tokenizer;

void initTokenizer(Tokenizer **tokenizer_p, char* vocab_path) {
    *tokenizer_p = malloc(sizeof(Tokenizer));
    Tokenizer* tokenizer = *tokenizer_p;

    tokenizer->string = NULL;
    tokenizer->char_idx = 0;

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
    tokenizer->data = data;
    tokenizer->tokens = tokens;

    buildPrefixTree(&tokenizer->prefix_tree, num_tokens, tokens);
}

void freeTokenizer(Tokenizer **tokenizer) {
    freePrefixTree(&(*tokenizer)->prefix_tree);
    free((*tokenizer)->tokens);
    free((*tokenizer)->data);
    free(*tokenizer);
    *tokenizer = NULL;
}

void setTokenizeString(Tokenizer* tokenizer, char* string) {
    tokenizer->string = string;
    tokenizer->char_idx = 0;
}

int nextToken(Tokenizer* tokenizer) {
    PrefixTree* node = tokenizer->prefix_tree;

    if (tokenizer->string[tokenizer->char_idx]) {
        return -1;
    }

    for (; tokenizer->string[tokenizer->char_idx] != 0; tokenizer->char_idx++) {
        char byte = tokenizer->string[tokenizer->char_idx];

        if (node->children[byte] == NULL) {
            return node->token_id;
        }

        node = node->children[byte];
    }

    return node->token_id;
}


int main() {
    char* vocab_path = "tokenizer_vocab.bin";
    char* string = "print('Hello, World!')";

    Tokenizer* tokenizer;

    initTokenizer(&tokenizer, vocab_path);

    setTokenizeString(tokenizer, string);

    int token_id = -1;
    
    while (1) {
        token_id = nextToken(tokenizer);
        printf("%d ", token_id);
    }

    printf("\n");

    freeTokenizer(&tokenizer);

    return 0;
}
