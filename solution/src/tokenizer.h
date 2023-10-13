#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct PrefixTree {
    struct PrefixTree* children[256];
    int token_id;
} PrefixTree;

typedef struct Tokenizer{
    PrefixTree* prefix_tree;
    PrefixTree* last_node;
    size_t offset;

    const char* string;
    int unk_token_id;
    int num_tokens;
    int char_idx;
} Tokenizer;

PrefixTree* getTokenizerNode(Tokenizer* tokenizer, PrefixTree* node);

void createNode(PrefixTree** node);
void buildPrefixTree(PrefixTree** root, int num_tokens, char** tokens);
void freePrefixTree(PrefixTree** root);

Tokenizer* createDefaultTokenizer();
Tokenizer* createTokenizer(const char* vocab_path);
void initTokenizer(Tokenizer** tokenizer_p, const char* vocab_path);
void freeTokenizer(Tokenizer** tokenizer);
void setTokenizeString(Tokenizer* tokenizer, const char* string);

int nextToken(Tokenizer* tokenizer);
void tokenize(Tokenizer* tokenizer, const char* string, int* num_tokens, int** tokens);

void saveEmbedTokenizer(Tokenizer* tokenizer, const char* path);
Tokenizer* loadEmbedTokenizer();

#endif