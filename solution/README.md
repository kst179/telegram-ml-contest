# Telegram ML contest submission

⚠️ **WARNING:** Do not open `embed_*.h` files, I warned you. ⚠️

### What was done:
 * Scraped code dataset from github (~1M files in total), partially using [gesslang tool](https://github.com/yoeo/guesslangtools), partially by my own scripts, and some repo lists were fetched half-manually (for languages that are hard to search for, like fift, func, icon etc). But at first iteration I had about 1K files per language, and only the most frequent ones.
 * Trained a BPE tokenizer on this data (actually only on a small part which was the first iteration of GH scraping).
 * Implemented the same tokenizer in C, with prefix trees, realized that actually BPE **is not a prefix tree** 🤯, but I've checked results of my greedy tokenizer and it is not so bad actually, and uses extra-simple and fast algorithm 🤓.
 * Learned simple LinearSVC (GNU Chervonenkis) on top of the tokens frequencies, it gives about 90% accuracy on the data, I had at that moment. Implemented C inference for sparse linear desicion function.
 * Written extra-mega-fast GRU inference in C with avx2 support 🤓, and checked it with the hidden state=96. Found that it was optimial size of a tiny-little-network. Trained same network in torch and moved the resulting weights to my C implementation. Also tried multithreading, but it seems that there is too much overhead on parallelisation so abandoned this idea (or I just very bad at OMP, and too lazy to write threading myself). GRU showed something like 93% accuracy.
 * I have a GRU, I have a SVC, aaaaah GRU-SVC, I decided to combine two models with linear mix, but linear mix of linear and non-linear models is a linear one trained on the outputs of the second (and we can abandon the classifier layer of GRU as well, cause it is also linear). So I implemented modified SVC which was trained on both token frequences and GRU last hidden state. This mix gives almost 94% of accuracy, cause GRU can forget some important things, and bag-of-words methods can't.
 * Then I made the final iteration of GH scraping, downloaded repos for all the missing languages and rerun the pipeline on them. The final accuracy 94.5% on my data (I really hope that I haven't made some stupid mistakes on data preprocessing and there is no leaks, and this is a real number, but I've tested it on some examples, and it was quite good).

### Why is this submission awesome and should definitely win?
 * It have no dependencies at all, the whole project (inference part at least) is written in pure C, so it can be run on any machine (only avx2 is needed for inference speed, but x86-64 modern processors should definitely have one).
 * The whole library (single .so file) can be used standalone, you aren't even need to take the resources dir with it, because the weights are compiled into the binary, which also helps to avoid slow initialization at first language detection run.
 * It is a real RNN which can be run in about 5ms per 4096 symols text.
 * The final accuracy 94.5% is not bad for classification of 100 languages and this small time of inference.

### What I wanted also to try, but had no time for:
 * The single-layer GRU is fast but not too precise, so I wanted to add a second layer, and to make it fit into the restrictions, quantize it into int8 and train in quantization-aware manner. But it is a lot larger project than one-week contest...
 * It is also good idea to try some other non-linear methods, like random forest or gradient boosting, however all this methods also follows the bag-of-words pradigm, so I decided to try it at last time because I've already had SVC for this purposes.

### Instead of conclusion

I really liked participating in this contest, I have learned a lot of things, recalled good-old friend C, wrote my own micro nn library. It was really fun and I want to thank organizers for the opportunity to try myself in such unsusual ML contest, cause in most contests the prediction metrics are important so you only train models and prepare data, and in this one you need also to show your low-level coding skills to make solution real fast and precise at once. That's cool!

UPD In this submission there is only inference part of my solution, all the data scraping, preparing and training scripts will be available [on my github](https://github.com/kst179/telegram-ml-contest), after deadline.
