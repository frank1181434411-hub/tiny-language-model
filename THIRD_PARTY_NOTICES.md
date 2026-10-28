# Third-party notices

The root LICENSE covers this project's original software and documentation
contributions. It does not replace the notices below or relicense external
dependencies, datasets, model weights, or third-party excerpts.

## Stanford CS336 Assignment 1

This project grew from the owner's implementation of Stanford CS336 Spring 2025
Assignment 1. Migrated components include BPE, tokenization, model layers and
attention, loss, optimization, scheduling, sampling, checkpointing and generation.
The migration and local source evidence are recorded in docs/provenance.md.
The upstream notice is retained for any incorporated or adapted course material;
this does not attribute all project code to Stanford University.

Source: https://github.com/stanford-cs336/assignment1-basics

License: licenses/Stanford-CS336-LICENSE.txt (MIT; preserved from the local source).

## GPT-2 pre-tokenization pattern

The GPT-2-style pattern in src/tiny_lm/tokenizer/rules.py was carried over from
the owner's course implementation. It uses a grouped contraction alternative
instead of the separate alternatives in OpenAI's encoder.py. The upstream notice
is retained for this adapted pattern. The project does not bundle GPT-2 model
weights or the complete GPT-2 encoder implementation.

Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py

License: licenses/OpenAI-GPT2-LICENSE.txt (upstream Modified MIT License).

## External materials

Dependencies retain their own licenses and are installed separately. TinyStories
corpora, derived token files, trained tokenizer artifacts, checkpoints and
reproduction bundles are not granted a license by the project's MIT LICENSE.
Any separate distribution of those materials needs its own source and license
record. Experimental text samples are not a declaration of exclusive ownership
of training data or third-party text.
