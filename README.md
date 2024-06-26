# Care-tuning, another way to train and fine-tune language models

## Code

The GPT training code in this repository is mostly taken from the excellent [nanoGPT](https://github.com/karpathy/nanoGPT), originally written by Andrej Karpathy, with minor modifications and extensions. The license for the original code can be found in the LICENSES folder.

The main contribution of this codebase is a pipeline for creating semantically-informed training sets, using various types of open data.

## Data

So far, we use the following resources:

* The [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
* [Simple Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
* [Project Gutenberg](https://www.gutenberg.org/)


## Activities

Your system will train on different types of activities, linked to different input and output formats. At present time, activities can be:

* observation (mostly from Visual Genome)
* reading (Wikipedia and Gutenberg)
* skill-training (Visual Genome for question answering over situations, Wikipedia for general knowledge)
* hearing / having conversations (user-generated data)
* thinking (test 'imagination' of the system, using a combination of datasets)
* dreaming (memory replay over all the above)

Different types of data will be suited to one or more of these activities. For instance, the Visual Genome data is good for *observing* and *skill-training*.
