# megaDNA: A long-context language model for the generation of bacteriophage genomes
![Online_figure](https://github.com/lingxusb/megaDNA/assets/12596418/ef85a641-0a79-4232-9d09-4abf498f04be)

Generative pre-trained transformers (GPTs) have revolutionized the field of natural language processing. Inspired by the success of large language models, we develop a long-context generative model for genomes. Our multiscale transformer model was pre-trained on unannotated bacteriophage genomes with byte-level tokenization. We demonstrate the foundational capabilities of our model including the prediction of essential genes, genetic variant effects, regulatory element activity and taxonomy of unannotated sequences. Furthermore, it generates de novo sequences up to 96K base pairs, which contain functional regulatory elements and novel proteins with phage-related functions.

#### Trained model
The trained 145M model is availale at [huggingface](https://huggingface.co/lingxusb/megaDNA_145M/tree/main)

#### Model inference
jupyter notebook: [megaDNA_generate.ipynb](https://github.com/lingxusb/megaDNA/blob/main/megaDNA_generate.ipynb). GPU recommended.

Or you can easily run the [Colab Notebook](https://colab.research.google.com/drive/1T7pDY-pL2aJk8mogUKhDu5DpG9r7bjv4?usp=sharing) in the browser. Please make sure to connect to a GPU instance.

### Reference
- [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185)
- [MEGABYTE-pytorch by Phil Wang](https://github.com/lucidrains/MEGABYTE-pytorch)
- Please contact shaobinlx@gmail.com or raise an issue in the github repo with any questions.
