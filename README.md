# Recreating the BERT Architecture (µBERT)

## Abstract

This paper presents a replication of the BERT architecture, µBERT, detailing design choices, methodology, and performance. The goal is to investigate how faithfully one can reproduce the BERT architecture in a resource-constrained environment while adhering to its principal components. To replace the massive corpora originally employed, a smaller, open-source corpus is created. Additionally, it is necessary to use smaller model dimensions (e.g., fewer layers, reduced hidden size) to accommodate limited GPU availability, yet preserve core mechanisms such as multi-head self-attention and learned positional encodings. Our training pipeline utilizes the same objectives, and is executed on a single A100 GPU. Although smaller-scale pre-training yields lower performance compared to official BERT releases and alternatives, µBERT still puts up strong results.

## Introduction

The Bidirectional Encoder Representations from Transformers (BERT) architecture, is a landmark achievement in the realm of multi-purpose pre-trained language models (PLM). It is the inspiration of many, modern encoder-only systems, and proposed foundational techniques for training such models. BERT provided new state-of-the-art performance on various benchmarks such as: named-entity recognition, natural language inference, and question-answering. BERT achieves such extraordinary results by using the transformer architecture to train bi-directional representations of individual tokens, meaning an individual token's representation is a combination of its right and left contexts. The goal of this project is to build a more in-depth familiarity with the BERT architecture and to attempt to replicate the performance in a low-resource setting with the µBERT model. The µBERT model is trained on a more limited corpus than the original BERT models as well as a significantly reduced training time, and despite these limitations, the model still puts up reaonable scores.

## Related Work

During recent years, many variations of the BERT model have been made. These models vary by size, domain, and training method.
One such variation is the robustly optimized BERT (RoBERTa) variant. This model retains much of the architecture of BERT, but is trained for a longer time and with more data. An additional modification of this model is the removal of the next-sentence prediction (NSP) head, training only on the masked language modeling (MLM) objective.
Another variation is a lite BERT (ALBERT), which is a parameter-efficient modification of the BERT architecture. The main advances in this approach are parameter sharing across transformer layers and factorized embedding parameterization. Factorized embedding means that instead of projecting from the vocabulary directly to the hidden size, there is an intermediate layer and dimension which can drastically reduce the total number of parameters needed. Additionally, the NSP loss is replaced with a sentence-order prediction loss, which aims to better capture multi-sentence structure.
DistilBERT is a distilled version of BERT. This model aims to maintain BERT's high-level of NLP ability while reducing its size. This is achieved by halving the number of layers in the model and initializing the parameters from alternating layers from the original BERT and then the resulting model is trained on the original BERT training dataset.
A final model that differs in architecture is SpanBERT. SpanBERT modifies the original MLM objective to instead mask spans, or sequences of consecutive tokens, aptly dubbed masked span prediction. The model then attempts to predict the masked spans using the representations of the tokens that begin and end the span, this is known as the span boundary objective.
Lastly, there are many other BERT versions that differ mainly in the training corpus used. Such example are BioBERT, which is a fine-tuned version of BERT which is trained on additional biomedical corpora. ClinicalBERT is another of these variants, which is mainly trained on health records and clinic notes.
As seen by the myriad variations presented above, the BERT architecture is not only flexible but viable across a multitude of tasks and domains.

## Methods

This section describes the methods used to construct the corpus, architecture, and challenges encountered to pretrain the µBERT model.

### Dataset

It is important to note that the datasets that the original BERT model was trained on were not used for this project, BooksCorpus and English Wikipedia. BooksCorpus is no longer publicly available due to copyright infringement and the English Wikipedia corpus is too large to realistically train on given the parameters of this project. Thus, this project is trained using a corpus of 5,290 books supplied by Project Gutenberg. Project Gutenberg is a website that stores and freely distributes electronic books that have fallen into the public domain. In order to collect this corpus, books from the Project Gutenberg website are systematically scraped from the and saved into separate text files. This separation allows for the extraction of contiguous sentences within a text, which is necessary for the next-sentence prediction (NSP) pre-training task. Additionally, in order to guarantee that the extracted text is meaningful and to remove repeated lines, all text not within the spans marking the beginning and the end of the book are removed. All the text in this corpus is then tokenized using the BERT-base-uncased tokenizer which is available on HuggingFace.

The script `create-dataset.py` handles the downloading, cleaning, and sentence splitting of texts from Project Gutenberg.

### Pre-Training Objectives

The approach to pre-training the µBERT model follows the original BERT implementation and involves two main tasks, masked-language modeling (MLM) and NSP. While later models do away with the latter of these objectives due to it having limited use, it is still included here.
The MLM task is defined as predicting the correct token given the token's context. The idea is that the model learns a rich representation of an individual token, which can then be used for downstream token-level tasks. In BERT pre-training, a MLM head is used to predict a probability distribution over the model's output vocabulary given the representation of token. In order to save on computation, the linear component of the MLM output head is tied to the token-embedding weights. The approach of the MLM task is that for every training example 15% of non-special tokens are selected (e.g., not `[CLS]`, `[SEP]`, etc). Then, 80% of these are replaced by the `[MASK]` token, 10% are replaced by a random token in the vocabulary, and 10% are not modified. The model then creates probability distributions for each of the selected tokens and the loss is calculated using categorical cross-entropy loss when the probability distribution is compared to the one-hot encoded original token for a selected index, and used to update all model parameters.
The NSP task is defined by predicting whether a sentence is followed or not by another sentence. The idea is that the model learns sentence relationships and inter-sentence coherence, which can then be used for natural-language inference and question-answering tasks downstream. In BERT, the NSP head is used to predict whether a sentence, B, follows the sentence, A, which are tagged with different segment-embeddings and separated by the `[SEP]` token. During training, 50% of sentence Bs do follow A, and 50% do not. This allows for equal training across categories. When selecting a sentence that does not follow, any sentence within the corpus becomes a candidate. This head has an output dimension of two, meaning the model can again calculate loss using categorical cross-entropy loss when compared to the one-hot encoded correct label, which is then used to update all model parameters.

### Architecture

The µBERT architecture is an encoder-only transformer. This paper will present the model architecture from the input to the output. Note that whenever the term 'normalized' is used in this section, it refers to the output being passed through a layer normalization and then a dropout layer.
The first step involves summing the outputs of three embedding layers: token, positional, and segment embeddings. The token embedding layer is a linear projection from the vocabulary size to the model's hidden size, using token IDs as input. The positional embedding layer uses learned positional encodings. It maps positional IDs (up to a maximum of 256) to the hidden size via a learned embedding matrix. Finally, the segment embedding layer maps segment IDs (with a maximum of 2 segments) to the hidden size, enabling the model to distinguish between different input segments.
The summed output of the embedding layers is then normalized. This output is then passed through the encoder stack. The encoder stack is made up of 6 multi-head self-attention transformer blocks. Each transformer block first applies multi-head self-attention to the input. This is done by first calculating the size of the head dimension, which is the hidden size divided by the number of heads; this must divide evenly. Different attention heads within a multi-head attention mechanism are able to capture different aspects of the input simultaneously. This division of attention into multiple heads is meant to increase the models expressiveness. The hidden state to the module is then linearly transformed to create both the query (Q), key (K), and value (V) vectors. Intuitively, these vectors can be understood as; what the model is looking for, what the model has, and what the model can retrieve, respectively. These vectors are then combined using scaled dot-product attention as seen below:
`Z = softmax((QK^T / sqrt(d_k)) + attn_mask)V` (1)
Where `softmax` refers to the softmax function and $d_K$ is the dimensionality of the key vector. The attention mask is created by setting all tokens to be masked to -10000, which essentially zeroes the values out after the softmax, causing these positions to not be attended to and Z is the output attention, often referred to as the context vector. The last part of the attention module is a simple projection layer of the context to the hidden dimension. This attention is then passed through a dropout layer before being added back to the residual stream which is then normed before being passed to the next steps.
The last steps of the transformer block has the residual stream being passed to a simple feed-forward network. This network consists of two linear layers that first project to four times the hidden dimension before being projected back to the hidden dimension, and separated by a GELU. This increase in dimension and non-linearity is used to introduce additional expressivity to the model. The output is then passed through a dropout layer before being added back to the residual stream which is again passed through a layer normalization. The purpose of using residual streams in transformers is mainly to prevent the vanishing gradient problem, allowing the model to back-propagate effectively.
The output of this stack is then used as the BERT model's outputs, however, for pre-training it is necessary to add two additional heads to this output. These are the aforementioned MLM and NSP heads. The MLM head is made of two linear layers separated by a GELU and a layer norm; the second linear layer projecting to the vocabulary size and having its weights tied to the embedding weights. The NSP head is simply a linear projection to a dimension of two.

The Python script `BERT-train.py` contains the implementation of this architecture.

### Hyperparameters

| Parameters        | µBERT | BERT-base | BERT-large |
|-------------------|-------|-----------|------------|
| Batch Size        | 1024  | 256       | 256        |
| Hidden Layers     | 6     | 12        | 24         |
| Hidden Size       | 384   | 768       | 1024       |
| Attn. Heads       | 6     | 12        | 16         |
| Dropout Prop.     | 0.1   | 0.1       | 0.1        |
| Learning Rate     | 1e-4  | 1e-4      | 1e-4       |
_Table 1: Relevant hyper-parameters when comparing µBERT to vanilla BERT implementations._

Additionally, the batch size used is 1024, which was the largest batch size that could fit on the GPU used to train, as it was shown that BERT models benefit from larger batch sizes.

### Model Parameters

| Model       | # Params |
|-------------|----------|
| BERT-base   | 110M     |
| DistilBERT  | 66M      |
| ALBERT      | 12M      |
| µBERT       | 22M      |
_Table 2: A comparison of the total number of parameters between BERT and its recreations._

### Difficulties and Challenges

Two primary challenges during the training of µBERT were the availability of sufficient training data and adequate computational resources. In terms of data resources, while the collected corpus is extensive and of high quality, it does not match in scale to the corpora used in previous BERT variants. Consequently, it is unlikely, even given all else being equal, that µBERT will be ability to reach the performance of the competition. Secondly, in regard to computational resources, the compute available for this project is severely limited compared to the original BERT. Specifically, the model described in this paper is trained using a single NVIDIA A100 GPU provided by Google Colab, with training limited to approximately one 8-hour session due to an automatic restart of the platform. In contrast, the original BERT model utilized 16 TPUs simultaneously over multiple days, highlighting the computational disparity. Due to this limitation, the model presented here is the highest-performing checkpoint obtained within the available training time.

## Experiment: Part-of-Speech (POS) Tagging

This section presents a benchmark for the performance of µBERT on a part-of-speech (POS) tagging task. It compares the model to several BERT-based baselines, including BERT-base, DistilBERT, and ALBERT. POS tagging serves as a metric for evaluating a model's syntactic understanding, particularly at the token level, something hopefully learned from MLM pre-training.

### Dataset (POS Tagging)

The dataset used is the Universal Dependencies English Web Treebank (UD-EWT), which contains approximately 16,000 annotated sentences and 17 POS tags. The data was preprocessed using the standard BERT tokenizer, and the pre-defined splits were used. The number of sentences per split can be seen in Table 3.

| Split      | Num. Sentences |
|------------|----------------|
| Train      | 12543          |
| Validation | 2002           |
_Table 3: Statistics from the English Web Treebank dataset from Universal Dependencies._

### Design and Training (POS Tagging)

The task of POS tagging requires a model to label words with their corresponding POS tag. This task is listed in the original BERT paper as a task on which this model excels. The implementation in this paper attempts to measure, mainly, the linguistic knowledge the model acquires during pre-training. To do so, the whole model is frozen during fine-tuning, and only a simple linear-layer is placed on top of the output of the BERT models. Evaluation was conducted using tagging accuracy on the test set. Figure 1 shows an example of how tagging is performed by µBERT. Only the train and validation splits are used, as there is no additional tuning to the validation data, it is used to compare the models' performances.

All models were fine-tuned using a batch size of 128 and a learning rate of 0.001. Additionally, the training length is maintained at five epochs. This ensures a balance in terms of time permitted to learn the new task and does not overtly benefit models that have higher output dimensions. It should be noted that training-time differed significantly between the different implementations, based on the number of parameters in the model.

The evaluation script is `BERT-evaluate.py`.

### Results (POS Tagging)

| Model      | Performance (Accuracy) |
|------------|------------------------|
| BERT-base  | 0.92                   |
| DistilBERT | 0.92                   |
| ALBERT     | 0.94                   |
| µBERT      | 0.77                   |
_Table 4: Performance of various BERT implementations on the POS tagging task._

While the original BERT and other variants perform relatively similarly, with greater than 90% accuracy, the µBERT model reaches only 77% percent. This performance gap between µBERT and the other models ranging from 15% to 17% highlights the limitations inherent to a severely undersized and under-trained model. However, ALBERT, being the smallest model, still manages to put the best results. Additionally, the loss on the train split and the accuracy on the validation split can be seen in Figure 2. It can be seen that while the training loss decreased drastically over the first 3 epochs, the validation accuracy did not increase by more than 5%. The training loss and validation accuracy curves for the other models can be seen in Appendix A of the paper.

## Discussion

Given the results of Table 4, it is shown that the µBERT model did successfully learn to internalize relevant information from its pre-training to apply to this task. As noted earlier, Figure 2 shows the train loss had not fully plateaued and that perhaps the model still had the capacity to better learn the task with additional fine-tuning. It is also possible that this sort of curve is showing that the model is overfitting and memorizing patterns in the training data. It is also possible that the µBERT model performed well on this task due since it involves modeling low-level linguistic structure. It has been shown that the earlier layers of the BERT model perform more strongly on such tasks than later layers. Thus µBERT, having fewer layers and less training time, may only have only had time to develop low-level linguistic understanding. It would be interesting to see how these models compare on a more semantic based task, such as sentiment analysis. This experiment also proves that size is not directly linked to the amount of performance expected from a model. As shown, ALBERT, which has 12M parameters, outperforms all other models, while having less parameters than even µBERT. This means that there are techniques to make such small language models usable. This additionally implies that there are many techniques that could improve the performance of µBERT such as parameter sharing, larger pre-training corpus, and longer pre-training time.

## Conclusion

Despite the slightly lower performance of the µBERT model, it demonstrates how flexible the BERT architecture is. By adhering to the encoder-only transformer framework, even a reduced model was able to internalize much of the fundamental linguistic structure necessary for the POS tagging task. However, its reduced performance, even compared to a smaller model, shows that there are still strategies to improve.
This project set out to understand the inner workings of BERT by building a version that remains faithful to its core components but operates within realistic resource limitations. The motivation was not to achieve state-of-the-art results, but to observe what can be expected from such a model. The process of designing, training, and evaluating µBERT offers a look into the decisions and trade-offs that are made in transformer design and how these decisions compare to other models, showing how data quantity, architecture size, and training length impact downstream task performance.

## Limitations

The main limitations of this study include the fact that only one downstream task was run, there was minimal hyperparameter tuning, and there was a reliance on a single model checkpoint. These factors likely affected the generality and reproducibility of results. While there are various changes that could be implemented to narrow the performance gap with other BERT variants without modifying the training corpus or compute budget, such as parameter sharing, distillation, or improved pre-training objectives, this project finds that such resource-limited replications of the BERT architecture, while not directly competitive with larger BERT variants, remain a valuable exercise in understanding the power of transformer-based language models.

## Scripts

* **`create_dataset.py`**: Downloads books from Project Gutenberg, cleans the text, and splits it into sentences to create the pre-training corpus. It requires a `gutenberg_metadata.csv` file (not provided in the prompt) with book information including download links.
* **`BERT_train.py`**: Defines the µBERT model architecture (including `BertModel`, `TransformerModule`, `MultiHeadSelfAttention`, `FeedForward`, and `BertPretrainModel` classes) and the pre-training procedure (MLM and NSP tasks using `BertPretrainDataset`). This script is designed for a Google Colab environment and uses Google Drive for data and model saving.
* **`BERT_evaluate.ipynb`**: Loads pre-trained BERT variants (defaulting to `bert-base-uncased` from HuggingFace so it runs from default), sets up a Part-of-Speech (POS) tagging task using the Universal Dependencies `en_ewt` dataset, fine-tunes a linear layer on top of the frozen BERT models, and evaluates their performance.

## How to Run

1.  **Dataset Creation**:
    * Obtain or create a `gutenberg_metadata.csv` file containing columns 'Author', 'Title', 'Link', and 'Bookshelf' for books from Project Gutenberg. Link to the corresponding Kaggle page is found in the file.
    * Place `gutenberg_metadata.csv` in the same directory as `create-dataset.py`.
    * Run `python create_dataset.py`. This will create a directory named `gutenberg_texts` and populate it with processed text files from the downloaded books.

2.  **Pre-training µBERT**:
    * Modify `DATA_DIR` in `BERT-train.py` to point to your `gutenberg_texts` directory.
    * Modify `SAVE_DIR` to specify the path where model checkpoints should be saved.
    * Run the `BERT_train.py` script in your Colab environment. Checkpoints will be saved to the specified `SAVE_DIR`.

3.  **Evaluating Models on POS Tagging**:
    * The `BERT_evaluate.ipynb` script by default loads `bert-base-uncased` from HuggingFace for evaluation.
    * To evaluate your custom-trained µBERT:
        * You will need to modify the model loading section in `BERT_evaluate.ipynb`. This involves:
            1.  Instantiating `BertModel` (defined in `BERT-train.py`, so you might need to copy this class definition or import it) with µBERT's specific hyperparameters (number of layers, hidden size, attention heads).
            2.  Loading the state dictionary of the `bert_encoder` part of your saved µBERT checkpoint into this `BertModel` instance.
            3.  Ensuring the `NeuralTagger` class in `BERT-evaluate.py` is compatible with your µBERT's output dimension (e.g., hidden size 384 for µBERT instead of 768 for BERT-base). The `NeuralTagger`'s linear layer input size (`768`) should be changed to match µBERT's `HIDDEN_SIZE` (384).
        * The script uses the Universal Dependencies `en_ewt` dataset, which it will attempt to download.
    * Run the `BERT_evaluate.ipynb` script. It will fine-tune the linear tagger and output training loss and validation accuracy, along with a plot.
    * The model can then be compared with other, competing models on this task.
