# Language

**Syntax** is sentence structure. **Semantics** is the meaning of words or sentences. While the sentence "Just before nine o'clock Sherlock Holmes stepped briskly into the room" is syntactically different from "Sherlock Holmes stepped briskly into the room just before nine o'clock," their content is effectively identical.

**Formal Grammar** is a system of rules for generating sentences in a language. In **Context-Free Grammar**, the text is abstracted from its meaning to represent the structure of the sentence using formal grammar. `She saw the city.` => `N V D N` (noun, verb, determiner, noun)

Bag-of-words is a model that represents text as an unordered collection of words. This model ignores syntax and considers only the meanings of the words in the sentence. This approach is helpful in some classification tasks, such as sentiment analysis (another classification task would be distinguishing regular email from spam email). Naive Bayes is a technique that can be used in sentiment analysis with the bag-of-words model.

One-hot vs. distributed representations (where meaning is distributed across multiple values in a vector): "He wrote a book".

    [1, 0, 0, 0] (he)
    [0, 1, 0, 0] (wrote)
    [0, 0, 1, 0] (a)
    [0, 0, 0, 1] (book)

    [-0.34, -0.08, 0.02, -0.18, ...] (he)
    [-0.27, 0.40, 0.00, -0.65, ...] (wrote)
    [-0.12, -0.25, 0.29, -0.09, ...] (a)
    [-0.23, -0.16, -0.05, -0.57, ...] (book)

This allows us to generate unique values for each word while using _smaller vectors_. Additionally, now we are able to _represent similarity_ between words by how different the values in their vectors are.

## Introduction

- **Natural language processing (NLP)** has as a goal:
  - Natural language understanding (NLU):
    - Setence-level:
      - Recognizing actions and states and the actors that play a role in the events, which includes semantic role labeling, i.e., recognizing "who" does "what", "where", "when" and "how"
      - Detecting modality, i.e., the factual status of a statement, which can involve negation, possibility and obligational entailments of statements
      - Coreference resolution, i.e., identifying coreferring expressions in a discourse (e.g., that refer to the same entity)
    - Discourse-level:
      - Coreference resolution
      - Recognizing temporal, spatial and causal relations between actions, entities and events, and identification of other relationships
      - Understanding full discourses and dialogues
    - Is often evaluated by a question-answering task, but is that sufficient?
    - Does real understanding involve the reconstruction of the physical and social world that is conveyed by language?
  - Natural language generation (NLG):
    - Starting from representations (e.g., symbolic, neural) generate valid and contextually meaningful natural language: E.g., used in machine translation, text rewriting, text simplification, dialogue creation, summarization, etc.
- **Challenges**: Why are NLU and NLG difficult?
  - Processing methodology: ML vs. rule-based techniques, feature engineering vs. (large training data needed)
  - Lack of annotated training data: manual annotation is expensive; foundation models whose pre-trained knowledge about language can often be leveraged in downstream tasks to reduce the number of annotated training data
  - Dealing with ambiguity, uncertainty and incompleteness of language: need the context of the discourse to disambiguate
  - The symbolic system of language versus the continuous nature of current representations and their integration: Large vocabulary, symbolic encoding of words creates a problem of sparsity, distribution of linguistic patterns (words and syntactic structure) with long tail - many rare cases
  - Sparse versus dense representations
    - One-hot encoding: a word is represented by a high-dimensional, sparse vector
      - Pro: each dimension has an explicit semantic (i.e., we know what it is)
      - Cons: hard to use in ML/DL, no representation of word correlations
    - Word embeddings: each word is associated with a real valued vector in d-dimensional space (usually d = 50 - 1000, the semantic representation) often computed with a NN
  - The role of memory when processing language
    - When humans process language they rely on memory of: prior knowledge acquired during a lifetime, content previously occurring in the discourse.
    - Open challenge: how to model this memory, retrieve information and integrate it in the current processing.
- **Evaluation of classification tasks**: Recall, precision, etc.
  - Macro averaging: averaged over classes.
  - Micro averaging: averaged over all classification decisions.
  - F-measure: trade off precision against recall. (F1 = balanced F / harmonic mean)
- **Evaluation of NLG**:

  - BLEU (BiLingual Evaluation Understudy): precision-oriented, measures the amount of overlap between system-generated text and reference texts (normalised by the number of n-grams in the test corpus or text)
  - ROUGE (Recall-Oriented Understudy for Gisting Evaluations): based on n-gram recall
  - For evaluating NLG by LLMs BLEU and ROUGE do not suffice, need human evaluation.
  - Inter-rater agreement: e.g., Cohen's kappa, which is a chance-corrected statistic for assessing agreement among annotators
    $$\kappa = \frac{P(a) - P(e)}{1 - P(e)} = 1 - \frac{1 - P(a)}{1 - P(e)}$$
    - $P(a)$ = the probability of agreement between annotators, $P(e)$ = the expected agreement by chance if the ratings are independent
    - If the annotators completely agree, then $P(a) =1, \kappa = 1$.
    - If there is no agreement ($P(a) = 0$) or the agreement is worse than chance ($P(a) < P(e)$), $\kappa$ is negative.
    - Agreement rate between humans on gold standard usually seen as upper bound for system performance.

## Sequential tagging

- Use case: Part-of-speech (PoS) tagging
  - Parts of speech = word classes or syntactic categories of words, e.g. ADJ, ADV, NOUN, VERB, etc.
  - POS tag = linguistic signal on how a word is being used within the scope of a phrase, sentence or discourse - important to infer semantic information, e.g. object, subject, etc.
- Tagging models:
  - Generative models: e.g. Hidden Markov model, how to generate the input data using the class conditional density
  - Discriminative models: e.g. Conditional Random Fields (CRF), RNNs, Transfomers, how likely a label is to apply to the instance
  - Token taggers: e.g. Maximum entropy (Maxent)
  - Sequence taggers, e.g. HMM, CRF
- HMMs: a generative model to _choose the tag sequence_ $l_{1:T}$ _that is most probable given the observation sequence of words_ $x_{1:T}: \argmax_{l_1:T}{P(l_{1:T} \mid x_{1:T})}$.
  $$\hat{l}_{1:T} = \argmax_{l_1:T}{P(l_{1:T} \mid x_{1:T})} \propto \prod_{t=1}^T{P(x_t \mid l_t) P(l_t \mid l_{t-1})}$$
  - $P(x_t \mid l_t)$ emission probabilities, $P(l_t \mid l_{t-1})$ transition probabilities
  - directed graphical model (Bayes net) to compute the joint probability distribution of the sequence of T tags or labels $l_{1:T} = P(l_1) P(l_2 \mid l_1) P(l_3 \mid l_1, l_2) \dots P(l_T \mid l_{T-1})$ (Markov chain).
  - assumption of conditional independence (limited horizon): $l_{1:T} = P(l_1) P(l_2 \mid l_1) P(l_3 \mid l_2) \dots P(l_T \mid l_{T-1}) = P(l_1) \prod_{t=2}^T{P(l_t \mid l_{t-1})}$ (first-order Markov chain - dependence only on previous tag).
  - assumption of time-invariance (stationarity) or probabilities: does not change over time
  - usually trained with Baum-Welch algorithm for learning the HMM parameters, with MLE on the (labelled) training data
  - Next: _discriminative models can easily model non-independent features_
- Maximum entropy tagging and conditional random field
  - Note: the output of a linear classifier is a _score_ for each class y, not a probability.
- Maximum entropy classifier (multinomial logistic regression) is a popular classification approach in NLP: the probability of assigning class y to instance x given K feature functions $f_j$ is
  $$P(y \mid x) = \frac{\exp\big( \sum_{j=1}^K{w_j f_j(y, x)} \big)}{\sum_{y_c \in C}{\exp\big( \sum_{j=1}^K{w_j f_j(y_c, x)} \big)}}, \qquad \hat{y} = \argmax_{y_c \in C}{P(y_c \mid x)}$$
  - It allows not only to integrate many different features (or feature functions) that act as constraints on the model
  - It allows to model uncertainty in an elegant way: choose the model $p^*$ that preserves as much uncertainty as possible, or which maximises the entropy $H(p)$ between all the models $p \in P$ that sastisfy the constraints enforced by the training examples:
    $$p^* = \argmax_{p \in P}{H(p)}, \qquad H(p) = - \sum_{(x, y)}{P(x, y) \log{P(x, y)}}$$
  - _Maximum entropy_: The optimal weights (parameters) are the ones for which each feature's empirical or observed expectation in the training data $E_p(f_j) = E_{\hat{p}}(f_j)$ (the predicted expectation).
- Conditional random field: undirected graphical model of the family of Markov random fields or Markov networks:
  - There is a dependency between output labels modeled, but the dependency has no directions
  - a discriminative model to predict $P(Y \mid X; \theta)$
  - (Binary or real-valued) feature functions $f_j$ model features of local inputs as well as transitions between labels
  - Maximise the conditional log likelihood over the training data.
- Decoding (i.e. inference) strategies for sequence taggers: finding the best or good sequence based on the probabilities of the tag/word prediction
  - Strategy 1 - greedy decoding: predict the target sequence label by label (or generate target sequence word by word) by emitting the label / word with maximum probability (no backtracking, decisions cannot be undone).
  - Strategy 2 - predicting the best (i.e. most probable) sequence of words / labels: search through all possible output sequences and find the one with highest probability, complexity of $\mathcal{O}(\lvert V \rvert^T)$ where V is the vocabulary and T the sequence; this is often computationally intractable (NP-complete)
  - Strategy 3 - Beam search: at every step $t$, the decoder keeps track of the $k$ most likely continuations of the $k$ partial solutions of the previous step where $k$ is the beam size (often between 5 and 10)
  - Sometimes it is possible to remember partial results rather than recomputing them, decreasing computational complexity: dynamic programming or memorisation, e.g. Viterbi
- Evaluation of POS tagging
- Use cases of sentence meaning: named entity recognition, entity linking and relation extraction
  - Named entity recognition (NER) = to find each mention of a named entity in the text and label its type, e.g. biomedical domain: names of genes and proteins
    - Two problems: segmentation, classification
    - Focus on sequence taggers: prediction of a sequence of labels, e.g. CRF, RNNs such as LSTMs.
    - RNNs: learn distributed representations of text across positions in a sequence
      - Suffer from short-term memory: for long sequences have a hard time carrying informatoin from earlier time steps to later ones; during backprop the vanishing gradient problem arises in the early RNN layers - layers that get a small gradient update stop learning, therefore an RNN can forget what it has seen in longer sequences.
      - LSTM: solution to vanishing gradient problem with gated recurrent units - an internal cell state which acts as a memory that can preserve information across multiple timesteps and gates that determine how the memory is updated.
        - Input gate: control what input information to incorporate.
        - Forget gate: control what past information to forget.
        - Output gate: control what information from the memory cell to output.
        - State (memory) cell: combine past and new information.
        - BiLSTM with 2-4 layers was regarded as the default option for NLP between 2016 and 2018
        - LSTMs: strong tendency towards count, add, mul
        - Transformers: strong tendency towards memorization and hierarchical
    - Neural networks with CRF top layer: Neural networks are very powerful feature learners and on top you have a classification layer which mimics the structural dependencies of a CRF
      - Without CRF layer: we would choose the sequence labeling that for each output of the Bi-LSTM chooses the label with maximum score (or probability if softmax is applied)
      - The CRF layer will impose additional constraints learned from the training data: e.g., first-order Markov dependencies between labels
    - Character-based approaches struggle to produce meaningful embeddings if a rare string is used in an underspecified context - Pooling (= combining results to obtain an estimate of the global effect):
      - Dynamically aggregate contextualized embeddings of each unique string using a pooling operation to distil a global word representation from all contextualized instances
      - Combination with the current contextualized representation, producing evolving word representations that change over time as more instances of the same word are encountered in the data
    - The best performing systems use good word representations pretrained on large corpora
  - Entity linking: for each mention, link to the corresponding entity in the knowledge base; if there is no corresponding entity in the knowledge base, predict that.
    - BLINK uses a two-stages approach for entity linking, based on fine-tuned BERT architectures:
      1. It perforrms retrieval in a dense space defined by a bi-encoder that independently embeds the mention m + context c and the entity description e.
      2. Each candidate is then examined more carefully with a cross-encoder, that combines the mention + context and entity description.
  - Relation extraction (RE) predicts relations between entities in sentences
    - Input: sequence of words or tokens (usually a sentence)
    - Output: sequence of labels (often in form of structured labels, e.g. triplets)
    - Task: translate observation sequence of words $w_{1:T}$ to most probable sequence of triplets $\text{tr}_{1:L}$
- Multi-task learning:
  - assume a shared underlying syntactic-semantic representation
  - separate loss for each available supervision signal and then sum the losses into a single loss that is used for computing the gradients

## Language Modelling and Attention

By taking the attention score, multiplying them by the hidden state values generated by the network, and adding them up, the neural network will create a final context vector that the decoder can use to calculate the final word. A challenge that arises in calculations such as these is that recurrent neural networks require sequential training of word after word. This takes a lot of time. With the growth of large language models, they take longer and longer to train. A desire for parallelism has steadily grown as larger and larger datasets need to be trained. Hence, a new architecture has been introduced.

Transformers is a new type of training architecture whereby each input word is passed through a neural network simultaneously. An input word goes into the neural network and is captured as an encoded representation. Because all words are fed into the neural network at the same time, word order could easily be lost. Accordingly, position encoding is added to the inputs. The neural network, therefore, will use both the word and the position of the word in the encoded representation. Additionally, a self-attention step is added to help define the context of the word being inputted. In fact, neural networks will often use multiple self-attention steps such that they can further understand the context. This process is repeated multiple times for each of the words in the sequence. What results are encoded representations that will be useful when it’s time to decode the information.

In the decoding step, the previous output word and its positional encoding are given to multiple self-attention steps and the neural network. Additionally, multiple attention steps are fed the encoded representation from the encoding process and provided to the neural network. Hence, words are able to pay attention to each other. Further, parallel processing is possible, and the calculations are fast and accurate.

- **Attention**: in psychology, the _cognitive process of selectively concentrating_ on one or a few things while ignoring others; contributes to the explainability of the model!
  - (Cross) attention: between tokens of different sequences or modalities (e.g. between encoder and decoder)
  - Self (or intra) attention: between tokens of the same sequence or instance of same modality (e.g. within the encoder or within the decoder)
- Encoder-decoder architectures: in RNNs there is a bottleneck because requiring the context $c$ to be only the encoder's final hidden state forces all the information from the entire source sentence to pass through this representational bottleneck.
- Cross attention in encoder-decoder architectures
  - The attention mechanism allows each hidden state of the decoder to see a different, dynamic context, which is a function of all the encoder hidden states.
  - Attention thus replaces the static context vector with one that is dynamically derived _from all the encoder hidden states_, different for each token in decoding.
  - The attention weights focus on ("attend to") a particular part of the encoded text that is relevant for the token the decoder is currently producing: this information is captured in a context vector $c_i$
  - This context vector $c_i$ is generated anew with each decoding step $i$ and takes all encoder hidden states into account but in a weighted manner
  - The context vector $c_i$ is often concatenated to the current decoder hidden state
  - At each state $i$ during decoding, a $\text{score}(\bm{h}_{i-1}^D, \bm{h}_j^E)$ for each encoder state $\bm{h}_j^E$ is computed.
  - The simplest such score, called dot-product attention, implements relevance as similarity: measuring how similar the decoder hidden state is to an encoder hidden state, by computing the dot product between them:
    $$\text{score}(\bm{h}_{i-1}^D, \bm{h}_j^E) = (\bm{h}_{i-1}^D)^T \cdot \bm{h}_j^E$$
  - The vector of these scores across all the encoder hidden states gives us the relevance of each encoder state to the current step of the decoder
  - Normalise the obtained scores with a softmax to create a vector of attention weights $\alpha_i$ that gives the proportional relevance of each encoder hidden state $\bm{h}_j^E$ to the prior hidden decoder state $\bm{h}_{i-1}^D$; each component of $\alpha_i$ is computed as:
    $$\alpha_{i, j} = \frac{\exp\big( \text{score}(\bm{h}_{i-1}^D, \bm{h}_j^E) \big)}{\sum_k{ \exp\big( \text{score}(\bm{h}_{i-1}^D, \bm{h}_k^E) \big) }}$$
  - Finally, given the distribution in $\alpha_i$, we can compute a fixed-length context vector for the current decoder state by taking a weighted average over all the encoder hidden states:
    $$c_i = \sum_j{\alpha_{i, j} \bm{h}_j^E}$$
  - With this, we finally have a fixed-length context vector that takes into account information from all encoder states that is dynamically updated to reflect the needs of the decoder at each step of decoding
- Self-attention: one word differently attends to other words in a sentence - refine the representation of a word by considering the representation of related words in the sentence.
  - The attention operation can be seen as retrieving:
    - In the case of cross-attention from a set of elements $\bm{h}_j^E$ (keys, values) from the encoder according to the $\bm{h}_i^D$ query.
    - In the case of self-attention from a set of elements $\bm{h}_j^E$ (keys, values) from an encoder according to the $\bm{h}_i^E$ query, or $\bm{h}_j^D$ (keys, values) from a decoder according to the $\bm{h}_i^D$ query.
  - Given a sequence of representations $\bm{x}_{1:L}$, compute a query and key vector both of dimension $d_k$, and a value vector of dimension $d_v$ with linear transformations, learning the weight matrices $W^Q \in \mathbb{R}^{d_\text{in} \times d_k}, W^K \in \mathbb{R}^{d_\text{in} \times d_k}, W^V \in \mathbb{R}^{d_\text{in} \times d_v}$
    $$
    \bm{q}_i = \bm{x}_i^T W^Q, \quad
    \bm{k}_j = \bm{x}_j^T W^K, \quad
    \bm{v}_j = \bm{x}_j^T W^V
    $$
    - Compute a score for each element in the attended sequence by taking the dot product of the query and key vectors:
      $$\text{score}_{ij} = \frac{\bm{q}_i^T \cdot \bm{k}_j}{\sqrt{d_k}} \quad \forall j = 1, \dots, L$$
    - Note: The dot product grows large in magnitude, pushing the softmax function into regions where it has extremely small gradients: the division by $\sqrt{d_k}$ counteracts this effect
    - A softmax on the scores obtains a probability distribution: $\alpha_{i,j} = \frac{e^{\text{score}_{ij}}}{\sum_{k=1}^L{ e^{\text{score}_{ik}} }}, \forall j = 1, \dots, L$
    - Multiply each value vector $\bm{v}_j$ with its corresponding attention weight: $\bm{v}_{i, j}' = \alpha_{i, j} \bm{v}_j, \quad \forall j = 1, \dots, L$, and sum up the weighted value vectors as the final output context vector $c_i$: $c_i = \sum_{j=1}^L{\bm{v}_{i, j}'}$
    - This can be done efficiently with matrix multiplications:
      $$\text{Attention}(Q, K, V) = \text{softmax}\bigg( \frac{Q K^T}{\sqrt{d_k}} \bigg) V$$
- Multi-head attention: different heads can focus on different types of information, e.g. one head can focus on nouns, another on context words, etc.
  - The resulting vectors are concatenated in a final context vector: $\text{c\_multihead}_i = [\text{head}_1; \text{head}_2; \dots; \text{head}_m]^T W_o$ where $W_o \in \mathbb{R}^{m d_v \times d_\text{in}}$
- Transformer: neural architecture that uses self-attention
  - Encoder: The encoder block consists of multi- head self-attention layers followed by add and norm layer, and a feed forward neural network followed by add and norm layer
  - Decoder: The decoder block is similar to the encoder block but uses an extra attention step where it calculates cross-attention between the encoder tokens (i.e., the output of the final encoder block) and already predicted decoder tokens
  - Masked attention: only consider past tokens in self-attention weights, enforced by adding a large negative constant to the input to the softmax (or equivalently, setting $\alpha_{ij} = 0$ when $j > i$)
  - Layer normalization (or layer norm) is used to improve training performance in deep neural networks by keeping the values of a hidden layer in a range that facilitates gradient-based training, i.e. with zero mean and standard deviation of one: $\hat{x}_i = \frac{x_i - \mu}{\sigma}, \quad \text{LayerNorm} = \gamma \hat{x} + \beta$ where $\gamma, \beta$ are trainable parameters, respectively referred to as gain and offset.
- Language Models: probabilistic model of word sequences used for word prediction
  - "A foundation model is any model that is trained on broad data (generally using self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks." [Bommasani et al. 2022]
  - Frequency-based n-gram language models which predict the next word from the previous n-1 words, using the chain rule to decompose the joint probability into a sequence of conditional probabilities where $x_i$ is a word or unigram
    - A first order Markov model or chain: dependence is limited within 2 successive states
    - A second order Markov model or chain: dependence is limited within 3 successive states
    - Computed with MLE
    - Some observations:
      - The longer the n-grams, the more coherent the sentences are when generated
      - Language models are pretty useless if evaluation on different dataset than train set: the Wallstreet trigrams do not well predict Shakespeare's language
    - Closed vocabulary: Lexicon of known words
    - Open vocabulary:
      - Unknown words are tagged as <UNK>
      - OOV rate: percentage of unknown words
- Smoothing, discounting, interpolation
  - Problem with MLE when training a frequency-based language model: sparse data !
    - Some n-gram patterns have a zero probability based on the train set: in product => 0
    - Some very small estimates might be rounded to 0
    - Rare n-gram patterns in the training set might mislead the estimates
  - Smoothing assigns some non-zero probability to any n-gram, even if it is not seen in the training data
    - Laplace smoothing, Kneser-Ney smoothing, etc.
  - Smoothing can be viewed as discounting (lowering) non-zero counts to get the probability mass that will be assigned to zero counts
    - E.g., Good-Turing discounting: the count of the things you have seen help estimate the count of things you have never seen
- Neural language models
  - Bigram neural language model trained on large text collection
  - Previous word is used to predict the current word by going through hidden layer (classifier with as many outputs as there are words in the vocabulary)
  - RNN language models: for each time step, apply update equations
    - Advantages:
      - RNNs can represent unbounded dependencies
      - RNNs compress histories of words into a fixed size hidden vector
    - Disadvantages:
      - Pure RNNs are hard to learn and will often not discover long range dependencies => LSTMs
  - Transformer language models
    - Encoder models: e.g., BERT: Tokens are represented with word piece embeddings: obtained by breaking words into pieces given a word unit dictionary (automatically learned) and adding word continuation information
      - BERT training based on two unsupervised prediction tasks:
        - Masked language model: 15% of random input word piece tokens in each sequence are masked and the system has to predict the masked token (during training a small amount of them predicts a noise token to avoid overfitting of the model)
        - Next sentence prediction: Choosing two sentences A and B from the training corpus:
          - 50% of the cases: B is the actual next sentence of A
          - 50% of the cases: B is just a random sentence from the corpus
        - Multi-task with two objectives: masked token prediction and next sentence prediction; Both losses are combined in a sum
    - Decoder models: e.g., GPT-3: fully connected feedforward architecture with multi-head self-attention, decoder blocks with masked self-attention, next word prediction, autoregressive or causal (left-to-right) language model; unsupervised learning
- Byte-pair encoding = compression technique that captures frequent substrings, popular for identifying subword token
  - Input: dictionary of words obtained from a corpus
  - Initial set of tokens = set of characters of the language
  - Until no more bigram tokens can be merged or until a threshold number of tokens is reached (e.g., $10^4$)
    - Merge the most common bigram of tokens into a new symbol
    - Update set of tokens
  - Output: dictionary of segmented words
  - Given dictionary of words {fish, fished, want, wanted, bike, biked} we find tokens {fish, ed, want, bik, e} and can segment into: {fish, fish+ed, want, want+ed, bik+e, bik+ed}
- Evaluation with perplexity: the inverse of the probability of the sequence of test corpus, normalized by the number of words T in the sequence:
  $$PP = P(w_{1:T})^{-\frac{1}{T}} = \sqrt[T]{\frac{1}{P(w_{1:T})}} = \sqrt[T]{\prod_{t=1}^T{ \frac{1}{P(w_t \mid w_{t-1})} }}$$
  - $\frac{1}{T}$: normalisation by number of words
  - Lower is better: minimising perplexity is the same as maximising probability

## Foundation Models and Parameter Efficient Finetuning

- Foundation Models (/ Pre-trained / Self-supervised / Large language models):
  - "A foundation model is any model that is **trained on broad data** (generally using **self-supervision** at **scale**) that **can be adapted** (e.g., fine-tuned) **to a wide range of downstream tasks**." [Bommasani et al. 2022]
  - Self-supervision: Models predict parts of data from other parts.
    - Efficient use of large amounts of unlabelled data.
  - Broad knowledge enables adaptation with little data via transfer learning.
  - Capabilities: performance!
    - _Pretraining + finetuning_ generally improves generalization over training from scratch drastically! [Devlin et al., 2018]
    - Without any finetuning, Large Language Models can achieve strong results from zero or few examples. [Brown et al., 2020]
    - Larger scale unlocks _emergent capabiltiies_ in pretrained language models. [Wei, Tay, et al., 2022]
    - Instruction tuning enables models to solve previously unseen tasks. [Wei, Bosma, et al., 2022]
  - Limitations:
    - High (environmental) cost
    - Limited access
    - Biased datasets
    - Generally uninterpretable
    - Alignment to human values remains a challenge.
- Training:
  - Considerations before pre-training:
    - Downstream Applications
      - Classification, generation, etc.
      - Specific domains of interest?
    - Suitable Training Objective and Architecture
      - Align with desired capabilities
      - Generality vs specificity
    - Data Requirements
      - Size vs quality
      - Domain relevance
    - Computation and Cost
      - Training time estimates
      - Infrastructure needs
  - Self-supervised learnings: predict relations between parts of input, e.g.
    - Directly predict one part from the other (e.g. language modeling)
    - Predict if different transformations belong to the same input (e.g. contrastive learning)
    - Line between unsupervised and self-supervised is still somewhat blurry (e.g. autoencoders)
- Encoder-only models: e.g. masked language models
  - No autoregressive decoding component
  - Encoder: Transformer, CNN, LSTM, RNN
  - Entire input is processed before solving the task
  - Appropriate for tasks like classification
  - E.g., BERT, RoBERTa, ALBERT
  - Masked Language Modelling: randomly mask p% of the input tokens, replace 10% of the masked tokens with the correct word, replace another 10% of tokens with random word; introduced in BERT paper
  - BERT: MLM only
  - ELECTRA: generator (replaces tokens, trained via MLM) and discriminator (detects fake tokens), more compute-efficient than standard MLM
  - Sentence-Pair Pretraining Tasks
- Sequence-to-Sequence Models: e.g. denoising autoencoders
  - Encoder-decoder models:
    - Autoregressive decoding component, typically Transformer
    - Entire input is processed before decoding
    - Appropriate for seq2seq generation tasks
    - E.g., BART, T5
- Decoder-only Models: e.g. language models (see previous section on language models)
  - Scale matters:
    - In general: innovation around scaling efficiently
    - Scaling laws: how to scale model size vs data size?
    - Data quality: how to select the right data for the downstream task?
    - Efficient attention: how to make long contexts manageable?
- Finetuning
  - Finetuning vs. Feature Extraction
    - Adapting for classification: for each task, take initial model $\theta_\text{model}$ and add new classification head $\theta_\text{classifier}$
    - Feature extraction: keep $\theta_\text{model}$ fixed, only update $\theta_\text{classifier}$, i.e. $\nabla J(\theta_\text{classifier})$ with $\theta_\text{model}$ fixed.
    - Finetuning: update both pre-trained model weights $\theta_\text{model}, \theta_\text{classifier}$: $\nabla J(\theta_\text{model}, \theta_\text{classifier})$.
  - Finetuning often performs better, but...
    - takes more memory and compute during training.
    - needs a separate copy of the model for every task.
    - is sensitive to hyperparameters.
    - may suffer from catastrophic forgetting.
  - Parameter-Efficient Finetuning: adapt pre-trained models to new tasks without significant expansion in parameter size
    - Leverage knowledge from pre-training while avoiding overfitting on smaller downstream tasks
    - Methods:
      - Additive: Add a small amount of additional parameters while freezing original model
      - Selective: Select few parameters from the original model
      - Reparametrization-based: Leverage low-rank representations for parameters
    - Adapter modules in Transformers: add a set of small additional modules per task, freeze the rest of the model.
      - Adapter: small additional MLP in each Transformer layer.
      - Skip-connection in adapter for near-identity initialisation.
      - During finetuning: train adapter, layer-norm, classification layers
      - _Performance is considerably worse_ with PEFT with adapters, but in some cases it is the only option.
    - Low-Rank Adaptation (LoRA): reparametrise large weight matrices for efficient adaptation.
      - The original weights $W_0 \in \mathbb{R}^{d \times k}$ are frozen and remain unchanged
      - Two small matrices $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$ where $r \ll \min(d, k)$. $A$ initialised randomly, $B$ initialised at zero to cause no change initially.
      - Low-rank update: $\Delta W = BA$
      - Forward pass: $h = W_0 x + \Delta W x = W_0 x + BA x$
      - Fewer trainable parameters = memory and compute efficiency.
      - Advantages:
        - No additional latency: forward pass $h = W_0 x + \Delta W x = W_0 x + BA x$ at training time, $h = (W_0 + \Delta W) x$ at inference time
        - LoRA can approximate full finetuning: pre-trained LMs have a low intrinsic dimension, by setting $r$ to the rank of pre-trained weight matrices one can recover full finetuning
- Prompting: generating solutions to tasks by conditioning a language model on specifically crafted input text:
  - In-context learning
    - Zero-shot learning: model predicts the answer given only a natural language description of the task, no gradient updates are performed
    - One-shot learning: task description and a single example of the task (e.g. sea otter => loutre de mer for EN->FR translation), no gradient updates are performed
    - Few-shot learning: task description and a _few examples_, no gradient updates are performed
    - _Without any finetuning, Large Language Models can achieve strong results from zero or few examples_!
  - Chain-of-Thought (CoT) prompting: make the model think _step by step_
    - Greatly improves performance! Can solve problems that otherwise gets wrong
    - can be zero-shot, i.e. no examples, or may be one or few-shot - note: examples then are not examples of input-output but step-by-step reasoning examples

Summary:

- Foundation models are pretrained models that can be adapted to a wide range of tasks
- Finetuning foundation models improves model performance greatly.
- Large language models enable zero- and few-shot learning.
- Foundation models improve predictably with scale.
- Three ways for improving prompts: context, few-shot (examples), chain-of-thought (force model to split complex tasks into smaller steps, ideally by providing examples with reasoning steps BEFORE output example)

Open problems:

- Will scaling "work" forever? Do we run out of data?
- Grounding agents in environments via multi-modal inputs
- Hallucinations, retrieval-augmented LLMs
- Planning
- Tokenization is problematic for some tasks (e.g., counting characters, words, reversing strings)

## Recognition and Induction of Tree Structures

- Syntactic parsing: recognising the syntactic structure of a sentence
  - Challenges:
    - structural ambiguity
- Constituent parsing: use of a context-free grammar (CFG) for parsing using terminal and non-terminal symbols
  - A CFG is a type of formal grammar and is defined as a tuple $G = \{ S, N, \Sigma, R \}$ with $S$ = start symbol, $N$ = finite set of non-terminal symbols, $\Sigma$ = finite set of terminal symbols, $R$ = representing the production rules in Chomsky normal form: $A \rarr B \, C, \, A \rarr w$ where $A, B, C$ are non-terminals ($\in N$), $w$ is a terminal symbol ($\in \Sigma$).
  - A probabilistic CFG extends a CFG by assigning a probability $\pi_r$ to each production rule $r \rarr R$, such that for each non-terminal $A$ the sum of the probabilities of all rules that derive from $A$ must be $1$: $\sum_{r: A \rarr \gamma}{\pi_r} = 1$, where $\gamma$ represents the right-hand side of $r$.
  - Efficient decoding: (probabilistic) CKY algorithm
    - to find possible parses given a CFG
    - to find the highest scoring parse given a PCFG
    - is a DP-based approach to parsing
    - requires conversion of the grammar to CNF
  - Lexicalization and tag splitting
  - Tree recursive neural networks
  - Self-supervised constituent grammar induction
- Dependency parsing:
  - A dependency tree G satisfies the following constraints:
    1. There is a single designated root node that has no incoming arc
    2. Except for the root node, each vertex has exactly one incoming arc
    3. There is a unique path from the root node to each vertex $V$ in $G$
  - Graph based dependency parsing
    - Inference or prediction: search for the maximum spanning tree
  - Transition based dependency parsing
- Evaluation
  - Accuracy
  - Labelled recall
  - Labelled precision
  - Unlabeled attachment score (UAS): recognition of head
  - Labeled attachment score (LAS): recognition of head and label

## Recognition of Complex Graph Structures

- Semantic role labelling: recognising the basic event structure of a sentence, i.e. who does what to whom / what when where how?
  - Determining which constituents in a sentence act as predicate or as semantic arguments to a predicate: segmentation
  - Determining the role for each of those components of the sentence: classification
  - Sequential modelling for SRL: CRF, RNN
  - GCNs: neural network that given a graph computes the representation of a node conditioned on the neighboring nodes
    - Can be seen as a message passing algorithm where the representation of a node is updated based on "messages" sent by neighboring nodes
    - During training representations or embeddings of nodes are learned
    - Node embeddings can be aggregated to form graph embeddings
    - Add structural information to the GCN [Marcheggiani & Tivo EMNLP 2020]
  - Evaluation
- Abstract meaning representation (AMR) parsing
  - Graph-based parsing
  - Iterative inference
  - Evaluation
- AMR for text generation
  - Graph Transformers: Incorporates structure-aware self-attention encoding
  - Evaluation

## Semantic Parsing and Solving Math Word Problems

- Traditional rule-based approaches
  - Define possible syntactic structures using a context-free grammar
  - Construct semantics bottom-up, following syntactic structure and manually drafted rules
- ML approaches
  - Learning to map to a semantic knowledge representation
  - Learning to directly map to a denotation
  - Neural network models:
    - Sequence-to-sequence models
    - Sequence-to-tree models: tree-based decoding
- Solving math word problems: Most successful approaches usually consist of two main steps:
  1. Parsing text and diagram by generating logical expressions to represent the key information of the text and diagram as well as the confidence scores
  2. Addressing the optimization problem by aligning the satisfiability of the derived logical expressions in a numerical method that requires manually defining indicator functions for each predicate
  - The models rely on carefully designed prompts or chain-of-thought (CoT) prompting that break down the natural language description in simpler parts
- Evaluation: Current models struggle with obtaining correct results
  - When solving simple arithmetic word problems, most existing methods often use shallow heuristics relying on superficial patterns, while even ignoring the question
  - Questions with numerous and varied mathematical operations using rare numerical tokens are difficult to solve
  - Lengthy questions with low readability scores and those requiring real-world knowledge are seldom correctly solved
  - CoT prompting of LLMs for solving math word problems often leads to mistakes at early steps of the solution, resulting in incorrect final answers
  - There is a lack robust generalization across various datasets, grade levels, and types of math problems, inconsistent performance of LLMs when faced with questions presented in a mixture of words and numbers, different final answers generated by the LLM in multiple trials, and vulnerability to adversarial attacks
- Reflection: Neural approaches are very valuable for semantic parsing but when they avoid linguistic features and grammar rules, they need many training examples to learn a valuable model, especially in open domains:
  - Large LLMs have seen lots and lots of training examples
  - But compositional feature representations remain an open problem

## Discourse Analysis

- Discourse = collocated, structured coherent groups of sentences
  - Monologue
  - Human-human dialogue
  - Human-computer dialogue
- Noun phrase coreference resolution:
  - Rule-based model (Hobbs algorithm)
  - Mention detection
  - Mention pair and mention ranking models
  - Inference with integer linear programming as optimization step
    - Linear programming (LP) = collection of techniques and methods for finding an assignment for a set of variables so that the linear objective function composed of these variables is optimal, i.e., by maximizing or minimizing the linear function, while subject to linear equality or inequality constraints
    - Integer linear programming (ILP) = variant of LP : variables are restricted to only have integer values
  - Neural end-to-end coreference resolution
  - Leveraging LLMs
  - Evaluation
- Recognition of rhetorical discourse structure:
  - Introduction to rhetorical structure theory and argumentation mining

## Spatial and Temporal Representation Learning and Reasoning

- Temporal information extraction
  - Introducing the use case of temporal information extraction
  - Timex recognition and normalization
  - Temporal relation extraction and integration of temporal reasoning
  - Direct prediction of a timeline and its loss functions
- Spatial information extraction
  - Introducing the use case of spatial information extraction
  - Spatial relation extraction and integration of spatial reasoning
  - Direct prediction of spatial coordinates and its loss functions
- Spatiotemporal information extraction

## Machine Reading Comprehension

- Machine reading comprehension: a testbed for evaluating natural language understanding (NLU)
  - Given a text or a paragraph (often called context) $P$ that consists of a sequence of words, $p_1, \dots, p_n$, and a question $Q$ that consists of a sequence of words $q_1, \dots, q_m$, the goal is to retrieve the answer $A$.
- Models for machine reading comprehension (MRC) and their components:
  1. Representation of input question and questioned documents (here text): Building a feature representation of the question and the text
     - RNN encoding: word level encoding
  2. Question-(con)text interaction: Attention mechanisms play here an important role and emphasize which parts of the text are more important to answer the question:
     - Unidirectional question-to-text attention
     - Bidirectional question-to-text and text-to-question attention
     - One-hop interaction = interaction between question and (con)text is computed only once
     - Multi-hop reasoning or interaction = interaction between question and (con)text is computed more than once mimicking the rereading process of humans (cf. the iterative inference seen in the Chapter on the Recognition of complex graphs)
  3. Answer prediction
     - Multiple-choice test: choice of most candidate answer that is most similar to the attentive (con)text representation, where similarities can be computed in different ways (e.g., cosine similarity) and scores are possibly combined
     - Cloze test: word predictor based on attentive (con)text representation
     - Span extractor: predicts start and end position in (con)text
     - Answer generation: cf. decoding in sequence-to-sequence models by selecting token with the highest probability until an end token is predicted
- Memory MRC networks:
  - RNNs (e.g., LSTMs): their memory (encoded by hidden states and weights) is typically too small, and is not compartmentalized enough to accurately remember facts from the past (knowledge is compressed into dense vectors)
  - Instead of a single memory cell we use a dedicated memory: Memory storage is explicitly segregated from the computation of the neural network [Weston et al. ICLR 2015]
    - Store/write long-term and short-term context into a memory
    - Iteratively reads from the memory (i.e., multiple hops) relevant information to answer the question
    - Memory addressing is guided by an attention mechanism usually performed by a controller
    - Combine the successful machine learning strategies for inference with a memory component that can be read and written to: the model is trained to learn how to operate effectively with the memory component
  - A memory network consists of a memory $m$ (an array of objects $m_i$) and four (potentially learned) components $I$, $G$, $O$, $R$: the input feature map, generatlisation, output feature map, response.
  - The **KV-MemNN** performs QA by first storing facts in a key-value structured memory before reasoning with them in order to predict an answer:
    - Keys address relevant memories with respect to the question
    - Corresponding values are subsequently returned
    - This structure allows the model to encode prior knowledge for the considered task as well as representations of the (con)text, and to leverage possibly complex transforms between keys and values, while still being trained using standard back-propagation via stochastic gradient descent
- **MAMBA**: linear-time sequence modeling with selective state spaces [Gu & Dao COLM 2024]
  - Selective state space model: to effectively select data (i.e., focus on or ignore particular inputs)
  - Linear RNN: to boost efficiency by a parallel scan algorithm for computing the recurrence
  - 5X generation throughput compared to a transformer architecture of a similar size
  - Discretisation: transforms continuous parameters ($\Delta, A, B$) to discrete parameters ($\bar{A}, \bar{B}$) through a discretisation rule:
    $$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B$$
- Structured state space sequence model (S4):
  - $A \in \mathbb{R}^{N \times N}$, with $N$ the dimension of $h_t$m is the matrix that captures information from the previous state $h_{t-1}$ to build the new state $h_t$.
  - $B \in \mathbb{R}^{N \times 1}$m with $1$ the dimension of the input representation $x_t$, allows for fine-grained control on what to let of $x_t$ into the state $h_t$.
  - $C \in \mathbb{R}^{1 \times N}$, with $1$ the dimension of the output representation $y_t$, allows for fine-grained control on what to let of $h_t$ into the output $y_t$.
  - The scalar $\Delta$ controls per dimension of $x_t$ how much to focus on or ignore from the current input $x_t$:
    - A large $\Delta$ focuses strongly on $x_t$ and basically resets the state $h_t$.
    - A small $\Delta$ largely ignores $x_t$.
- SSM + Selection (S6): selective state space model with time-varying parameters:
  - The parameters $B, C, \Delta$ that affect interaction along the input sequence are now input-dependent
  - The parameters $B, C, \Delta$ have an extra dimension $L$ = the sequence length: different parameters per time step
  - The model very selectively forgets or stores information based on the input it sees
- RNN vs. transformer:
  - RNN:
    - Theoretically deals with infinite context window
    - Computation and memory cost of order size $\mathcal{O}(n)$
    - Training is not parallelizable, but inference is done in constant time for each token: $\mathcal{O}(1)$
  - Transformer:
    - Transformers are computationally expensive when processing long sequences: $\mathcal{O}(n^2)$
    - But have the advantage of hardware-aware parallel processing
- Can we combine the advantages of both architectures in one model?
  - **Linear RNN** [Orvieto et al. DeepMind 2023]: Use linear recurrences in the RNN, i.e., remove the non-linearities in the recurrence:
    $$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$
  - How to parallelize this recurrent computation? **Parallel scan operation**:
    - The recurrent computation of the linear RNN can be thought of as a scan operation, in which each state is the sum of the previous state and the current input.
    - Any binary operation can be executed in parallel while synchronized at each step under condition that the binary operation is associative.
      - E.g., at each iteration step: compute sums of previous steps: sum the outputs of 2 steps, of 4 steps, of 8 steps, etc. => $\mathcal{O}(\log{n})$
      - The time complexity instead of being 𝑂(𝑛) is reduced to 𝑂(𝑛/𝑇) with 𝑇 the number of parallel threads
      - The recurrence $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ can be unrolled easily:
        $$h_0 = \bar{B} x_0$$
        $$h_1 = \bar{A} h_{0} + \bar{B} x_1 = \bar{A} \bar{B} x_0 + \bar{B} x_1$$
        $$h_t = \sum_{k=0}^{t-1}{\bar{A}^k \bar{B} x_{t-k}}$$
      - We can paralellize these operations on different processors. We can separately store the matrix powers and the computations related to $h_t$ resulting in the unsweep operation.
  - Efficient and effective computation requires imposing structure on the $A$ matrix:
    - The repeated computation of matrix multiplications is inefficient and may lead to numerical issues as $k$ increases (e.g., overflow)
    - Diagonal linear RNN layers allow for the parallelizable of the recurrence to substantially improve training speeds : selective state space models use complex numbers in state 𝒉 allowing for diagonalization of 𝑨 [Orvieto et al. DeepMind 2023]
    - HiPPO initialization of matrix 𝑨 allows to effectively capture long-range dependencies
- Reducing the computational complexity of SSMs:
  - Sequential nature of the recurrence computation is avoided by a parallel scan algorithm:
    - Results in 𝑂(𝑛 log 𝑛) time complexity for training
    - Unrolling the model autoregressively during inference requires only constant time per step as it does not require a cache of previous elements
  - To reduce memory usage:
    - Instead of preparing the scan input ($\bar{A}, \bar{B}$) in GPU HBM (high-bandwidth memory) the SSM parameters (Δ, 𝑨, 𝑩, 𝑪) are loaded directly from slow HBM to fast SRAM, discretization and recurrence are computed in SRAM, and their outputs are written back to HBM
    - Intermediate states needed for backpropagation are not stored but recomputed in the backward pass when the inputs are loaded from HBM to SRAM
- Evaluation
  - Exact match (EM) or accuracy
  - F1
  - BLEU-N
  - ROUGE-N, ROUGE-L
- Reflection on NLU
  - Most of the questions are already correctly answered by the models, do not require grammatical and complex reasoning and do not require understanding the discourse structure of the text
  - Surprisingly, questions are often answered correctly even when in the text read by the machine the content words are shuffled
  - Some of the questions in the datasets do not require multi-hop reasoning and can be answered by exploiting statistical shortcuts
  - Lexical overlap and word order might be sufficient to answer the questions

## Machine Translation and Multilingual Large Language Models

- Machine translation: automating translation of a text $x$ from a source language to a text $y$ in a target language
  - Machine translation models:
    - Neural machine translation (NMT): estimate $P(y \mid x)$ with a neural network $P(y \mid x; \theta)$ where $x = (x_1, x_2, \dots, x_l)$ is the source sentence (or text) and $y = (y_1, y_2, \dots, y_j)$ is the target sentence (or text), i.e. $P(y \mid x) = P(y_1 \mid x) P(y_2 \mid x, y_1) \dots P(y_l \mid x, y_1, \dots, y_{l-1})$ by the chain rule
    - Encoder-decoder models: encoder maps a source language sequence $x$ to one or more vectors, decoder predicts target language sequence $y$ symbol by symbol using the source sequence vector representations and previously predicted symbol
      - RNN-based architecture: encoder encodes the input sequence with the hidden state of the last RNN cell, decoder takes the last encoder hidden state as the initial decoder hidden state and a special start token to generate the target sequence
      - Transformer architecture: encoder encodes _each word_ of the input sequence with the hidden state of the corresponding RNN cell, decoder uses _attention to compute a context vector at each timestep_ which is used together with the embedding of the previously generated symbol to generate the next symbol
        - Note: Attention does not inherently take into account sequence order - to keep information about the sequence order, it is added to the input vectors themselves: _positional encoding_ (not necessary in RNNs b/c learn information about sequence order automatically)
      - CNNs: CNN encoder, CNN decoder with attention mechanism
      - Unsupervised NMT: trained by minimising _denoising auto-encoding loss_, _cross-domain loss_,
  - Decoder-only models: simultaneous translation
    - The model dynamically reads source sentence $x = (x_1, \dots, x_j)$ and generates translation $y = (y_1, \dots, y_l)$.
    - Training is based on a negative log likelihood loss $\mathcal{L}_\text{SiMT} = - \sum_{i=1}^I{ \log{P(y_i \mid x_{\leq g_i}, y_{< i})} }$ where $g_i$ represents the number of source tokens involved in predicting $y_i$.
  - Decoding
    - Maximisation-based
    - Sampling-based: random sampling with temperature (low / high temperature = low / high randomness, i.e. in selecting a wider variety of words), top-k sampling
  - Evaluation:
    - Human evaluation: good but costly and sometimes subjective
    - Syntactic-based metrics: BLEU, ROUGE, ROUGE-L, METEOR
    - Semantic-based metrics: COMET, BERTScore
- NMT best performing, but still some challenges:
  - biases in training data
  - domain mismatch between train and test data
  - maintaining context over long text
  - low-resource language pairs, chat language
  - sparse data: words that occur once or twice have unreliable statistics
- Multilingual large language models and evaluation

## Conversational Dialogue Systems and Chatbots

Suppose a user poses question or statement 𝑥: we want to find the best answer or
question 𝑦, given 𝑥 : $\argmax_y{P(y \mid x)}$

𝑥 can refer to one statement or full dialogue history. Goes on iteratively until user is satisfied or system can execute action. Conversational agents and dialogue systems combine language understanding and
language generation, which are ultimate goals of natural language processing!

- Task oriented dialogue agents: Rule based (e.g. ELIZA in 1966) versus neural based approaches
- Chatbots: End-to-end sequence-to-sequence neural models
- Evaluation and challenges

## References

Jurafsky, D., & Martin, J. H. (2024). Speech and language processing (3rd ed., draft). Retrieved from <https://web.stanford.edu/~jurafsky/slp3/>

Malan, D., & Yu, B. (2024). CS50's Introduction to Artificial Intelligence with Python [Course materials]. Harvard OpenCourseWare. Retrieved from <https://cs50.harvard.edu/ai/2024/notes/6/>
