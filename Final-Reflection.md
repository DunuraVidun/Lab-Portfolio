# Final Reflection -- PUSL3205 Large Language Models and Generative AI

**Student Name:** Senarathge Senarath\
**Student ID:** 10965304\
**Module:** PUSL3205 -- Large Language Models and Generative AI

## 1. Introduction

Throughout the PUSL3205 laboratory sessions, I developed a practical
understanding of how machine learning, natural language processing
(NLP), transformer-based language models, fine-tuning,
Retrieval-Augmented Generation (RAG), and tool-using AI agents can be
applied to different problems.

The laboratories helped me progress from building a traditional
text-classification model to exploring how language models can be
trained, adapted, connected to external knowledge, and integrated with
tools. They also showed me that implementing an AI system is not only
about making a model produce an answer. It is equally important to
evaluate the answer, understand the system's limitations, protect user
information, and provide evidence that the implementation works.

## 2. Reflection on the Laboratory Activities

### [Lab 01 -- Machine Learning and Sentiment Analysis](./Lab-01/README.md)

In Lab 01, I implemented a sentiment classification system using the
IMDB portion of the Sentiment Labelled Sentences dataset. I performed
text preprocessing, explored word frequencies, converted text into
numerical features using Bag of Words, and trained a Logistic Regression
classifier.

The model achieved **71.33% accuracy** on the test set. Evaluating it
with precision, recall, F1-score and a confusion matrix helped me
understand why accuracy alone does not fully describe classification
performance. Testing new sentences also showed that mixed opinions and
contextual expressions can be difficult for a basic Bag of Words model
to interpret.

This laboratory gave me a foundation in the NLP workflow: preparing
text, extracting features, training a model and evaluating predictions.
It also introduced an important limitation of traditional feature-based
approaches: word counts do not adequately represent word order or
context.

### [Lab 02 -- LLMs in Everyday Life: FocusMate](./Lab-02/README.md)

In Lab 02, my group designed **FocusMate**, an AI-powered student focus
management system for the problem of student distraction. The proposed
concept combined a mobile application, a smart headset and an
eye-tracking camera. We explored how NLP could prioritise notifications,
computer vision could estimate gaze and eye-closure patterns, machine
learning could learn individual study habits, recommendation methods
could suggest interventions, and an optional LLM assistant could support
learning.

This activity helped me think about AI from a product and user
perspective rather than only from a model-development perspective. I
learned to map different AI techniques to specific user needs and
consider how they might work together in one system.

An important outcome was recognising that the proposed system was a
design concept, not a completed or experimentally validated application.
We also identified risks such as camera privacy, notification
misclassification, inaccurate attention estimates, hardware cost and
accessibility. This reinforced the importance of user consent, user
control and honest communication about what AI measurements can and
cannot establish.

### [Lab 03 -- Developing a Tiny Transformer Language Model](./Lab-03/README.md)

In Lab 03, I implemented a small decoder-style, character-level
Transformer language model from scratch using PyTorch and a subset of
the TinyStories dataset. I worked with character tokenisation, token and
positional embeddings, causal masked self-attention, multi-head
attention, feed-forward layers, residual connections and layer
normalisation.

The main model was trained for next-character prediction. Its validation
loss reached **0.7600**, with a validation perplexity of **2.138** and
validation bits per character of **1.097**. I also tested configurations
with different numbers of layers, attention heads and training
durations, and explored text generation using temperature and top-k
sampling.

This laboratory helped me understand the internal components of a
Transformer more clearly than studying the architecture only at a
conceptual level. Implementing causal attention and the training loop
showed me how the model learns to predict the next token from preceding
context.

The generated stories also revealed limitations. Although the model
learned some common story patterns, it sometimes produced spelling
errors, repetition and incoherent events. I also learned that
experimental comparisons need to be interpreted carefully: the
configurations were single runs, and some experiments differed in more
than one training setting. Therefore, the results were useful for
exploration but not sufficient for broad statistical conclusions.

### [Lab 04 -- Fine-Tuning an LLM using LoRA](./Lab-04/README.md)

In Lab 04, I fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` for a fictional
computer-shop customer-support domain called TechZone Computers. I
created and validated a dataset of 50 question-and-answer pairs,
configured LoRA using PEFT, and performed supervised fine-tuning with
TRL's `SFTTrainer`.

The LoRA configuration made **4,399,104 parameters trainable**,
representing approximately **0.8826%** of the model parameters after
adapter application. This demonstrated how parameter-efficient
fine-tuning can adapt a pretrained model without updating all of its
weights.

I compared the original and fine-tuned models using the same five
held-out questions. The fine-tuned model achieved a higher ROUGE-L F1
score (**0.3586**, compared with **0.0924** for the original model), but
both models had an Exact Match score of zero. Manual evaluation showed
that the fine-tuned model improved on some domain-specific responses but
still generated incorrect details, including an unsupported cancellation
period and an incorrect laptop price and currency.

This was one of the most important lessons from the module: fine-tuning
does not guarantee factual accuracy. A model may produce responses that
sound more relevant or share more words with reference answers while
still giving incorrect information. I also learned why evaluation should
combine automatic metrics with manual inspection, particularly when
factual details matter. Standard LoRA was implemented in this practical;
QLoRA was considered but not implemented.

### [Lab 05 -- Retrieval-Augmented Generation (RAG) Pipeline](./Lab-05/README.md)

In Lab 05, I developed a document-based University AI Assistant. The
system accepts university PDF documents, extracts their text, divides
the content into chunks, creates embeddings, stores them in ChromaDB,
retrieves relevant passages, reranks candidates using a CrossEncoder,
and uses a locally hosted LLM through Ollama to generate answers.

The implementation helped me understand the full RAG workflow and the
roles of its components. In particular, I learned that the embedding
model, vector database, retrieval strategy, reranker, context
construction and answer-generation model all contribute to the final
response.

The practical tests showed that the assistant could answer some
questions using information from indexed university documents and
display retrieved source details. It also abstained on one tested
question when the requested information was not available in the
retrieved material. However, a question asking for available modules
received an incomplete response, and I identified limitations involving
retrieval coverage, the number of chunks passed to the model, scanned
PDFs without OCR, lack of conversational context and the absence of
automated citation verification.

This laboratory demonstrated that RAG can connect an LLM to external
documents, but it does not automatically guarantee complete or correct
answers. The quality of the retrieved evidence is especially important.
A response can be fluent and relevant while still omitting information
that exists elsewhere in the source document.

### [Lab 06 -- Building a Tool-Using AI Agent with Ollama](./Lab-06/README.md)

In Lab 06, I extended a locally hosted `llama3.2:3b` model into a
tool-using agent. The work covered four parts.

-   **Part A:** I connected the agent to calculator, date and time,
    Notepad, file search, file creation and system-information tools.
-   **Part B:** I implemented persistent memory using SQLite and tested
    storing and retrieving information, including after restarting the
    application.
-   **Part C:** I connected the agent to the RAG pipeline developed in
    Lab 05 so it could answer university-related questions using the
    existing knowledge base.
-   **Part D:** I integrated Google Calendar using OAuth 2.0 to retrieve
    schedules and create events after user confirmation.

This laboratory helped me understand the difference between an LLM that
only generates text and an agent that can select and invoke external
functions. I also learned that tool execution requires careful handling
of tool names, arguments, returned results and errors. The SQLite
exercise demonstrated persistence beyond a single run, while the
calendar integration introduced authentication and the need for user
approval before taking an external action.

The RAG integration also showed the benefit of reusing an existing
component rather than duplicating the whole pipeline. At the same time,
the tests confirmed that the retrieval limitations observed in Lab 05
can also affect an agent that uses that RAG system. The agent's ability
to call tools does not remove the need to validate the information those
tools return.

## 3. Overall Skills and Knowledge Gained

Across the six laboratories, I developed practical skills in several
connected areas:

-   **Machine learning and NLP:** preparing text, extracting numerical
    features, training classifiers and interpreting evaluation metrics.
-   **Transformer architectures:** understanding tokenisation,
    embeddings, causal self-attention, model training and text
    generation.
-   **Fine-tuning:** preparing domain-specific data, configuring LoRA
    adapters, using supervised fine-tuning tools and comparing model
    outputs.
-   **Retrieval-Augmented Generation:** processing documents, generating
    embeddings, storing vectors, retrieving and reranking passages, and
    grounding answers in external content.
-   **AI agents and tool use:** registering Python functions as tools,
    implementing tool execution workflows, and connecting an LLM to
    local and external services.
-   **Software development:** using Python libraries, local models,
    APIs, databases, notebooks, Streamlit and Git/GitHub to build and
    document practical systems.
-   **Testing and evaluation:** using quantitative metrics, qualitative
    inspection, test questions, screenshots and output verification to
    assess implementations.

The sequence of activities also helped me see how these techniques
relate to one another. Traditional machine learning can classify text,
Transformers learn language patterns, fine-tuning adapts a pretrained
model to a task, RAG supplies external knowledge at inference time, and
agents allow a model to interact with tools and services.

## 4. Challenges and Lessons Learned

The practical work involved challenges beyond the core AI concepts. I
encountered issues involving Python environments and dependencies, model
and library compatibility, dataset preparation, model-training
resources, document retrieval quality, API authentication and project
organisation.

Working through these issues taught me to investigate error messages,
verify the active environment, check installed dependencies, test
components separately and confirm that a change has actually worked. I
also learned the importance of keeping credentials and access tokens out
of a public GitHub repository.

A recurring lesson was that successful execution is not the same as a
successful AI outcome. A model can train without producing reliable
answers; a RAG system can retrieve passages that are relevant but
incomplete; and an agent can invoke a tool correctly while still relying
on imperfect information. For this reason, testing should examine both
whether the software runs and whether its outputs are appropriate for
the task.

## 5. Conclusion

Overall, the PUSL3205 practical sessions gave me a broad, hands-on
introduction to the development and evaluation of systems based on
machine learning and Large Language Models. I progressed from building a
sentiment classifier to implementing a Transformer, fine-tuning a
pretrained model, developing a RAG application and creating a tool-using
AI agent.

The most valuable lesson I gained is that building a useful AI system
requires more than selecting a model. Data quality, architecture,
retrieval, evaluation, software integration, security, user control and
responsible use all influence the final result. The limitations observed
throughout the laboratories have also helped me understand where further
testing and improvement are necessary.

These practical experiences have strengthened my technical foundation
and provided a clearer understanding of how LLM-based applications can
be designed, implemented, evaluated and improved.

## Related Laboratory Documentation

For practical implementation details, source code, screenshots and
laboratory-specific results, refer to the individual lab documentation:

- [Lab 01 – Machine Learning and Sentiment Analysis](./Lab-01/README.md)
- [Lab 02 – LLMs in Everyday Life: FocusMate](./Lab-02/README.md)
- [Lab 03 – Developing a Tiny Transformer Language Model](./Lab-03/README.md)
- [Lab 04 – Fine-Tuning an LLM using LoRA](./Lab-04/README.md)
- [Lab 05 – Retrieval-Augmented Generation (RAG) Pipeline](./Lab-05/README.md)
- [Lab 06 – Building a Tool-Using AI Agent with Ollama](./Lab-06/README.md)
