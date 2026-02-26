# Low-Resource-NLP-Approach-to-Analyzing-Reddit-Discussions-on-Sri-Lankan-Education

This document outlines an Natural Language Processing (NLP) assignment, focusing on data collection, cleaning, tokenization, vectorization, clustering, and classification of Reddit posts and comments from the "r/srilanka" subreddit.

**Task 1: Data Collection and EDA**

  * Data was collected from the "srilanka" subreddit using the Reddit API (PRAW).
  * The fetch included posts and their top-level comments across categories like "top," "controversial," "hot," "new," and "rising".
  * Data enrichment involved extracting image captions using the BLIP model and performing OCR on images using `pytesseract` to extract text.
  * Initial data had 5,049 unique posts, which was reduced to 2,345 after content-based deduplication and further reduced to 2,337 after semantic deduplication using Sentence Transformer (all-MiniLM-L6-v2) with a cosine similarity threshold of 0.85.
  * The final dataset, merging posts and comments, contained 124,874 raw entries, which were cleaned to 66,705 unique entries with 2,247 unique post titles.
  * The average word count for the merged content was approximately 172.25 words, with a median of 104.0 words.

**Task 2: Data Pre-processing and Tokenization**

  * Cleaning steps included replacing emojis with sentiment tags (e.g., `EMO_POS`), removing URLs, Reddit-specific tags, and non-ASCII characters.
  * Posts were filtered to have a word count between 25 and 1,000 words, resulting in 46,651 unique entries and 1,734 unique posts.
  * Tokenization schemes were compared:
      * **Traditional (Word-based):** Included lemmatization and stopword removal. It had the smallest average sequence length (\~86 tokens) and a vocabulary size of 30,781.
      * **Byte Pair Encoding (BPE):** Had a longer average sequence length (\~202 tokens) and the largest vocabulary (31,952).
      * **WordPiece:** Had a slightly longer average sequence length (\~207 tokens) but the smallest vocabulary (19,425).
  * Perplexity for an N-gram model (N=3) on all tokenization schemes was "Infinite," indicating high linguistic diversity and justifying the need for Transformer models.

**Task 3: Vector Representation and Classification**

  * **Vectorization** included sparse (Bag-of-Words and TF-IDF) and static dense (Word2Vec and Doc2Vec) representations. Both sparse matrices had a shape of (46651, 30781), while dense vectors had a shape of (46651, 100).
  * **Clustering/Topic Modeling** was performed using K-Means and Latent Dirichlet Allocation (LDA) on the TF-IDF matrix. K-Means with an optimal K=6 was selected, as LDA suffered from Topic Collapse.
  * The 6 K-Means clusters were mapped to categories: Politics, Economy, Tourism & Living, Social Issues, Education & Work, and Culture. Cluster 4 was identified as the "Education" cluster.
  * **Classification Comparison** used five models on all four vector types, employing GroupShuffleSplit to prevent data leakage.
  * **Results:** Logistic Regression with Sparse (BoW) features was the top performer with 86.00% accuracy. Sparse representations consistently outperformed dense embeddings, suggesting classification is keyword-driven.
  * **Deep Learning Classification** preparation involved creating `TensorDataset` and `DataLoader` for a Multi-Layer Perceptron (MLP) on Doc2Vec and an LSTM/CNN on padded token sequences.


**Key Findings and Learning Process (Part A):**

  * **Data Leakage:** An initial train\_test\_split caused data leakage, leading to a deceptive $\\sim$98% accuracy. This was fixed using GroupShuffleSplit, dropping the accuracy to $\\sim$86%, which was the highest overall accuracy.
  * **Tokenization:** Traditional (NLTK), BPE, and WordPiece tokenizers yielded infinite perplexity due to too many unique/out-of-vocabulary words on Reddit data, teaching the reliance on Transformers over N-gram models.
  * **Model Performance:** Logistic Regression (86.00%) with Sparse (Bag-of-Words) features outperformed fine-tuned BERT (84.79%) because the ground truth labels came from K-Means on TF-IDF, which are based on keywords rather than deep meanings.
  * **Generative AI:** Zero-Shot and Few-Shot classification with TinyLlama, GPT-2, and Phi-2 performed very poorly ($\\leq$ 20% accuracy), showing generative models are good at writing text, not classifying it.
  * **Knowledge Distillation:** A non-distilled RoBERTa teacher model (68% accuracy) was used to train a small Logistic Regression student model, which achieved 70.00% accuracy, demonstrating the power of distillation for smart, lightweight models.

**Future Improvements for Strengthening Democracy (Part B):**

  * **Technical:** Implement Aspect-Based Sentiment Analysis (ABSA) for mixed opinions, use LoRA tuning on a larger model like Llama-3-8B to reduce hallucination and improve summarization, and add Multilingual and Code-Switched Support (Singlish and Tanglish).
  * **Policymakers:** Implement a real-time dashboard, require human checking to verify summaries, and ensure demographic balance by adding data from multiple platforms and comparing AI results with traditional surveys to avoid Reddit bias.
