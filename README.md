# NLU Assignment 2 — Word Embeddings & Character-Level Name Generation

## 📌 Project Overview

This assignment is divided into two parts:

- **Part 1**: Build a domain-specific corpus by web-scraping IIT Jodhpur's website, then train **Word2Vec** models (CBOW and Skip-gram with Negative Sampling) from scratch using NumPy.
- **Part 2**: Train three **character-level language models** (Vanilla RNN, Bidirectional LSTM, RNN with Attention) in PyTorch to generate Indian names.

---

## 📁 Repository Structure

```
.
├── NLU2_P1.ipynb          # Part 1: Corpus Building + Word2Vec
├── NLU2_P2.ipynb          # Part 2: Character-Level Name Generation
└── README.md
```

---

## 🔵 Part 1 — Domain-Specific Word2Vec

### 📊 Dataset / Corpus

The corpus was built by scraping 21 pages from [IIT Jodhpur's website](https://www.iitj.ac.in), covering:
- Main / About pages
- Academic departments (CSE, EE, ME, Chemical, Civil, etc.)
- Programmes (B.Tech, M.Tech, Ph.D, M.Sc)
- Research and student regulations

**Corpus Statistics:**

| Metric | Value |
|--------|-------|
| Total Documents | 21 |
| Total Tokens | 12,782 |
| Vocabulary Size (raw) | 2,297 |
| Vocabulary Size (min_count=2) | 1,307 |

### ⚙️ Methodology

**1️⃣ Web Scraping & Preprocessing**
- Fetched pages using `requests` + `BeautifulSoup`
- Removed scripts, styles, boilerplate, and navigation elements
- Lowercased, tokenized, lemmatized using NLTK
- Applied custom stopword list (domain-specific noise like `iit`, `jodhpur`, `click`, `nav`, etc.)
- Applied term normalization (e.g., `b.tech` → `btech`)

**2️⃣ Word2Vec Models (from scratch — NumPy only)**
- **CBOW** (Continuous Bag of Words): predicts center word from context average
- **Skip-gram**: predicts context words from center word
- Both trained with **Negative Sampling** (Mikolov et al.)
- Negative samples drawn proportional to `freq(w)^0.75`

**3️⃣ Hyperparameter Grid**

| Config | Dimensions | Window | Neg Samples |
|--------|-----------|--------|-------------|
| 1 | 50 | 3 | 5 |
| 2 | 100 | 5 | 5 |
| 3 | 100 | 5 | 10 |
| 4 | 200 | 7 | 10 |

- Epochs: 20 | Learning Rate: 0.025

**4️⃣ Evaluation**
- **Nearest neighbors** (cosine similarity) for query words: `research`, `student`, `phd`, `exam`
- **Analogy experiments** (e.g., `undergraduate : btech :: postgraduate : ?`)
- **t-SNE / PCA** visualizations of word clusters (Degree Programmes, Research, Departments, etc.)

---

## 🔴 Part 2 — Character-Level Name Generation

### 📊 Dataset

- **1,000 Indian names** generated via LLM (as per assignment instructions)
- Covers male, female, and gender-neutral names from North Indian Hindu, South Indian, Muslim, Sikh, and other regional traditions
- Lowercased; character-level vocabulary of **39 tokens** (26 letters + digits + special tokens)

**Special tokens:**

| Token | Index |
|-------|-------|
| `<PAD>` | 0 |
| `<SOS>` | 1 |
| `<EOS>` | 2 |

### ⚙️ Methodology

Each name is modeled as a character sequence. Models receive `<SOS> + name` as input and predict `name + <EOS>` as target.

**Models Implemented:**

### 1️⃣ Vanilla RNN

| Hyperparameter | Value |
|---------------|-------|
| Embedding Dim | 64 |
| Hidden Size | 256 |
| Num Layers | 2 |
| Dropout | 0.3 |
| Trainable Params | 226,535 |

### 2️⃣ Bidirectional LSTM (BLSTM)

| Hyperparameter | Value |
|---------------|-------|
| Embedding Dim | 32 |
| Hidden Size | 64 × 2 (bidir) |
| Dropout | 0.5 |
| Trainable Params | 56,455 |

### 3️⃣ RNN with Attention (Bahdanau-style)

| Hyperparameter | Value |
|---------------|-------|
| Embedding Dim | 128 |
| Hidden Size | 512 |
| Num Layers | 2 |
| Dropout | 0.3 |
| Trainable Params | 1,162,151 |

**Training:**
- Optimizer: Adam with `ReduceLROnPlateau` scheduler
- Loss: Cross-Entropy (ignoring `<PAD>`)
- Epochs: 60 (RNN), 60 (BLSTM, early stop), 150 (Attention, early stop at loss ≤ 0.20)

### 📈 Results

| Model | Final Loss | Novelty Rate | Diversity |
|-------|-----------|-------------|-----------|
| Vanilla RNN | 1.2807 | 61.0% | 90.5% |
| BLSTM | ~0.20 (early stop) | 100% | 94% |
| RNN + Attention | ~0.20 (early stop) | 100% | 74% |

- **Novelty Rate**: % of generated names not seen in the training set (measures generalization vs. memorization)
- **Diversity**: unique generated names / total generated names (measures output variety)
- Evaluated over **200 generated names** per model at temperature `0.8`

### 🔍 Qualitative Observations

- **Vanilla RNN** generates plausible-sounding names (`Banila`, `Prathik`, `Aryansh`, `Ritavati`) but occasionally memorizes training names (39% overlap).
- **BLSTM** achieves 100% novelty but early outputs show repetitive character patterns (`Jajapypyr`, `Ukukshsh`) — indicating the small model learns n-gram patterns.
- **RNN + Attention** also achieves 100% novelty but initially struggles with character distribution before converging.

---

## 🚀 How to Run

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2. Install Dependencies

**Part 1 (NumPy-based Word2Vec):**
```bash
pip install requests beautifulsoup4 wordcloud matplotlib scikit-learn nltk
```

**Part 2 (PyTorch Name Generation):**
```bash
pip install torch torchvision torchaudio
```

### 3. Run Notebooks

Open and run the notebooks in order:

```bash
jupyter notebook NLU2_P1.ipynb   # Part 1: Word2Vec
jupyter notebook NLU2_P2.ipynb   # Part 2: Name Generation
```

> **Note**: Part 1 scrapes IIT Jodhpur's live website. An internet connection is required. Part 2 benefits from a GPU (CUDA) but runs on CPU as well.

---

## 🛠️ Tech Stack

| Library | Usage |
|---------|-------|
| `numpy` | Word2Vec implementation from scratch |
| `requests` + `beautifulsoup4` | Web scraping |
| `nltk` | Tokenization, stopwords, lemmatization |
| `wordcloud` | Corpus visualization |
| `scikit-learn` | PCA, t-SNE for embedding visualization |
| `torch` (PyTorch) | RNN, BLSTM, Attention model training |
| `matplotlib` | Plots and figures |

---

## 👤 Author

Aditya Padhy - B22CS103

