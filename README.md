# GA-vs-LLM-Notebooks
Master Dissertation Project files, Notebooks generating keyboards for English, French and Afrikaans

# AI-Driven Keyboard Layout Optimisation

## Overview
This project explores the use of **Genetic Algorithms (GA)** and **Large Language Models (LLMs)** to generate ergonomic keyboard layouts.  
The pipeline has been applied to **English, French, and Afrikaans**, enabling cross-language comparison.

- **GA** → Fine-grained optimisation for ergonomics and workload balance  
- **LLMs** → Fast schema-valid layouts and natural language-based edits  
- **Hybrid Approach** → GA ↔ LLM integration for iterative design with community feedback  

---

## Notebooks
- **`English_annotated.ipynb`**  
  Heavily annotated with Markdown explanations of the methodology, GA setup, evaluation, and results.  
  Serves as the **reference notebook**.

- **`French_with_header.ipynb`**  
  Applies the same pipeline to the **French Tatoeba corpus**.  
  Includes a clear header introduction with light commentary (methodology identical to English).

- **`Afrikaans_with_header.ipynb`**  
  Applies the same pipeline to the **Afrikaans Tatoeba corpus (100k sentences)**.  
  Includes a clear header introduction with light commentary.

---

## Data Sources
- **English:** TED2013 transcripts from the MKLOGA project  
- **French:** Full-sentence corpus from [Tatoeba](https://tatoeba.org/en)  
- **Afrikaans:** 100k sentences from [Tatoeba](https://tatoeba.org/en)  


How to Run:

- Clone the repository and install dependencies (jupyter, pandas, numpy, matplotlib, deap, etc.).
- Open English_annotated.ipynb in Jupyter Notebook to explore the full methodology.
- Run the French and Afrikaans notebooks to see the same pipeline applied to other languages.

Outputs include:

- Ergonomic cost/fitness scores (less-negative = better).
- Baseline comparisons (QWERTY, AZERTY, BÉPO, etc.).
- Optimised layouts from GA and LLM methods.

Notes

- Only the English notebook is heavily annotated.
- French and Afrikaans notebooks follow the same structure and are lightly documented to avoid redundancy.
- Layouts can be exported for deployment in LDML/CLDR-compatible systems.

Author: Adesh Maharaj
Date: August 2025
