# TEI Encoding with LLMs


This repository contains code and resources for using Large Language Models (LLMs) to assist in the encoding of texts in the Text Encoding Initiative (TEI) format. The TEI is a standard for representing texts in digital form, and LLMs can help automate and enhance the encoding process.


## Installation

To install project dependencies:

```sh
python -m venv .venv  # create virtual env.
source .venv/bin/activate  # use virtual env (linux, macos)
python -m pip install -r requirements.txt  # install dependencies
```


## Project structure:

```
├── data/  
│   ├── input/            # fichiers d'entrée
│   │   └── TR5_p489-490.txt
│   ├── prompts/          # prompts utilisés
│   │   ├── Prompt_1.txt
│   │   ├── Prompt_2.txt
│   │   └── Prompt_3.txt
│   ├── gt/               # ground truth
│       └── TR5_p489-490.xml
│   ├── output/           # sorties des modèles
│   │    ├── gpt-5-mini/
│   │    │   ├── Prompt_1/
│   │    │   │   └── TR5_p489-490.xml
│   │    │   └── Prompt_2/
│   │    │   │   └── TR5_p489-490.xml
│   │    │   └── gemma-3-27b-it/
│   │    │       ├── Prompt_1/
│   │    │       │    └── TR5_p489-490.xml
│   │    │       └── Prompt_2/
│   │    │           └── TR5_p489-490.xml
└── reports/       # métriques / résultats d'évaluation
```


## Générer des tableaux et graphiques pour un article

Le script suivant agrège les fichiers JSON de `reports/` et produit:

- des tableaux CSV et Markdown par document / modèle / prompt,
- des scores pour `mainEntry`, `relatedEntry` et tous les tags,
- des graphiques (heatmaps + bar chart).

Commande:

```sh
python scripts/generate_article_results.py
```

Sorties:

- `article_results/tables/entries_by_document.md`
- `article_results/tables/tags_by_document.md`
- `article_results/tables/mean_f1_by_model_prompt.md`
- `article_results/figures/entries_f1_heatmaps.png`
- `article_results/figures/tags_f1_heatmaps.png`
- `article_results/figures/mean_f1_by_element_model_prompt.png`

Options:

```sh
python scripts/generate_article_results.py --reports-dir reports --out-dir article_results
```