# TEI Encoding with LLMs

Ce depot contient un pipeline d'evaluation pour comparer des sorties TEI generees par differents LLMs, sur plusieurs documents et strategies de prompt.

Objectifs principaux:

- evaluer `mainEntry` et `relatedEntry` (precision, rappel, F1),
- evaluer les tags globaux (`usg_dom`, `usg_other`, `milestone`),
- produire des tableaux et figures resumant les performances par modele/prompt/element.


## Installation

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Structure du projet

```text
.
├── data/
│   ├── input/      # textes source
│   ├── gt/         # TEI de reference (gold)
│   ├── output/     # sorties TEI des modeles
│   └── prompts/    # prompts utilises
├── reports/        # rapports JSON par modele/prompt/document
├── scripts/
│   └── generate_article_results.py
├── evaluation.ipynb
└── article_results/
		├── tables/
		└── figures/
```


## Convention des rapports

Le script agrege les fichiers de `reports/` avec le schema suivant:

`<model>_<prompt>_<document>.json`

Prompts supportes:

- `ZS`
- `FS`
- `FS-R`


## Evaluation (notebook)

Le notebook `evaluation.ipynb`:

1. lit un XML predit et un XML gold,
2. aligne les entrees (`mainEntry`, `relatedEntry`),
3. calcule precision/rappel/F1 + comptes,
4. extrait les tags globaux et calcule leurs scores,
5. ecrit un JSON dans `reports/`.

Important pour les tags:

- les scores sont calcules a partir des comptes globaux par document (pas d'alignement positionnel fin).


## Generation des tableaux et figures

Script principal:

```sh
python scripts/generate_article_results.py
```

Options:

```sh
python scripts/generate_article_results.py \
	--reports-dir reports \
	--out-dir article_results \
	--gt-dir data/gt
```

Ce script produit:

- tables CSV + Markdown dans `article_results/tables/`,
- figures PNG dans `article_results/figures/`.


## Sorties principales

Tables:

- `article_results/tables/entries_by_document.md`
- `article_results/tables/tags_by_document.md`
- `article_results/tables/mean_f1_by_model_prompt.md`
- `article_results/tables/mean_scores_overall_documents.md`
- `article_results/tables/scores_matrix_entries.md`

Figures:

- `article_results/figures/entries_f1_heatmaps.png`
- `article_results/figures/tags_f1_heatmaps.png`
- `article_results/figures/micro_f1_by_element_model_prompt.png`
- `article_results/figures/micro_f1_by_model_mainEntry.png`
- `article_results/figures/micro_f1_by_model_relatedEntry.png`
- `article_results/figures/micro_f1_milestone_by_prompt_model.png`
- `article_results/figures/micro_f1_usg-dom_by_prompt_model.png`
- `article_results/figures/micro_f1_usg-other_by_prompt_model.png`
- `article_results/figures/fs_f1_by_document_and_model_mainEntry.png`
- `article_results/figures/fs_f1_by_document_and_model_relatedEntry.png`


## Macro vs micro

Dans `mean_scores_overall_documents.*`:

- `macro_f1`: moyenne simple des F1 par document,
- `micro_f1`: calcul global a partir des comptes agreges (TP/FP/FN implicites).


## Notes

- Le tri des modeles dans les figures est personnalise pour faciliter la lecture
- Les scores sont formates a 2 decimales dans les exports.