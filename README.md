# TEI Encoding with LLMs

Ce dépôt contient un pipeline permettant de générer des sorties TEI par différents LLMs, sur plusieurs documents et stratégies de prompts, puis d'évaluer la performance de chaque modèle comparant les sorties à une vérité de terrain.

Objectifs principaux:

- génération de sorties TEI-XML à complexité progressive selon des niveaux différents d'encodage,
- évaluer des éléments différents selon chaque niveau d'encodage,
- à partir de l'évaluation du premier niveau, produire des tableaux et figures resumant les performances par modele/prompt/element.


## Installation

```text
import os
from lxml import etree
from pathlib import Path
import re
import glob
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from tqdm import tqdm
from rapidfuzz import fuzz
```


## Structure du projet

```text
.
├── internship-data/
    ├── input/      # textes source
    ├── output/         # sorties TEI des modeles
	|	├── niveau1/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau1.1/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau1.2/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.1/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.2/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.3/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.4/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.5/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.6/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.7/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau2.8/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau_parallele_2/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau_parallele_2.1/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	├── niveau_parallele_3/
	|	|	├── gemini_3.1-flash-lite-preview/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gemma-3-27b-it/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	├── gpt-5-mini/
	|	|	|	├── FS/
	|	|	|	├── FS-R/
	|	|	|	└── ZS/
	|	|	└── gpt-5.4-mini/
	|	|		├── FS/
	|	|		├── FS-R/
	|	|		└── ZS/
	|	└──  niveau_parallele_3.1/
	|		├── gemini_3.1-flash-lite-preview/
	|		|	├── FS/
	|		|	├── FS-R/
	|		|	└── ZS/
	|		├── gemma-3-27b-it/
	|		|	├── FS/
	|		|	├── FS-R/
	|		|	└── ZS/
	|		├── gpt-5-mini/
	|		|	├── FS/
	|		|	├── FS-R/
	|		|	└── ZS/
	|		└── gpt-5.4-mini/
	|			├── FS/
	|			├── FS-R/
	|			└── ZS/
    ├── prompts/    # prompts utilises
	|	├── niveau1/
	|	├── niveau1.1/
	|	├── niveau1.2/
	|	├── niveau2/
	|	├── niveau2.1/
	|	├── niveau2.2/
	|	├── niveau2.3/
	|	├── niveau2.4/
	|	├── niveau2.5/
	|	├── niveau2.6/
	|	├── niveau2.7/
	|	├── niveau2.8/
	|	├── niveau3/
	|	├── niveau3.1/
	|	├── niveau3.2/
	|	├── niveau_parallele_2/
	|	├── niveau_parallele_2.1/
	|	├── niveau_parallele_3/
	|	└──  niveau_parallele_3.1/
    ├── gt/    # TEI de reference (gold)
	|	├── niveau1/
	|	├── niveau2/
	|	└── niveau3/
	├── evaluation/        # rapports JSON par modele/prompt/traitement/document
	├── prepare_input_files.ipynb/
	├── few_shot_encoding_from_raw_text_refairexmlnonvalide.ipynb/ #génération de fichiers TEI par modele/prompt/traitement/document
	├── Eval_niveau1.ipynb/ # évaluation du niveau 1
	├── Eval_niveaux2et3.ipynb/ # évaluation des niveaux 2 et 3
	├── Eval_niveauxParall.ipynb/ # évaluation des niveaux 2 et 3 parallèles
	├── fromNote_toSeg.ipynb/ # notebook qui permet de transformer l'élément <note> en <seg>
    ├── generate_article_results.py/ 
	└── results/
			├── tables/
			└── figures/
```


## Modèles

On a utilisé quatre LLMs :

- `gemini-3.1-flash-lite-preview`
- `gemma-3-27b-it`
- `gpt-5-mini`
- `gpt-5.4-mini`

Stratégies de prompting :

- `ZS`
- `FS`
- `FS-R`

Traitement :

- `prompt unique` (niveaux 1, 2, 2.3, 2.6, 3, 2 parallèle, 2.1 parallèle, 3 parallèle et 3.1 parallèle)
- `prompt chaning`
  	- `premier prompt` (niveaux 1.1, 2.1, 2.4, 2.7 et 3.1)
    - `deuxième prompt` (niveaux 1.2, 2.2, 2.5, 2.8 et 3.2)


## Evaluation (notebook)

1) Le notebook `Eval_niveau1.ipynb` :

	1. lit un XML predit et un XML gold,
	2. aligne les entrees (`mainEntry`, `relatedEntry`),
	3. calcule precision/rappel/F1 + comptes,
	4. extrait les tags globaux et calcule leurs scores,
	5. ecrit un JSON dans `reports/`.

	Important pour les tags:

	- les scores sont calculés a partir des comptes globaux par document (pas d'alignement positionnel fin).

2) Les notebooks `Eval_niveaux2et3.ipynb` et `Eval_niveauxParall.ipynb` :

	1. lit un XML predit et un XML gold,
	2. extrait les tags globaux de chaque niveau et calcule leurs scores,
	3. ecrit un JSON dans `reports/`.

## Génération des tableaux et figures pour le niveau 1

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
