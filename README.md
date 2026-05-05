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

'''
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
'''