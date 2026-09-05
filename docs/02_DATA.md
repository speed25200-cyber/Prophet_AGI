# 02 — Données

> Le mélange est produit par `prophet/data/recipes.py` et validé par
> `prophet.data.mixture.Mixture.validate()`. Ce document est **généré** : ne pas
> l'éditer à la main, éditer la recette.
>
> Régénérer : `python scripts/build_data_docs.py`

## Pourquoi ce budget et pas celui de la littérature

Le track R06 a écrit son plan pour 300 milliards de tokens. Notre budget mesuré
(`python -m prophet.scaling`) est d'environ **40 milliards** à 300 heures-A100 et
~500M paramètres actifs. Les **proportions** sont le résultat de recherche ; les valeurs
absolues suivent le budget réel.

Cinq décisions portent la recette (R06) :

| # | Décision | Raison |
|---|---|---|
| D1 | WSD en trois phases, avec une phase de recuit anormalement grasse (70/20/10) | La phase de décroissance est là où les benchmarks se font, et l'effet est **le plus fort sur les petits modèles** — donc sur nous. |
| D2 | Nemotron-CC-v2 comme dorsale web, pas FineWeb-Edu | Classé au-dessus de DCLM et FineWeb-Edu sur MMLU aval. FineWeb-Edu reste pour la décorrélation. |
| D3 | ~13 % de synthétique, **entièrement amorcé sur des documents réels** | Capte l'effet de reformulation sans entrer dans le régime d'effondrement. Aucune auto-génération libre. |
| D4 | Maths et code sur-pondérés (26.7 % cumulés) | Seules catégories avec des gains de benchmark à deux chiffres par unité de part de tokens. Quatre de nos neuf benchmarks cibles en dépendent. |
| D5 | Multilingue plafonné à 2.5 %, contexte long à 1.4 % | Tous les benchmarks cibles sont anglophones. À 40B tokens, 12 % de multilingue coûterait 4.8B tokens pour zéro point. |

## Garde-fous automatiques

`Mixture.validate()` refuse un mélange avant tout téléchargement si :

- les poids de phase ou de source ne somment pas à 1 ;
- une source est répétée plus de **4 époques** (au-delà, la répétition cesse de payer) ;
- une phase est vide.

De plus, chaque source **doit déclarer une licence**, et les sources dont la taille n'est
pas vérifiée sont listées explicitement plutôt que silencieusement acceptées.

## Décontamination

`prophet/data/decontaminate.py` : recouvrement de n-grammes (13-grammes par défaut,
seuil 50 %) sur un texte normalisé en casse, ponctuation et accents. Les items trop
courts pour produire assez de n-grammes basculent sur une correspondance exacte au lieu
d'être ignorés. Le rapport par benchmark est destiné à la carte du modèle.

La contamination ne se signale pas : la perte d'entraînement baisse normalement et le
score **monte**. C'est précisément le signal qu'on aurait envie de célébrer.

## Le chemin réel : des fichiers au chargeur reprenable

`prophet/data/corpus.py` relie le mélange ci-dessous au chargeur de
`prophet/data/streaming.py`, qui ne sait rien de l'origine des documents — et c'est ce qui
le rend testable hors ligne et reprenable à l'identique. Quatre propriétés, chacune une
décision :

| Propriété | Mécanisme | Ce qui aurait cassé sans |
|---|---|---|
| **Reprise en O(1)** | Un index d'offsets par ligne, construit une fois et mis en cache à côté du fichier ; `open(start)` est un `seek`. Un flux Hub ne sait pas chercher : `open(start)` y est un `skip`, en O(start), et le dit. | Rejouer 40 000 pas de flux à chaque reprise. |
| **Le rejet ne décale pas le flux** | Un document contaminé produit une liste de tokens **vide** au lieu d'être sauté : le curseur compte les documents bruts, le packer ignore les vides. | Un curseur dépendant du filtre : reprendre avec un jeu de benchmarks ré-indexé aurait lu un autre corpus, sans erreur. |
| **Plafond d'époques au tirage** | `Mixture.validate()` vérifie le plan ; `TokenisedSource` vérifie le run. Son curseur ne revient jamais à zéro, donc `curseur / n_documents` est le nombre d'époques et une source tirée au-delà de 4 **lève** au lieu de se répéter. Une source de taille inconnue (Hub) ne peut pas être plafonnée et est déclarée telle. | Une répétition silencieuse là où elle cesse de payer. |
| **Les phases sont un planning de chargeurs** | Une phase = ses sources et ses poids = son `StreamingLoader` ; `PhasedLoader` bascule au pas que le budget de tokens implique et sauvegarde chaque sous-chargeur. | Une interruption en phase C reprenant en phase A. |

Le script d'entraînement l'assemble :

```bash
python scripts/train_tokenizer.py --data-root corpus/ --out tokenizer.json
python scripts/train.py --config configs/prophet_mini.json --tokenizer tokenizer.json \
    --data-root corpus/ --benchmarks benchmarks/ --tokens 16.1e9 ...
```

où `corpus/` contient `<source>.jsonl` ou `<source>/*.jsonl` nommés comme les sources du
mélange, et `benchmarks/` un `<nom>.jsonl` par jeu de test. `--hub` autorise le streaming
des sources absentes en local — après `scripts/verify_datasets.py`, jamais avant. Sans
`--benchmarks`, le script refuse de partir ; `--benchmarks ''` est le seul moyen de
courir non décontaminé, et il faut l'écrire. Vérifié de bout en bout sur CPU avec un
corpus minuscule : trois phases, checkpoint, reprise dans la phase en cours.

Une seule longueur de séquence sert toutes les phases : la croissance de contexte par
phase du plan R06 est l'extension longue R02, non financée, et changer la forme du batch
en cours de run changerait la mémoire contre laquelle le budget a été vérifié.

---

# Data mixture — prophet-v1

Three-phase WSD mixture from track R06, proportions preserved and token counts scaled to the measured single-A100 budget.

Total budget: **40.0B tokens**

## Phase A-stable — 28.0B tokens (70%), context 4096, LR warmup_then_constant

Build the world model. Broadest mixture, highest token volume, constant peak learning rate. No instruction data at all — it is deliberately saved for the phases where recency makes it count.

| Source | HF id | Domain | Share | Tokens | Epochs | Licence |
|---|---|---|---:|---:|---:|---|
| nemotron-cc-v2-hq | `nvidia/Nemotron-CC-v2` | web | 22.0% | 6.16B | 0.01 | NVIDIA Open Data (permissive) |
| fineweb-edu | `HuggingFaceFW/fineweb-edu` | web | 18.0% | 5.04B | 0.00 | ODC-By-1.0 |
| dclm-baseline | `mlfoundations/dclm-baseline-1.0` | web | 14.0% | 3.92B | 0.00 | CC-BY-4.0 |
| stack-edu | `HuggingFaceTB/stack-edu` | code | 9.0% | 2.52B | 0.02 | ODC-By-1.0 |
| nemotron-cc-synthetic | `nvidia/Nemotron-CC-v2` | synthetic | 8.0% | 2.24B | 0.00 | NVIDIA Open Data (permissive) |
| nemotron-cc-math | `nvidia/Nemotron-CC-Math-v1` | math | 6.0% | 1.68B | 0.03 | NVIDIA Open Data (permissive) |
| nemotron-code | `nvidia/Nemotron-Pretraining-Code-v2` | code | 5.0% | 1.40B | — | NVIDIA Open Data (permissive) |
| curated-reference | `allenai/dolma3` | reference | 5.0% | 1.40B | — | ODC-By-1.0 |
| cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | synthetic | 4.0% | 1.12B | 0.04 | Apache-2.0 |
| megamath-finemath | `LLM360/MegaMath` | math | 3.0% | 0.84B | 0.03 | ODC-By-1.0 |
| proof-pile-2 | `EleutherAI/proof-pile-2` | reference | 3.0% | 0.84B | 0.02 | mixed (per-subset) |
| fineweb2-hq | `epfml/FineWeb2-HQ` | multilingual | 3.0% | 0.84B | 0.00 | ODC-By-1.0 |

## Phase B-midtrain — 8.0B tokens (20%), context 16384, LR hold_then_slow_decay

Capability injection: math, code and reasoning format, while the model is still plastic. Context is extended 4k -> 16k over the last third of the phase.

| Source | HF id | Domain | Share | Tokens | Epochs | Licence |
|---|---|---|---:|---:|---:|---|
| stack-edu-anneal | `HuggingFaceTB/stack-edu` | code | 16.0% | 1.28B | 0.01 | ODC-By-1.0 |
| nemotron-cc-v2-hq | `nvidia/Nemotron-CC-v2` | web | 16.0% | 1.28B | 0.00 | NVIDIA Open Data (permissive) |
| nemotron-cc-math-up | `nvidia/Nemotron-CC-Math-v1` | math | 14.0% | 1.12B | 0.02 | NVIDIA Open Data (permissive) |
| fineweb-edu-4 | `HuggingFaceFW/fineweb-edu` | web | 10.0% | 0.80B | 0.00 | ODC-By-1.0 |
| nemotron-diverse-qa | `nvidia/Nemotron-CC-v2` | synthetic | 10.0% | 0.80B | 0.00 | NVIDIA Open Data (permissive) |
| megamath-finemath | `LLM360/MegaMath` | math | 8.0% | 0.64B | 0.03 | ODC-By-1.0 |
| instruction-as-documents | `HuggingFaceTB/smoltalk2` | instruction | 8.0% | 0.64B | — | Apache-2.0 |
| cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | synthetic | 6.0% | 0.48B | 0.02 | Apache-2.0 |
| curated-reference | `allenai/dolma3` | reference | 5.0% | 0.40B | — | ODC-By-1.0 |
| long-documents | `EleutherAI/proof-pile-2` | long_context | 5.0% | 0.40B | 0.01 | mixed (per-subset) |
| fineweb2-hq | `epfml/FineWeb2-HQ` | multilingual | 2.0% | 0.16B | 0.00 | ODC-By-1.0 |

## Phase C-anneal — 4.0B tokens (10%), context 32768, LR linear_to_zero

Learning rate decays to zero on the highest-quality data available. Zero low-quality web. Run three times from the same phase-B checkpoint with different orderings and soup the results — nearly free relative to the phase cost, and reliably worth a point.

| Source | HF id | Domain | Share | Tokens | Epochs | Licence |
|---|---|---|---:|---:|---:|---|
| openthoughts3 | `open-thoughts/OpenThoughts3-1.2M` | instruction | 18.0% | 0.72B | 0.36 | Apache-2.0 |
| nemotron-cc-math | `nvidia/Nemotron-CC-Math-v1` | math | 16.0% | 0.64B | 0.01 | NVIDIA Open Data (permissive) |
| opc-annealing | `OpenCoder-LLM/opc-annealing-corpus` | code | 14.0% | 0.56B | — | MIT |
| instruction-constraints | `HuggingFaceTB/smoltalk2` | instruction | 14.0% | 0.56B | — | Apache-2.0 |
| fineweb-edu-top | `HuggingFaceFW/fineweb-edu` | web | 12.0% | 0.48B | 0.00 | ODC-By-1.0 |
| nemotron-diverse-qa | `nvidia/Nemotron-CC-v2` | synthetic | 10.0% | 0.40B | 0.00 | NVIDIA Open Data (permissive) |
| curated-reference | `allenai/dolma3` | reference | 6.0% | 0.24B | — | ODC-By-1.0 |
| cosmopedia-v2 | `HuggingFaceTB/smollm-corpus` | synthetic | 6.0% | 0.24B | 0.01 | Apache-2.0 |
| long-context-32k | `EleutherAI/proof-pile-2` | long_context | 4.0% | 0.16B | 0.00 | mixed (per-subset) |

## Aggregate by domain

| Domain | Tokens | Share |
|---|---:|---:|
| web | 17.68B | 44.2% |
| code | 5.76B | 14.4% |
| synthetic | 5.28B | 13.2% |
| math | 4.92B | 12.3% |
| reference | 2.88B | 7.2% |
| instruction | 1.92B | 4.8% |
| multilingual | 1.00B | 2.5% |
| long_context | 0.56B | 1.4% |

## Licences

| Licence | Sources |
|---|---|
| Apache-2.0 | 6 |
| CC-BY-4.0 | 1 |
| MIT | 1 |
| NVIDIA Open Data (permissive) | 9 |
| ODC-By-1.0 | 12 |
| mixed (per-subset) | 3 |

## Unverified sizes

Epoch counts could not be checked for these sources; confirm before use:

- A-stable/nemotron-code
- A-stable/curated-reference
- B-midtrain/curated-reference
- B-midtrain/instruction-as-documents
- C-anneal/opc-annealing
- C-anneal/instruction-constraints
- C-anneal/curated-reference

---

## Statut de vérification

Les identifiants de datasets proviennent des rapports de recherche. Avant tout
téléchargement, chacun doit être confronté au Hub : existence, taille réelle, licence
exacte, et disponibilité de la configuration nommée. Les sources marquées « non
vérifiées » ci-dessus sont celles dont la taille n'a pas pu être confirmée — leur nombre
d'époques n'est donc pas contrôlé.
