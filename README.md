<div align="center">

# Prophet

**Une architecture de modèle de langage repensée pour le matériel que l'on possède réellement.**

`1× RTX 5090` · `Mac Studio` · `iPhone 17 Pro`
Entraîné sur **un seul A100 80GB**.

</div>

---

## Pourquoi

Les laboratoires actuels achètent la capacité avec du capital : 36 000 milliards de
tokens, des dizaines de milliers de GPU, des milliards de dollars. Cette voie nous est
fermée. Prophet part du constat inverse : **quelles décisions architecturales prendrait-on
si le compute d'entraînement était rare et la mémoire d'inférence encore plus rare ?**

Ce sont des décisions que personne disposant de 200 000 GPU n'a de raison de prendre.
C'est exactement là que se trouve l'espace de conception inexploré.

```
Qwen3-4B      ≈ 8.6e23 FLOPs d'entraînement   (2 200 000 heures-A100)
SmolLM2-360M  ≈ 8.6e21 FLOPs d'entraînement   (    22 000 heures-A100)
Prophet       ≈ 1.2e20 FLOPs d'entraînement   (       300 heures-A100)
```

Nous sommes **73× sous le plus frugal** des concurrents, et 7 326× sous le plus gros.
Ces chiffres sont produits par `python -m prophet.scaling`, pas estimés.

Nous ne pouvons pas gagner en échelle. Nous pouvons gagner en **allocation**.

## Les cinq paris

| # | Pari | Verrou attaqué |
|---|---|---|
| 1 | **Dépenser les paramètres dans le calcul, pas dans le vocabulaire** — frontend adaptatif au lieu d'une table d'embedding de 128k | [§1](docs/00_PROBLEM_LANDSCAPE.md#1-le-verrou-du-tokenizer) |
| 2 | **Mémoire à état borné plutôt qu'à croissance linéaire** — pile hybride récurrent/attention | [§2](docs/00_PROBLEM_LANDSCAPE.md#2-le-coût-quadratique-de-lattention-et-le-mur-du-cache-kv) |
| 3 | **Acheter la profondeur avec du calcul, pas avec des poids** — cœur récurrent bouclé, profondeur réglable à l'exécution | [§4](docs/00_PROBLEM_LANDSCAPE.md#4-le-raisonnement--profondeur-fixe-et-pensée-verbeuse) |
| 4 | **Accumuler après le déploiement** — mémoire persistante + consolidation hors-ligne ([implémentée](docs/06_MEMORY.md)) | [§3](docs/00_PROBLEM_LANDSCAPE.md#3-le-cerveau-gelé--aucune-mémoire-persistante-aucun-apprentissage-continu) |
| 5 | **Savoir ce qu'on ignore** — abstention calibrée plutôt que couverture factuelle (physiquement hors d'atteinte à 1.3B params) | [§9](docs/00_PROBLEM_LANDSCAPE.md#9-lhallucination-et-labsence-de-calibration) |

La profondeur récurrente est le pari central : elle transforme la profondeur en un
**cadran réglé à l'exécution** — faible sur iPhone, élevé sur une RTX 5090 — et donne
donc un seul modèle qui couvre les trois cibles matérielles.

## Cibles matérielles

| Cible | Mémoire | Bande passante | Variante |
|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | ~1.79 TB/s | Prophet (complet, profondeur max) |
| Mac Studio Ultra | 96–512 GB unifiée | ~0.8 TB/s | Prophet (complet, contexte long) |
| iPhone 17 Pro | ~8 GB unifiée | ~0.06–0.12 TB/s | Prophet-mini (dense, profondeur réduite) |

## Configurations retenues

Produites par `python scripts/design_search.py`, qui énumère l'espace de conception et ne
retient que ce qui satisfait **simultanément** la mémoire d'entraînement (un A100 80GB),
le budget de tokens, la mémoire de l'appareil cible et l'absence de mauvaise allocation.

| | Total | Actifs/token | Prof. effective | Tokens | Cible |
|---|---:|---:|---:|---:|---|
| **Prophet-main** | 3.83B | 408M | 24 (k=4) | 16.1B | 5090 / Mac Studio |
| **Prophet-mini** | 253M | 236M | 14 (k=2) | 52.1B | iPhone 17 Pro |

Rapport de sparsité 9.4× : la capacité d'un modèle de 3.8B pour le coût par token d'un
modèle de 408M.

## État du projet

> **Phase 0 — Recherche et conception terminées. Aucun poids entraîné.**
> Le dépôt contient la recherche, l'architecture arbitrée, le tokenizer, le chemin de
> données réel (fichiers ou Hub, décontamination dans le flux, phases reprenables),
> l'infrastructure d'entraînement, le harnais d'évaluation, le pilier agentique
> (actions typées, vérification à tiers, quarantaine, et le chemin des épisodes promus
> vers le corpus) et le plan d'exécution. **~430 tests passent** ; la boucle
> d'entraînement tourne de bout en bout sur corpus synthétique et sur un corpus local
> minuscule, avec reprise dans la phase en cours.
>
> Réserve honnête : les identifiants de datasets et le tableau de bord concurrent
> proviennent de rapports rédigés alors que l'accès au Hub et à arXiv était bloqué par le
> proxy sortant. Ce sont des **cibles à vérifier**, pas des preuves — c'est la première
> tâche du plan.

| Document | Contenu |
|---|---|
| [`docs/00_PROBLEM_LANDSCAPE.md`](docs/00_PROBLEM_LANDSCAPE.md) | Les 12 verrous attaqués, priorisés — **commencer ici** |
| [`docs/research/README.md`](docs/research/README.md) | **Synthèse** des 12 tracks + avertissement de provenance |
| [`docs/research/`](docs/research/) | Les 12 rapports détaillés (R01–R12, ~7 500 lignes) |
| [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) | Architecture et **registre de décisions** |
| [`docs/02_DATA.md`](docs/02_DATA.md) | Mélange de données (généré depuis la recette) |
| [`docs/03_TRAINING.md`](docs/03_TRAINING.md) | Recette d'entraînement |
| [`docs/04_EVAL.md`](docs/04_EVAL.md) | Tableau de bord et protocole |
| [`docs/05_ROADMAP.md`](docs/05_ROADMAP.md) | Plan sur 11 semaines et arbitrage du budget |
| [`docs/06_MEMORY.md`](docs/06_MEMORY.md) | Mémoire persistante : conception, mesures, limites |
| [`docs/07_WALLS.md`](docs/07_WALLS.md) | **Les murs** : mécanisme des verrous profonds, y compris ceux qu'on ne franchit pas |
| [`docs/08_AGENT.md`](docs/08_AGENT.md) | **Le pilier agentique** : la boucle, les têtes d'action, la vérification, ce qui est construit et ce qui ne l'est pas |

## Outils

Tout est sans dépendance lourde et exécutable immédiatement :

```bash
python -m prophet.scaling --sweep          # points de fonctionnement par budget
python -m prophet.budget configs/*.json    # paramètres, mémoire, débit par appareil
python -m prophet.plan                     # allocation des heures-A100 entre tracks
python scripts/design_search.py            # recherche de conception sous contraintes
python scripts/build_data_docs.py          # régénère le mélange de données
python scripts/verify_datasets.py          # confronte les identifiants au Hub (semaine 1)
python scripts/verify_donors.py            # confronte les donneurs au Hub (semaine 1)
python scripts/convert_donor.py --donor qwen3-1.7b --plan-only
python scripts/train.py --config configs/prophet_tiny_smoke.json --smoke
python scripts/train_tokenizer.py --data-root corpus/ --out tokenizer.json   # puis :
python scripts/train.py --config configs/prophet_mini.json --tokenizer tokenizer.json \
    --data-root corpus/ --benchmarks benchmarks/ --tokens 16.1e9        # run réel
python scripts/gpu_check.py --config configs/prophet_mini.json   # sur A100 : noyau, tok/s, mémoire
python scripts/colab_session.py --config configs/prophet_mini.json --work /content/drive/MyDrive/prophet \
    --session-minutes 600 -- --tokenizer tokenizer.json --data-root corpus/ --benchmarks benchmarks/
python -m pytest tests/ -q                 # ~430 tests (les tests GPU sont sautés sans CUDA)
```

Ces outils ne sont pas décoratifs : ils ont corrigé deux erreurs de conception avant
qu'elles ne coûtent quoi que ce soit — un budget de tokens surestimé d'un facteur 20, et
805M de paramètres gaspillés dans des tables de hachage.

## La voie retenue : deux modèles, deux origines

Cinq analyses indépendantes concluent que **surpasser la concurrence par
pré-entraînement depuis des poids aléatoires est arithmétiquement exclu** à ce budget.
La réponse retenue n'est pas de choisir un camp, mais de faire les deux sur deux modèles :

| Modèle | Origine | Budget | Rôle |
|---|---|---:|---|
| **Prophet-mini** (253M) | Poids aléatoires | 85 h-A100 | Preuve honnête de l'architecture. Cible iPhone. |
| **Prophet-main** (~970M) | Conversion d'un donneur Apache-2.0 | 30 h-A100 | Modèle compétitif. 89 % des paramètres hérités. |

Le rapport de coût — 85 heures contre 30 — est le résultat central : la conversion coûte
un tiers de l'entraînement de zéro pour un modèle quatre fois plus gros, parce qu'elle
n'achète que l'architecture. Mesuré sur Qwen3-1.7B : 1.72B paramètres et 28 couches
deviennent 1.02B paramètres et 12 blocs pour **la même profondeur effective de 28**.

```bash
python scripts/convert_donor.py --donor qwen3-1.7b --plan-only
```

Détail et garde-fous (licence, couverture minimale, vérification) en
[`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) §7.

## Ce que ce projet ne prétend pas faire

L'honnêteté sur les limites fait partie du cahier des charges. Prophet **ne** résoudra
**pas** : la couverture factuelle du monde (borne physique d'environ 2 bits par
paramètre), la généralisation hors distribution de type ARC-AGI, l'ancrage
sensorimoteur, ni la robustesse adversariale. Ces exclusions sont argumentées en
[§13](docs/00_PROBLEM_LANDSCAPE.md#13-verrous-reconnus-mais-hors-périmètre-v1).

## Licence

Apache-2.0 pour le code. Les jeux de données et modèles tiers conservent leurs licences
respectives ; l'audit est suivi dans [`docs/02_DATA.md`](docs/02_DATA.md).
