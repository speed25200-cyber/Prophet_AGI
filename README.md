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
Qwen3-4B   ≈ 8.6e23 FLOPs d'entraînement
Prophet    ≈ 2.3e21 FLOPs d'entraînement      →  ~370× moins
```

Nous ne pouvons pas gagner en échelle. Nous pouvons gagner en **allocation**.

## Les cinq paris

| # | Pari | Verrou attaqué |
|---|---|---|
| 1 | **Dépenser les paramètres dans le calcul, pas dans le vocabulaire** — frontend adaptatif au lieu d'une table d'embedding de 128k | [§1](docs/00_PROBLEM_LANDSCAPE.md#1-le-verrou-du-tokenizer) |
| 2 | **Mémoire à état borné plutôt qu'à croissance linéaire** — pile hybride récurrent/attention | [§2](docs/00_PROBLEM_LANDSCAPE.md#2-le-coût-quadratique-de-lattention-et-le-mur-du-cache-kv) |
| 3 | **Acheter la profondeur avec du calcul, pas avec des poids** — cœur récurrent bouclé, profondeur réglable à l'exécution | [§4](docs/00_PROBLEM_LANDSCAPE.md#4-le-raisonnement--profondeur-fixe-et-pensée-verbeuse) |
| 4 | **Accumuler après le déploiement** — mémoire persistante + consolidation hors-ligne | [§3](docs/00_PROBLEM_LANDSCAPE.md#3-le-cerveau-gelé--aucune-mémoire-persistante-aucun-apprentissage-continu) |
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

## État du projet

> **Phase 0 — Recherche et conception.** Aucun poids entraîné à ce jour.
> Ce dépôt contient la cartographie des problèmes, les rapports de recherche par
> verrou, la spécification d'architecture et le plan d'exécution.

| Document | Contenu |
|---|---|
| [`docs/00_PROBLEM_LANDSCAPE.md`](docs/00_PROBLEM_LANDSCAPE.md) | Les 12 verrous attaqués, priorisés — **commencer ici** |
| [`docs/research/`](docs/research/) | Un rapport de recherche approfondi par verrou (R01–R12) |
| [`docs/01_ARCHITECTURE.md`](docs/01_ARCHITECTURE.md) | Spécification de l'architecture Prophet |
| [`docs/02_DATA.md`](docs/02_DATA.md) | Corpus, mélanges et pipeline de données |
| [`docs/03_TRAINING.md`](docs/03_TRAINING.md) | Recette d'entraînement et budget de compute |
| [`docs/04_EVAL.md`](docs/04_EVAL.md) | Tableau de bord et protocole d'évaluation |
| [`docs/05_ROADMAP.md`](docs/05_ROADMAP.md) | Plan d'exécution semaine par semaine |

## Ce que ce projet ne prétend pas faire

L'honnêteté sur les limites fait partie du cahier des charges. Prophet **ne** résoudra
**pas** : la couverture factuelle du monde (borne physique d'environ 2 bits par
paramètre), la généralisation hors distribution de type ARC-AGI, l'ancrage
sensorimoteur, ni la robustesse adversariale. Ces exclusions sont argumentées en
[§13](docs/00_PROBLEM_LANDSCAPE.md#13-verrous-reconnus-mais-hors-périmètre-v1).

## Licence

Apache-2.0 pour le code. Les jeux de données et modèles tiers conservent leurs licences
respectives ; l'audit est suivi dans [`docs/02_DATA.md`](docs/02_DATA.md).
