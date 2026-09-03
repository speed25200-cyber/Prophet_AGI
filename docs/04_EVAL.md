# 04 — Évaluation

> Track R11. Sans boucle d'évaluation rapide et fiable, tous les autres tracks sont
> aveugles — et un tableau de bord non vérifié n'est pas un tableau de bord, c'est une
> ambition.

---

## 1. Le problème du signal précoce

La plupart des benchmarks sont **au niveau du hasard sous 500M paramètres**. À 130M
paramètres et 2.6B tokens — l'échelle de nos ablations — ARC-Challenge, WinoGrande,
CommonsenseQA et MMLU sont tous cloués au hasard. Décider d'un mélange de données sur ces
scores revient à décider à pile ou face.

**Métrique de décision retenue : les bits par octet (BPB) sur des domaines tenus à l'écart**,
pas l'exactitude. Le BPB bouge de façon monotone dès les premiers milliards de tokens,
alors que l'exactitude ne décolle pas.

Suite d'ablation (niveau A1, ~5.6 heures-A100 par bras) :

| Usage | Tâches |
|---|---|
| **Décision primaire** | BPB sur 8 domaines tenus à l'écart ; BPB de la tâche sur les réponses de référence |
| Exactitude exploitable | `lambada_openai`, `sciq`, `piqa`, `arc_easy`, `hellaswag`, `social_iqa` |
| Bruité, à lire avec prudence | `openbookqa` |
| **BPB uniquement** (hasard en exactitude) | `arc_challenge`, `commonsense_qa`, `winogrande`, `mmlu_continuation` |

**Explicitement exclus des ablations** : TruthfulQA (scaling inverse — un modèle qui
s'améliore y régresse), BoolQ (dominé par le prior majoritaire), GSM8K et HumanEval
(plancher à 0.0 à cette échelle, donc zéro information).

---

## 2. Système à trois niveaux

| Niveau | Durée | Fréquence | Rôle |
|---|---|---|---|
| **Tier-0 — fumée** | < 5 min | Chaque checkpoint | Le modèle produit-il encore du texte cohérent ? Attrape les divergences en quelques minutes plutôt qu'en un jour. |
| **Tier-1 — ablation** | ~16 min | Tous les 3 jours | La suite de signal précoce. Décide des ablations. |
| **Tier-2 — publication** | heures | 4 fois dans le projet | Suite complète. **Trois benchmarks sont réservés** et ne sont jamais exécutés avant l'évaluation finale pré-enregistrée. |

Réserver des benchmarks est une contrainte volontaire : dix-sept ablations offrent
largement assez d'occasions de surajuster le tableau de bord sans s'en rendre compte.

---

## 3. Le tableau de bord

R11 a produit les scores à battre. **Avertissement de provenance, à conserver** : l'accès
web des agents de recherche a été dégradé (arxiv et HuggingFace bloqués par le proxy,
quota de recherche épuisé), donc ces chiffres ont été écrits de mémoire. **Ce sont des
cibles, pas des preuves.**

La correction est obligatoire et fait partie du plan : chaque cellule doit être sourcée
sur une publication primaire, **puis les quatre concurrents principaux doivent être
réexécutés dans notre propre harnais**. Comparer notre score mesuré au score annoncé par
quelqu'un d'autre, avec un format de prompt différent, ne prouve rien.

| Benchmark | Meilleur concurrent à battre | Cible |
|---|---|---|
| MMLU (5-shot) | Qwen3-4B ~73 | ≥ 70 |
| MMLU-Pro | Phi-4-mini 52.8 | ≥ 50 |
| GSM8K (8-shot) | Phi-4-mini 88.6 | ≥ 88 |
| HumanEval+ / MBPP+ | Phi-4-mini 74.4 / 65.3 | ≥ 70 / ≥ 62 |
| AIME'24 (avg@32, mode réflexion) | Phi-4-mini-reasoning 57.5 | ≥ 45 |
| IFEval (prompt-strict) | Qwen3-4B 81.9 | ≥ 80 |
| RULER @32k | — | ≥ 85 |
| BFCL-v3 (outils) | Llama-3.2-3B 67.0 | ≥ 62 |

**Nuance honnête, à ne pas enterrer.** Ces cibles sont celles d'un modèle de 1.3B actifs.
Notre point de fonctionnement réel est de **369M actifs pour ~25B tokens**, soit 73× à
7 300× moins de compute que les modèles ci-dessus. Sur les benchmarks de connaissance
(MMLU, GPQA), les atteindre par pré-entraînement de zéro est **arithmétiquement exclu** —
c'est le constat de [`00_PROBLEM_LANDSCAPE.md`](00_PROBLEM_LANDSCAPE.md) §9 sur la capacité
de ~2 bits par paramètre.

Le tableau ci-dessus est donc la cible d'un Prophet **issu d'une conversion de donneur**
(§7 de l'architecture). Pour un Prophet-mini entraîné de zéro, le groupe de comparaison
honnête est SmolLM2-360M, Qwen3-0.6B et Llama-3.2-1B.

---

## 4. La métrique où nous pouvons réellement dominer

Les benchmarks de connaissance sont perdus d'avance. La qualité **par gigaoctet déployé**
ne l'est pas — et c'est la métrique qui compte pour quelqu'un qui fait tourner un modèle
sur son propre matériel.

**Métrique phare proposée : PQI/GB** — qualité par gigaoctet de mémoire déployée.
Délibérément plus dure que la qualité par paramètre actif, qui flatte tautologiquement les
MoE.

Protocole d'efficacité, par appareil : tokens/seconde en décodage, temps jusqu'au premier
token, mémoire de pointe, et énergie. Mesurés dans les mêmes conditions pour nous et pour
les concurrents, sinon le chiffre ne veut rien dire.

---

## 5. Décontamination

Implémentée dans `prophet/data/decontaminate.py`. Recouvrement de 13-grammes à seuil 50 %
sur texte normalisé en casse, accents et ponctuation, avec repli sur correspondance exacte
pour les items trop courts.

Le rapport par benchmark va dans la carte du modèle. La contamination ne se signale
jamais : la perte baisse normalement et le score **monte** — exactement le signal qu'on
aurait envie de célébrer.

---

## 6. Règles d'intégrité

1. **Décider sur le BPB tenu à l'écart**, jamais sur un benchmark cible.
2. **Trois benchmarks Tier-2 réservés**, jamais exécutés avant l'évaluation finale.
3. **Reproduire les lignes de base nous-mêmes** avant toute comparaison.
4. **Publier le rapport de décontamination** avec les scores.
5. **Rapporter les échecs.** Un score non reproduit est marqué comme tel.
