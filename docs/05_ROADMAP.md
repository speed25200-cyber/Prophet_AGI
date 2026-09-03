# 05 — Plan d'exécution

> Généré par `python -m prophet.plan`. Le budget est la contrainte structurante :
> les douze tracks ont demandé **434 heures-A100** pour un budget de **300**.
> Ce document est l'arbitrage, y compris ce qui est coupé.

## Principe : les portes de décision d'abord

Dix-sept pour cent du budget est consacré à des expériences **bon marché qui peuvent tuer
un plan cher**. Ce n'est pas de la prudence, c'est de l'arithmétique : 24 heures pour
savoir si le pari architectural central tient valent mieux que 200 heures dépensées à le
supposer.

Chaque porte a un **critère d'échec écrit à l'avance**. Une porte sans seuil pré-enregistré
n'est pas une porte, c'est une occasion de se convaincre.

---

# Compute plan — 300 A100-hours

Requested across all tracks: **434 h**. Available after a 10% reserve: **270 h**. Oversubscribed **1.6x**.

Funded 11 of 18 requests, 268 h allocated, 2 h unspent, 30 h held in reserve for reruns and preemption losses.

## Funded

| Pri | Track | Work | Kind | Hours |
|---:|---|---|---|---:|
| 1 | R06 | Prophet-mini pretraining | production | 90 |
| 1 | R04 | loop-vs-depth go/no-go | gate | 24 |
| 1 | R11 | evaluation | production | 19 |
| 1 | R07 | optimiser bake-off (trimmed) | gate | 14 |
| 1 | R01 | byte-frontend MFU probe | gate | 2 |
| 2 | R10 | post-training | production | 60 |
| 2 | R06 | data mixture ablations | ablation | 20 |
| 2 | R02 | hybrid recall gate (MK-NIAH) | gate | 8 |
| 2 | R09 | confidence-probe AUROC probe | gate | 3 |
| 3 | R05 | MoE routing and upcycling | ablation | 16 |
| 3 | R02 | long-context extension | production | 12 |

## Not funded

| Pri | Track | Work | Hours | Why it was cut |
|---:|---|---|---:|---|
| 3 | R04 | depth ablations | 24 | ranked below the funding line |
| 3 | R02 | interleave and long-context ablations | 20 | ranked below the funding line |
| 3 | R08 | quantisation ladder | 20 | ranked below the funding line |
| 4 | R03 | two-tier memory | 20 | ranked below the funding line |
| 4 | R09 | confidence head training | 20 | optional; below the funding line |
| 5 | R01 | byte-frontend retrofit | 36 | optional; below the funding line |
| 5 | R12 | vision adapter | 26 | optional; below the funding line |

## Gates run first

51 hours of gate experiments (17% of the budget) decide whether the expensive work is worth doing at all:

- **R04 — loop-vs-depth go/no-go** (24 h). Iso-FLOP comparison of looped depth against plain depth, plus depth generalisation. If looping does not beat equal-FLOP depth, the central architectural bet is dead and everything downstream changes.
  - Failure cancels: R04 depth ablations, R03 two-tier memory
- **R07 — optimiser bake-off (trimmed)** (14 h). Muon against a properly tuned AdamW. Break-even is a 1.058x speedup; below that the bake-off costs more than it saves and we keep AdamW.
- **R01 — byte-frontend MFU probe** (2 h). Measure realised MFU of patch gather/scatter kernels. Below 0.18 the entire byte-level track is dead, and 2 hours settles it.
  - Failure cancels: R01 byte-frontend ablations
- **R09 — confidence-probe AUROC probe** (3 h). Every published confidence-probe result is at 7B or above. If AUROC at 0.6B is below 0.70 the head is dropped. No training required.
  - Failure cancels: R09 confidence head training
- **R02 — hybrid recall gate (MK-NIAH)** (8 h). Multi-key retrieval is where linear mixers collapse. Gate the interleave ratio on it before committing to the stack.

---

## Calendrier

Le budget de 300 heures-A100 s'étale sur plusieurs semaines de sessions Colab
interruptibles. Une semaine de travail réelle représente environ 25–40 heures-A100
utilisables une fois déduites les pertes de préemption.

| Semaine | Contenu | Livrable vérifiable |
|---|---|---|
| **1** | Portes R01 (MFU), R07 (optimiseur), R09 (sonde AUROC). Infrastructure d'évaluation R11. | Trois verdicts écrits. Harnais d'évaluation à trois niveaux opérationnel. |
| **2** | Porte R04 (boucle contre profondeur) à ≥ 350M. Porte R02 (rappel multi-clés). | **Décision go/no-go sur le pari central.** |
| **3** | Ablations de mélange de données R06. Reproduction de 4 lignes de base concurrentes dans *notre* harnais. | Mélange figé. Tableau de bord avec des scores que nous avons nous-mêmes mesurés. |
| **4–7** | Pré-entraînement Prophet-mini (229M), phases A et B. Checkpoints toutes les 200 étapes. | Courbe de perte, checkpoint de plateau, évaluations Tier-1 tous les 3 jours. |
| **8** | Phase C : trois recuits branchés depuis le checkpoint de plateau, puis fusion. Extension de contexte. | Modèle de base fusionné, contexte 32k. |
| **9–10** | Post-entraînement R10 : mid-training raisonnement, SFT bimode, distillation on-policy, RL. | Modèle instruct. |
| **11** | Évaluation Tier-2 complète, quantification et export, carte du modèle. | Résultats publiables avec rapport de décontamination. |

---

## Critères d'échec pré-enregistrés

| Porte | Seuil d'échec | Conséquence |
|---|---|---|
| R01 MFU octet | MFU réalisé < 0.18 | Le track octet est abandonné. BPE 32k à repli octet définitif. |
| R07 optimiseur | Accélération < 1.058× | Muon abandonné, AdamW 8 bits retenu. |
| R04 profondeur bouclée | Pas meilleur qu'une profondeur simple à iso-FLOP | **Le pari central tombe.** Repli sur une pile dense classique ; R03 est annulé en cascade. |
| R09 sonde de confiance | AUROC < 0.70 à 0.6B | Tête de confiance abandonnée. |
| R02 rappel multi-clés | Effondrement sur MK-NIAH | Augmenter le nombre de couches globales — **pas** élargir la fenêtre. |

---

## La décision qui reste ouverte

Le point D10 de [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) — **partir de poids aléatoires
ou convertir un donneur ouvert** — n'est pas tranché, et il change le calendrier des
semaines 4 à 8. Le plan ci-dessus suppose l'entraînement de zéro pour Prophet-mini, ce qui
est faisable et constitue la preuve honnête de l'architecture.

Tout ce qui précède la semaine 4 est **commun aux deux voies**.

---

## Ce que le plan ne finance pas

Sept demandes sur dix-huit ne passent pas la ligne de flottaison, dont la mémoire
persistante (R03, 20 h) et le retrofit octet (R01, 36 h). Ce sont précisément les paris les
plus originaux du projet, et ils sont coupés parce qu'ils sont les moins prouvés. Si une
porte de décision libère du budget — un échec R04 annule 24 heures d'ablations de
profondeur — la réserve et les heures libérées vont d'abord à R03.
