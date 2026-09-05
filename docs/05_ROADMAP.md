# 05 — Plan d'exécution

> Généré par `python -m prophet.plan`. Le budget est la contrainte structurante :
> les seize tracks ont demandé **470 heures-A100** pour un budget de **300**.
> Ce document est l'arbitrage, y compris ce qui est coupé.

## La voie retenue : mixte

Décision D10, tranchée ([`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) §7) — deux modèles,
deux origines, une seule architecture :

| Modèle | Origine | Budget | Rôle |
|---|---|---:|---|
| **Prophet-mini** (253M) | Poids aléatoires | 85 h | Preuve honnête de l'architecture. Cible iPhone. |
| **Prophet-main** (~1016M) | Conversion d'un donneur Apache-2.0 | 30 h | Modèle compétitif. 85% des paramètres hérités. |

Le rapport de coût — 85 heures contre 30 — est le résultat central : la conversion coûte
un tiers de l'entraînement de zéro pour un modèle quatre fois plus gros, parce qu'elle
n'achète que l'architecture.

## Principe : les portes de décision d'abord

Quarante heures-A100 sont consacrées à des expériences **bon marché qui peuvent tuer un
plan cher**. Ce n'est pas de la prudence, c'est de l'arithmétique : 24 heures pour savoir
si le pari architectural central tient valent mieux que 200 heures dépensées à le
supposer.

Chaque porte a un **critère d'échec écrit à l'avance**. Une porte sans seuil
pré-enregistré n'est pas une porte, c'est une occasion de se convaincre.

## Note sur l'ordonnancement

L'allocateur départage les demandes de même priorité par **ordre de déclaration**, pas par
coût. Le tri par coût croissant paraît efficace et ne l'est pas : il finance trois petites
demandes avant celle dont dépend le livrable, et le déficit retombe sur ce qui est le plus
gros plutôt que sur ce qui est le moins important. C'est exactement ce qui s'est produit
au premier essai — le post-entraînement, 45 heures, était éjecté au profit de trois
ablations.

---

# Compute plan — 300 A100-hours

Requested across all tracks: **542 h**. Available after a 10% reserve: **270 h**. Oversubscribed **2.0x**.

Funded 13 of 26 requests, 270 h allocated, 0 h unspent (added to the reserve), 30 h held for reruns and preemption losses.

Allocation is in strict priority order with no backfill: an item that does not fit stops the line rather than being skipped so that cheaper work behind it can squeeze in.

## Funded

| Pri | Track | Work | Kind | Hours |
|---:|---|---|---|---:|
| 1 | R06 | Prophet-mini pretraining | production | 85 |
| 1 | R10 | post-training | production | 45 |
| 1 | R02 | Prophet-main donor conversion | production | 30 |
| 1 | R04 | loop-vs-depth go/no-go | gate | 24 |
| 1 | R11 | evaluation | production | 19 |
| 1 | R07 | optimiser bake-off (trimmed) | gate | 14 |
| 1 | R01 | byte-frontend MFU probe | gate | 2 |
| 1 | W4 | accuracy-versus-depth sweep | gate | 1 |
| 1 | W2 | multi-key recall versus state and depth | gate | 1 |
| 2 | R03 | two-tier memory | ablation | 20 |
| 2 | R06 | data mixture ablations | ablation | 18 |
| 2 | R02 | hybrid recall gate (MK-NIAH) | gate | 8 |
| 2 | R09 | confidence-probe AUROC probe | gate | 3 |

## Not funded

| Pri | Track | Work | Hours | Why it was cut |
|---:|---|---|---:|---|
| 3 | A2 | agentic training recipe | 67 | below the funding line |
| 3 | R04 | depth ablations | 24 | below the funding line |
| 3 | R08 | quantisation ladder | 20 | below the funding line |
| 3 | R02 | interleave and long-context ablations | 20 | below the funding line |
| 3 | R05 | MoE routing and upcycling | 16 | below the funding line |
| 3 | W1 | halting: input-dependent depth | 12 | below the funding line |
| 3 | R02 | long-context extension | 12 | below the funding line |
| 3 | A2 | per-token depth ceilings versus one depth per sequence | 4 | below the funding line |
| 3 | A4 | depth-disagreement AUROC probe | 1 | below the funding line |
| 4 | R09 | confidence head training | 20 | optional; below the funding line |
| 4 | W4 | depth consolidation | 14 | optional; below the funding line |
| 5 | R01 | byte-frontend retrofit | 36 | optional; below the funding line |
| 5 | R12 | vision adapter | 26 | optional; below the funding line |

## Gates run first

53 hours of gate experiments (18% of the budget) decide whether the expensive work is worth doing at all:

- **R01 — byte-frontend MFU probe** (2 h). Measure realised MFU of patch gather/scatter kernels. Below 0.18 the entire byte-level track is dead, and 2 hours settles it.
  - Failure cancels: R01 byte-frontend retrofit
- **R07 — optimiser bake-off (trimmed)** (14 h). Muon against a properly tuned AdamW. Break-even is a 1.058x speedup; below that the bake-off costs more than it saves and we keep AdamW.
- **W4 — accuracy-versus-depth sweep** (1 h). Does accuracy actually rise with recurrence depth? One hour settles it. Our own R04 puts latent depth at ~1.8 GSM8K points against ~33 for verbalised CoT; if the sweep is flat there is nothing for depth consolidation to store and that track closes before it costs anything.
  - Failure cancels: W4 depth consolidation
- **W2 — multi-key recall versus state and depth** (1 h). The sharpest test of the bounded-state bet. R02 already measured 89.8% single-needle against 37.8% multi-needle; W2 adds that the effective linear-to-attention ratio degrades from 1:1 at k=1 to 8:1 at k=8, so raising the depth dial makes the known weakness worse. One hour shows whether the depth dial and the recall budget are the same dial pulling opposite ways.
- **R04 — loop-vs-depth go/no-go** (24 h). Iso-FLOP comparison of looped depth against plain depth, plus depth generalisation, at >=350M parameters. If looping does not beat equal-FLOP depth, the central architectural bet is dead and everything downstream changes.
  - Failure cancels: R04 depth ablations, R03 two-tier memory
- **R09 — confidence-probe AUROC probe** (3 h). Every published confidence-probe result is at 7B or above. If AUROC at 0.6B is below 0.70 the head is dropped. No training required.
  - Failure cancels: R09 confidence head training
- **R02 — hybrid recall gate (MK-NIAH)** (8 h). Multi-key retrieval is where linear mixers collapse. Gate the interleave ratio on it before committing to the stack.

---

## Calendrier

Le budget de 300 heures-A100 s'étale sur plusieurs semaines de sessions Colab
interruptibles. Une semaine réelle représente environ 25–40 heures-A100 utilisables une
fois déduites les pertes de préemption.

| Semaine | Contenu | Livrable vérifiable |
|---|---|---|
| **1** | Vérification : identifiants de datasets et champs d'architecture des donneurs confrontés au Hub. Portes R01 (MFU), R07 (optimiseur), R09 (sonde AUROC). Harnais d'évaluation R11. | Les deux scripts de vérification passent au vert. Trois verdicts écrits. |
| **2** | Porte R04 (boucle contre profondeur) à ≥ 350M. Porte R02 (rappel multi-clés). | **Décision go/no-go sur le pari central.** |
| **3** | Ablations de mélange de données R06. Reproduction de 4 lignes de base concurrentes dans *notre* harnais. | Mélange figé. Tableau de bord avec des scores que nous avons nous-mêmes mesurés. |
| **4–7** | **Voie A** : pré-entraînement Prophet-mini depuis zéro, phases A et B. | Courbe de perte, checkpoint de plateau, évaluations Tier-1 tous les 3 jours. |
| **6** | **Voie B, en parallèle** : conversion du donneur puis entraînement de récupération. Ne dépend pas de la voie A. | Prophet-main récupéré, comparé au donneur sur le BPB tenu à l'écart. |
| **8** | Phase C : trois recuits branchés depuis le checkpoint de plateau, puis fusion. Ablation mémoire R03. | Modèle de base fusionné. Verdict sur la mémoire persistante. |
| **9–10** | Post-entraînement R10 sur les deux modèles : mid-training raisonnement, SFT bimode, distillation on-policy. | Modèles instruct. |
| **11** | Évaluation Tier-2 complète, quantification et export, carte du modèle. | Résultats publiables avec rapport de décontamination. |

Les deux voies sont indépendantes après la semaine 3, ce qui est délibéré : si la
conversion échoue, la voie A produit quand même un modèle ; si le pré-entraînement de zéro
déçoit, la voie B produit quand même un modèle.

---

## Critères d'échec pré-enregistrés

| Porte | Seuil d'échec | Conséquence |
|---|---|---|
| R01 MFU octet | MFU réalisé < 0.18 | Le track octet est abandonné. BPE 32k à repli octet définitif. |
| R07 optimiseur | Accélération < 1.058× | Muon abandonné, AdamW 8 bits retenu. |
| R04 profondeur bouclée | Pas meilleur qu'une profondeur simple à iso-FLOP | **Le pari central tombe.** Repli sur une pile dense classique ; R03 est annulé en cascade. |
| R09 sonde de confiance | AUROC < 0.70 à 0.6B | Tête de confiance abandonnée. |
| R02 rappel multi-clés | Effondrement sur MK-NIAH | Augmenter le nombre de couches globales — **pas** élargir la fenêtre. |
| W4 exactitude contre profondeur | Courbe plate | La consolidation de profondeur n'a rien à stocker ; le track ferme avant de coûter quoi que ce soit. |
| W2 rappel multi-clés contre *k* | Le rappel se dégrade quand *k* monte | Le cadran de profondeur et le budget de rappel sont le même cadran en sens inverse ; plafonner *k* ou ajouter des couches globales. |
| A2 plafonds de profondeur par token | BPB dégradé de plus de 1 % contre une profondeur par séquence | `recurrent.token_depth` reste hors des configs livrées ; la boucle agentique tourne en régime `fixed` (une profondeur par épisode, la halte ne peut que la baisser). |
| A4 désaccord de profondeur | AUROC < 0.65 sur la suite Tier-1 | Le signal sort du vecteur de caractéristiques du vérificateur ; le second passage profond n'est plus déclenché. |
| Conversion de donneur | Couverture paramétrique < 50 % | Refus automatique : c'est du pré-entraînement à départ chaud, à budgéter comme tel. |
| Vérification des donneurs | Un champ ne correspond pas au Hub | Refus automatique de conversion. Un `head_dim` erroné ne casse pas bruyamment — il laisse des tenseurs en init fraîche et le modèle est simplement moins bon. |

---

## Ce que le plan ne finance pas

Huit demandes sur dix-neuf ne passent pas la ligne de flottaison. La mémoire persistante
(R03), coupée dans la version précédente de ce plan, est désormais **financée en priorité 2**
par décision explicite : c'est la seule capacité qu'aucun concurrent ne possède, et les
55 heures économisées en convertissant plutôt qu'en pré-entraînant Prophet-main sont
précisément ce qui la finance.

Restent coupées, par ordre de ce qu'on reprendrait en premier si du budget se libère :
ablations de profondeur (R04, 24 h), échelle de quantification (R08, 20 h), ablations
d'entrelacement (R02, 20 h), routage MoE (R05, 16 h), extension de contexte long (R02,
12 h), tête de confiance (R09, 20 h), adaptateur vision (R12, 26 h), retrofit octet (R01,
36 h).
