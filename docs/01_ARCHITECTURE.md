# 01 — Architecture Prophet

> Spécification et **registre de décisions**. Chaque choix cite le track de recherche qui
> le fonde, et chaque conflit entre tracks est arbitré explicitement plutôt que moyenné.
>
> Statut : v0.2 — les configurations sont produites par `scripts/design_search.py`, pas
> posées à la main. Les décisions marquées **[ABLATION]** ne sont pas acquises : elles
> attendent leur expérience de validation.

---

## 1. La forme du modèle

```
             ┌──────────────────────── prélude (attention) ───────────────────────┐
tokens ─────▶│  bloc SWA  │  bloc attention complète (NoPE)  │ ...                │
             └───────────────────────────────┬────────────────────────────────────┘
                                             │  h₀ (injecté à chaque tour)
                              ┌──────────────▼───────────────┐
                              │   CŒUR PARTAGÉ (récurrent)   │  ◀── bouclé k fois
                              │   GDN + FFN, poids uniques   │      k réglé à l'exécution
                              └──────────────┬───────────────┘
                                             │
             ┌───────────────────────────────▼────────────────────────────────────┐
             │  coda (attention)  │  norme  │  tête LM  │  tête MTP  │  confiance  │
             └────────────────────────────────────────────────────────────────────┘
```

Profondeur effective = `prélude + cœur × k + coda`.
Profondeur payée en mémoire = `prélude + cœur + coda`.

---

## 2. Décision 1 — L'attention ne va pas dans le cœur bouclé

**Le problème.** Une couche d'attention bouclée *k* fois a besoin de *k* caches KV
distincts : à l'itération *i*, les requêtes diffèrent de celles de l'itération *j*.
Boucler l'attention multiplie donc la mémoire KV par *k*. R04 signale que
l'implémentation de référence de Huginn fait précisément cela et appelle ça
« le piège de déploiement ».

**Deux solutions existent.** R04 propose le *partage du KV entre étapes* : calculer K,V à
la première itération seulement, recalculer les requêtes ensuite. Nous retenons une
solution plus simple et strictement bornée : **le cœur bouclé ne contient que des
mélangeurs à état borné** (gated DeltaNet), et l'attention est confinée au prélude et au
coda, où elle s'exécute exactement une fois.

**Vérification** (`tests/test_modeling.py::test_attention_cache_does_not_grow_with_recurrence_depth`) :

| | k = 1 | k = 8 |
|---|---:|---:|
| Octets de cache d'attention | 10 240 | **10 240** |
| Octets d'état récurrent | 11 264 | 90 112 |
| État récurrent à contexte × 8 | — | **inchangé** |

L'attention ne coûte rien de plus en profondeur ; l'état récurrent croît avec *k* mais
reste une matrice de taille fixe, indépendante de la longueur du contexte.

> **[ABLATION A-KV]** Le partage du KV entre étapes de R04 permettrait de l'attention
> dans le cœur. À comparer une fois le socle validé — mais pas avant, car cela
> réintroduit un couplage entre profondeur et mémoire.

---

## 3. Décision 2 — Taille du modèle : arbitrage R04 contre R07

**Le conflit.** R04 spécifie 9.48B paramètres au total. R07 calcule que l'état statique
d'entraînement d'un MoE de 10B occupe **97 Gio contre 77 Gio utilisables** sur un A100 —
et recommande ≤ 4B au total. Les deux ne peuvent pas être vrais.

**Arbitrage : R07 l'emporte**, pour deux raisons indépendantes.

1. *Contrainte dure contre préférence.* La mémoire d'un A100 n'est pas négociable ; la
   taille cible de R04 l'est.
2. *Le compute-optimal est déjà en dessous.* Avec 300 heures-A100, notre planificateur
   place l'optimum vers 1B de paramètres pour 20B tokens. À 1.3B actifs on ne peut
   acheter que ~12B tokens, soit 0.5× Chinchilla — nous serions dans un régime de
   sous-entraînement sévère avant même de parler de mémoire.

**Résultat de `scripts/design_search.py`** — configurations satisfaisant *simultanément*
la mémoire d'entraînement, le budget de tokens, la mémoire de l'appareil cible et
l'absence de mauvaise allocation :

| Configuration | Total | Actifs | Prof. eff. | Tokens | Tok/actif | Entraînement | Appareil |
|---|---:|---:|---:|---:|---:|---:|---:|
| **prophet-main** `d1536 p4c4×4o4 e128` | **3.79B** | **369M** | 24 | 24.6B | 31 | 38.1 GB | 2.59 GB (5090) |
| **prophet-mini** `d1280 p3c4×2o3` | **229M** | 211M | 14 | 58.3B | 173 | 3.4 GB | 0.67 GB (iPhone) |

Le rapport de sparsité de 10.3× est ce qui rend la chose intéressante : la capacité de
connaissance d'un modèle de 3.8B pour le coût par token d'un modèle de 369M.

---

## 4. Décision 3 — La pile de mélangeurs

Fondée sur R02, qui recense les architectures hybrides ayant réellement été livrées.

| Section | Motif | Justification |
|---|---|---|
| Prélude | `SWA(2048)` puis `attention complète (NoPE)` | La fenêtre glissante capte la structure locale ; une couche globale sans encodage positionnel donne l'extrapolation en longueur. |
| Cœur (bouclé) | `GDN` uniquement | État borné → boucler ne coûte pas de mémoire de contexte (§2). |
| Coda | `SWA(2048)` puis `attention complète (NoPE)` | Rappel exact avant la lecture finale. |

Paramètres d'attention (identiques sur toutes les couches d'attention, ce qui les rend
**compatibles avec un donneur** Llama-3.2/Qwen3 — voir §7) :

| Champ | Valeur | Raison |
|---|---|---|
| `head_dim` | 128 | Correspond aux donneurs ; bon pour les noyaux. |
| `n_kv_heads` | `n_heads / 8` (GQA) | Réduction du cache KV de 8×, point sûr éprouvé. |
| `qk_norm` | activé | Stabilise sans plafonnement de logits, **et** supprime les valeurs aberrantes d'activation qui ruinent la quantification des petits modèles (R08). |
| Puits d'attention | 1 par couche | Sans lui, l'attention fenêtrée s'effondre en streaming. |
| Positions | RoPE sur SWA, **NoPE** sur les couches globales | R02 : l'extension de contexte devient quasi gratuite. |

**Risque principal identifié par R02, à surveiller** : l'effondrement du rappel
multi-clés. Le meilleur mélangeur linéaire de 2026 atteint 89.8 % sur aiguille unique
mais **37.8 % sur aiguilles multiples**. L'hybridation corrige partiellement ; le
correctif en cas d'échec est **plus de couches globales, pas une fenêtre plus large**.

---

## 5. Décision 4 — La profondeur comme cadran d'exécution

Le pari central (R04). Un cœur à poids partagés appliqué *k* fois.

| | iPhone 17 Pro | Mac Studio | RTX 5090 |
|---|---:|---:|---:|
| `k` par défaut | 2 | 4 | 4–8 |
| Profondeur effective (main) | 16 | 24 | 24–40 |

**Mécanismes retenus :**

- **Injection de l'entrée à chaque tour.** Sans elle, une boucle profonde dérive loin du
  prompt. Effet secondaire vérifié par test : elle rétablit un chemin de gradient vers le
  prélude même sous troncature agressive — c'est une propriété voulue, pas une fuite.
- **Init aléatoire de l'état à l'entraînement, déterministe à l'inférence.** L'init
  aléatoire force la boucle à converger vers un attracteur indépendant du point de
  départ, ce qui est *précisément* ce qui rend *k* réglable après coup. Mais elle doit
  s'arrêter à l'inférence : mesuré, deux appels identiques différaient de 0.67 en logits
  et l'équivalence préremplissage/décodage était rompue.
- **Rétropropagation tronquée** aux 3 dernières itérations : entraîner une boucle
  profonde coûte la mémoire d'activation d'une pile courte.
- **Profondeur échantillonnée par pas** (log-uniforme) : le modèle reste utilisable à
  *toute* profondeur.

> **[ABLATION A1]** R04 est explicite : *Mixture-of-Recursions sous-performe le vanilla à
> 135M et ne gagne qu'à partir de 360M.* Nos ablations à 130M peuvent donc produire un
> **faux négatif**. Le test go/no-go doit être conduit à ≥ 350M.

---

## 6. Décision 5 — Ce que nous ne construisons pas en v1

| Composant | Décision | Fondement |
|---|---|---|
| Frontend octet / sans tokenizer | **Non en v1** | R01 : à l'échelle mini le frontend octet tourne à **0.44× le débit**. Vocabulaire BPE de 32 768 avec repli octet, chiffres isolés, tokens d'indentation. Le frontend octet reste un *retrofit* ultérieur, sous ablation. |
| Multimodalité | **Non en v1, ancrages oui** | R12 : la vision n'ajoute aucun point aux benchmarks qui décident de notre victoire, mais les ancrages coûtent ~0.5M paramètres et zéro FLOP. |
| Audio | **Non** | R12 : ASR en pipeline (modèle externe distillé), pas de modélisation audio native. |
| Mémoire persistante | **Ancrée, activable** | R03 : conception à deux étages retenue, mais le verrou est prouvé par une ablation dédiée avant intégration. |

**Ancrages multimodaux implémentés en v1** (R12, hooks H1–H10) : 256 identifiants de
vocabulaire réservés, embeddings typés par modalité, RoPE à dimensions multiples
(dégénéré en 1-D aujourd'hui), masque d'attention comme objet de première classe
supportant les segments bidirectionnels, chemin `inputs_embeds`, cache dimensionné à
l'exécution.

---

## 7. La question stratégique ouverte : partir de zéro ou convertir un donneur

**Quatre tracks indépendants sont arrivés à la même conclusion**, sans se coordonner :

| Track | Formulation |
|---|---|
| R01 | Le budget ne permet pas un frontend octet entraîné de zéro ; retrofit d'un checkpoint. |
| R02 | « 300 heures-A100 ≈ 5.6B tokens à 1.2B actifs — nous ne pouvons pas surpasser Qwen3 en pré-entraînement. La conversion de donneur est le chemin réaliste. » |
| R04 | « Nous sommes ~2 500× sous le compute de pré-entraînement de Qwen3-1.7B. La récurrence achète de la profondeur, pas de la connaissance. » |
| R07 | ≤ 4B au total, atteint par *upcycling* d'un checkpoint dense bien entraîné plutôt que depuis une initialisation aléatoire. |

Notre propre calculateur, écrit indépendamment, donne le même verdict : **73× à 7 326×**
sous le compute des concurrents.

**Ce que « conversion de donneur » signifie concrètement.** Partir de poids ouverts
(Qwen3 en Apache-2.0, par exemple), remplacer la majorité des couches d'attention par des
couches GDN, et poursuivre l'entraînement sur ~2–5B tokens pour récupérer la qualité. La
littérature 2026 sur la linéarisation rapporte des coûts de cet ordre. On hérite alors de
36 000 milliards de tokens de connaissance et on n'achète que l'architecture.

**La tension.** Le cahier des charges dit « repartir de A à Z ». Deux lectures :

- *Architecture de zéro* — compatible avec la conversion de donneur.
- *Poids de zéro* — incompatible.

**Ce que cela change.** La lecture retenue modifie le calendrier, le budget de tokens,
les scores atteignables et le positionnement du projet. Tout le reste — architecture,
pipeline de données, infrastructure d'entraînement, évaluation — est **commun aux deux
voies** et se construit sans attendre la réponse.

**Recommandation.** Une voie mixte : *entraîner Prophet-mini de zéro* (229M, faisable et
honnête, prouve l'architecture), *et* produire Prophet-main par conversion de donneur.
Les deux partagent l'architecture ; seule l'initialisation diffère. Cela donne une preuve
scientifique de l'architecture **et** un modèle compétitif.

---

## 8. Traçabilité des décisions

| # | Décision | Track | Statut |
|---|---|---|---|
| D1 | Cœur bouclé récurrent uniquement, attention hors boucle | R02, R04 | **Acquis** (test) |
| D2 | ≤ 4B total / ~370M actifs | R07, planificateur | **Acquis** (mémoire) |
| D3 | Hybride GDN 3:1 avec SWA + globale NoPE | R02 | **Acquis** |
| D4 | Profondeur réglable à l'exécution | R04 | **[ABLATION A1] à ≥ 350M** |
| D5 | Init d'état déterministe à l'inférence | interne | **Acquis** (test) |
| D6 | BPE 32k à repli octet, pas de frontend octet en v1 | R01 | **Acquis** |
| D7 | Ancrages multimodaux seulement | R12 | **Acquis** |
| D8 | Mémoire persistante à deux étages | R03 | **[ABLATION E2]** |
| D9 | Tête de confiance + abstention | R09 | **[ABLATION A1-R09]** |
| D10 | Zéro contre conversion de donneur | R01/R02/R04/R07 | **Ouvert — §7** |
