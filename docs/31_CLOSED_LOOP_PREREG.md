# 31 — Pré-enregistrement : la boucle fermée

**Statut : v0, écrit avant tout run de la boucle. Le pilote CPU à 7 M de paramètres est la
première exécution ; ses chiffres iront dans docs/32. Rien ici n'est un résultat.**
**Date :** 2026-09-21
**Base de code :** branche `claude/codex-results-analysis-5jz95y`.

## 0. La question, et pourquoi elle est différente des précédentes

Les programmes précédents mesurent ce qu'un tronc coûte et ce qu'il porte. Celui-ci
mesure si un modèle **s'améliore de ses propres épisodes**, quand seuls ceux qu'un
programme a vérifiés sont retenus, et **ce que cela coûte** en compute et en oubli.

La boucle, telle que le dépôt la construit (docs/08 §4 bis) :

```
tâches inédites ─► boucle d'agent (portes, outils) ─► vérificateur exécutable
     ─► quarantaine tier 0, promotion immédiate ─► rendu identique au flux de la boucle
     ─► entraînement avec rejeu du corpus de base ─► banc sur tâches jamais vues ─► tour suivant
```

Ce qui existait : chaque étage, exercé à 7 M de paramètres sur des trajectoires parfaites
fournies d'avance (docs/09). Ce qui n'avait jamais tourné : la boucle qui se nourrit
d'elle-même, tour après tour, avec sa comptabilité.

Ce que la littérature a déjà : l'itération d'expert (STaR 2022, ReST-EM 2024) et le
renforcement à récompense vérifiable, à des échelles de 7B et plus, où le modèle de départ
résout déjà une part des tâches. Ce qu'elle n'a pas, à ma connaissance `[S]` : la courbe
**capacité gagnée par heure de compute** d'un modèle de moins d'un milliard de paramètres,
sur un seul GPU, **avec le coût d'oubli mesuré au même endroit** et une référence oracle
au même budget de tâches. C'est cette courbe que ce programme produit.

## 1. Hypothèses et critères, fixés avant le premier tour

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H1 auto-amélioration** | Le bras `closed` gagne en succès sur tâches jamais vues, tour après tour, à partir de ses seuls épisodes vérifiés. | Succès(tour R) − succès(tour 0) > 0 sur 2 graines sur 2, avec un intervalle binomial qui exclut zéro sur le banc cumulé (80 tâches). |
| **H2 rendement** | Le gain du bras `closed` vaut une fraction stable de celui de l'oracle au même budget de tâches. | Rapport [gain closed] / [gain oracle] rapporté par tour ; pas de seuil, c'est la mesure. |
| **H3 coût d'oubli** | L'entraînement sur ses propres épisodes avec rejeu ne dégrade pas la langue plus que l'oracle. | Δ BPB(tour R − tour 0) du bras `closed` ≤ celui de l'oracle + 0,05 bit/octet. |
| **H4 compute** | La capacité gagnée par heure de compute (génération + vérification + entraînement) est décroissante mais positive. | Courbe rapportée ; critère : gain par heure du dernier tour > 0. |
| **H5 KLPO** (ajouté le 2026-09-21, avant que le bras ne tourne) | Ajouter, au fine-tuning par rejet, des pas KLPO sur **tous** les épisodes du tour (docs/research/A5_klpo.md) rapporte au moins autant en succès et dérive moins en langue. | Gain(`closed-klpo`) ≥ gain(`closed`) sur la moyenne des graines, **et** Δ BPB(`closed-klpo`) ≤ Δ BPB(`closed`). Les deux, sinon échec. |

Un échec de H1 est un résultat : il dit que le taux de succès de départ ne suffit pas
pour que la boucle démarre, et à quel taux. Le programme ne se « répare » pas en
changeant un critère après coup.

## 2. Bras, tous à partir du même modèle amorcé

| Bras | Épisodes ajoutés par tour | Entraînement |
|---|---|---|
| `closed` | ceux que le modèle a produits **et** qu'un programme a vérifiés (tier 0) | oui, sur tous les promus, avec rejeu |
| `oracle` | la trajectoire parfaite de chaque tâche du tour | identique |
| `frozen` | aucun | aucun |
| `closed-klpo` | comme `closed`, plus **tous** les épisodes du tour avec leur récompense 0/1 et les enregistrements de l'échantillonneur | fine-tuning par rejet identique à `closed`, puis K pas KLPO (β = 0,1, M = 8, température 1,0 à la génération) |

L'amorce est commune : le poids de base affiné sur peu de trajectoires parfaites de la
famille, juste assez pour que le succès de départ soit non nul. Un modèle à zéro ne
produit rien à apprendre ; c'est le mur d'amorçage de l'itération d'expert, et il est
mesuré ici plutôt que contourné.

Chaque tour : *N* tâches inédites (le générateur est la seule source de tâches, une graine
est un découpage), *A* tentatives par tâche, *S* pas d'entraînement, puis le banc sur deux
jeux fixes de tâches jamais entraînées (graines 7 et 11, décodage glouton) et les bits par
octet tenus à l'écart. Les graines des tâches de tours, d'amorce et de banc sont disjointes
par construction (`scripts/closed_loop.py`).

## 3. Échelle et budget

| Échelle | Modèle | Famille | Tours | Coût estimé |
|---|---|---|---|---|
| Pilote CPU | 7 M (docs/09), 4 cœurs | `calc` (60–80 % avec trajectoires parfaites à cette taille) — **faux, voir amendement 1 : `lookup`** | amorce 100 épisodes / 200 pas ; 5 tours × 30 tâches, 2 tentatives, 60 pas, rejeu 0,5 ; banc 2 × 30 | ≈ 25 min par bras et par graine (`scripts/closed_loop_cpu_pilot.sh`) |
| A100 | 375 M du programme docs/29, têtes d'action ajoutées à l'amorce | `calc`, puis `lookup` et `files` | 8 × 200 tâches, 300 pas | ≈ 1 h par bras et par graine |

Le pilote CPU n'est pas une revendication de capacité : il vérifie que la boucle tourne,
que la comptabilité est juste, et donne une première courbe. L'échelle A100 dépend d'un
poids du programme docs/29 et n'est pas lancée avant.

## 4. Mesures, par tour

- succès sur les deux bancs tenus à l'écart, moyenne et par graine ;
- tokens par succès, taux d'appels malformés, valeurs copiées ;
- bits par octet sur le texte tenu à l'écart (l'oubli) ;
- épisodes générés, tentatives, promus (nouveaux et cumulés) ;
- secondes de génération, d'entraînement, cumul du compute ;
- résumé de la quarantaine par tier.

## 5. Ce qui n'est pas dans ce programme

- Aucune mémoire consolidée hors des poids : la voie « registre plutôt que gradient » de
  docs/06 reste une ablation à part, une fois que la boucle par gradient a sa courbe.
- Aucune vérification apprise : seul le tier 0 promeut. Un vérificateur appris à
  AUROC 0,80 laisserait entrer 30 % de faux (docs/08 §4).
- Aucun transfert entre familles ; une famille à la fois, la suivante après.
- Aucune revendication de raisonnement, d'assistant ou d'AGI.

## 6. Pièces

| Pièce | Fichier | Test |
|---|---|---|
| Pilote : tours, bras, comptabilité, reprise par tour, protocole figé | `scripts/closed_loop.py` | `tests/test_closed_loop.py` |
| Perte KLPO, enregistrement de l'échantillonneur, bras `closed-klpo` | `prophet/train/klpo.py`, `prophet/agent/loop.py`, `scripts/closed_loop_cpu_pilot_klpo.sh` | `tests/test_klpo.py` |
| Boucle d'agent, quarantaine, rendu, banc (existants) | `prophet/agent/`, `prophet/eval/agent_bench.py` | suite existante |
| Poids de base 7 M | `scripts/first_run_cpu.py` | docs/09 |

## Amendement 1 — 2026-09-21, avant tout tour d'entraînement

**Observation.** Le pilote CPU tel que pré-enregistré (famille `calc`, amorce 100 épisodes /
200 pas, graine 0) donne un succès de **1,0 au tour 0** : 60 tâches sur 60, sur les deux bancs
(graines 7 et 11), 0 % de sorties malformées, 267 tokens par tâche. La prédiction de §3
(60–80 %) était fausse. Un départ à 100 % rend H1 (gain > 0) impossible par construction :
le run a été arrêté avant tout tour d'entraînement ; son tour 0 est conservé et sera rapporté
dans docs/32.

**Calibration** (graine 0, mêmes bancs, sans BPB, `--rounds 0`) :

| Famille | Amorce (épisodes / pas) | Succès tour 0 (banc 7 / banc 11) | Malformés | Pas d'amorce (s) |
|---|---|---|---|---|
| `calc` | 10 / 20 | 0,00 / 0,00 | 100 % | 71 |
| `calc` | 20 / 40 | 0,00 / 0,00 | 72 % / 64 % | 131 |
| `calc` | 40 / 80 | 0,00 / 0,00 | 0 % — bien formées, fausses | 270 |
| `calc` | 100 / 200 | 1,00 / 1,00 | 0 % | 747 |
| `lookup` | 100 / 200 | 0,67 / 0,80 | 17 % / 10 % | 639 |

À cette taille, `calc` n'a pas d'amorce donnant un départ non dégénéré : le succès passe de
0 à 100 % entre 40/80 et 100/200, un seuil, donc un départ instable d'une graine à l'autre.
`lookup` satisfait le but déclaré de l'amorce (« juste assez pour que le succès de départ soit
non nul », §2) au budget pré-enregistré, avec 27 points de marge et une part de sorties
malformées que des épisodes vérifiés peuvent corriger.

**Décision.** La famille du pilote CPU devient **`lookup`**. Tout le reste est inchangé :
amorce 100 / 200, 5 tours × 30 tâches × 2 tentatives, 60 pas, rejeu 0,5, bancs 2 × 30,
BPB 200 documents, les quatre bras, et les critères H1–H5 tels qu'écrits. Les scripts
prennent la famille par `FAMILY=lookup` ; le run abandonné et la calibration sont des
résultats et figurent dans docs/32.

## Amendement 2 — 2026-09-21, après la graine 0, avant la seconde graine

**Observation.** Sur `lookup`, l'amorce de la graine 1 (100 épisodes / 200 pas) donne elle
aussi **1,0 au tour 0** (60/60, 0 % malformées) ; son run a été arrêté après ce tour 0. Le
départ dépend fortement des 100 tâches tirées pour l'amorce :

| Graine | Succès tour 0 (banc 7 / banc 11) | Malformés | Statut |
|---|---|---|---|
| 0 | 0,667 / 0,800 | 17 % / 10 % | utilisée, trois bras terminés |
| 1 | 1,000 / 1,000 | 0 % | saturée, exclue de H1 |
| 2 | 0,500 / 0,633 | 0 % | utilisée |
| 3 | 0,633 / 0,767 | 5 % / 3 % | utilisée |

**Règle, fixée maintenant.** Une graine dont l'amorce démarre à 0 ou à 1 ne peut pas tester
H1 ; elle est rapportée mais exclue du critère. Les graines sont prises dans l'ordre
0, 1, 2, 3, … et retenues si leur départ est strictement entre 0 et 1, jusqu'à en avoir au
moins deux ; la calibration s'arrête là (graines 2 et 3, sans regarder au-delà). Le critère
H1 devient « gain > 0 sur **toutes** les graines retenues », soit trois ici, avec l'intervalle
qui exclut zéro sur le banc cumulé de chacune. Les poids d'amorce calibrés sont réutilisés
tels quels par le pilote (entraînement d'amorce déterministe, même sortie).
