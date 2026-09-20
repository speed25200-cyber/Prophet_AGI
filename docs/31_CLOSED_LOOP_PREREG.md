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

Un échec de H1 est un résultat : il dit que le taux de succès de départ ne suffit pas
pour que la boucle démarre, et à quel taux. Le programme ne se « répare » pas en
changeant un critère après coup.

## 2. Bras, tous à partir du même modèle amorcé

| Bras | Épisodes ajoutés par tour | Entraînement |
|---|---|---|
| `closed` | ceux que le modèle a produits **et** qu'un programme a vérifiés (tier 0) | oui, sur tous les promus, avec rejeu |
| `oracle` | la trajectoire parfaite de chaque tâche du tour | identique |
| `frozen` | aucun | aucun |

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
| Pilote CPU | 7 M (docs/09), 4 cœurs | `calc` (60–80 % avec trajectoires parfaites à cette taille) | amorce 100 épisodes / 200 pas ; 5 tours × 30 tâches, 2 tentatives, 60 pas, rejeu 0,5 ; banc 2 × 30 | ≈ 25 min par bras et par graine (`scripts/closed_loop_cpu_pilot.sh`) |
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
| Boucle d'agent, quarantaine, rendu, banc (existants) | `prophet/agent/`, `prophet/eval/agent_bench.py` | suite existante |
| Poids de base 7 M | `scripts/first_run_cpu.py` | docs/09 |
