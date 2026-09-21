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

## Amendement 3 — 2026-09-21, après le bras `closed` de la graine 2, avant le pilote v2

**Observation.** Le bras `closed` s'effondre par les sorties malformées : graine 2,
banc 0,567 → 0,500 → 0,233 (malformées 0 % → 4 % → 45 %), puis remonte à 0,583 ; graine 0,
0,883 → 0,600 au tour 2. Le diagnostic sur le checkpoint du tour 2 (graine 2, mêmes tâches
que le banc) montre le mécanisme, jeton par jeton : le span d'appel s'ouvre par des espaces
(`<|call|>`, ` `, ` `), la grammaire tolérante les admet (elle saute les blancs comme un
lecteur JSON), et deux jetons plus loin aucun des 64 premiers candidats n'est viable : le
span meurt, le pas est « malformé », et quatre pas de budget ne suffisent plus. Le modèle
amorce, lui, ouvre chaque appel par `{"`. Le rendu (`render_episode`) écrit les appels en
JSON compact, sans aucun blanc : la grammaire de décodage admettait donc ce que
l'entraînement n'a jamais montré, et un modèle qui dérive (peu de lignes, rejeu 0,5 de
prose et de code, 60 pas à taux plein) s'engouffre dans l'indentation. C'est le
dix-neuvième défaut silencieux du dépôt : un désaccord entraînement / décodage.

**Correctif, décodage seulement.** `ActionGrammar(compact=True)` par défaut : tout blanc
hors chaîne rend le préfixe mort, donc le premier jeton d'un appel est `{`. `compact=False`
rend l'ancien lecteur tolérant. Tests : la grammaire refuse les blancs hors chaîne et les
accepte dans les chaînes ; le décodeur contraint ne peut pas ouvrir un appel par un blanc ;
chaque appel rendu d'une trajectoire parfaite (`lookup`, `calc`, `files`) est une chaîne
complète pour la grammaire compacte.

**Décision.** Le pilote v1 (grammaire tolérante) va jusqu'au bout, bras KLPO compris, et
ses chiffres sont rapportés tels quels dans docs/32. Puis un **pilote v2** rejoue les quatre
bras sur les graines retenues (0, 2, 3) avec la grammaire compacte et **rien d'autre de
changé** : mêmes amorces (réutilisées), mêmes tours, mêmes critères H1–H5.

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H6 grammaire** | Avec la grammaire compacte, le bras `closed` ne meurt plus par la forme. | Taux de sorties malformées du banc ≤ 5 % à **chaque** tour, sur chaque graine retenue ; et H1 réévaluée sous v2. |

## Amendement 4 — 2026-09-21, bras KLPO v1 arrêté après la graine 0, tour 2

Le bras `closed-klpo` v1 tombe à **0,000** de succès au tour 2 de la graine 0 (0,733 →
0,517 → 0,000 ; malformées 13 % → 31 % → 52 %). Le critère H5 porte sur la moyenne des
graines ; avec −0,733 sur la graine 0, il ne peut plus passer quels que soient les gains des
graines 2 et 3 (au mieux +0,433 et +0,300, moyenne 0,000 < +0,111). Les graines 2 et 3 ne
sont donc pas lancées en v1 ; le verdict v1 de H5 est **échec**, rapporté avec les chiffres
dans docs/32 §6. Le bras KLPO tourne en v2 sur les trois graines, hyperparamètres inchangés.

## Amendement 5 — 2026-09-21, pendant v2, avant tout run v3

**Observation.** Sous la grammaire compacte (v2), le mode (a) a disparu (graine 2, tour 2 :
0,233 → 0,550 à poids égaux), mais le bras `closed` de la graine 2 reste instable
(0,567 → 0,500 → 0,550 → 0,400 → 0,517 → 0,600) et ses échecs sont désormais des copies du
mauvais champ, bien formées : le mode (b). L'oracle connaît le même creux (graine 0,
tour 2) et tous les bras oublient autant (+0,40 à +0,46 bit/octet en cinq tours). Le
suspect commun est la recette par tour : un planning WSD neuf à taux de pointe plein
(Muon 0,01, AdamW 2e-3) pendant 60 pas sur 14–30 lignes, soit 12 à 30 passages par ligne.

**Pilote v3, une seule variable.** `--lr-scale 0.25` : taux de pointe divisés par quatre
pour l'entraînement des tours (pas pour l'amorce, réutilisée). Tout le reste comme v2 :
grammaire compacte, 60 pas, rejeu 0,5, graines 0, 2, 3, quatre bras, mêmes critères
H1–H6. Le bras KLPO garde son propre taux (5e-4) ; seul son fine-tuning par rejet change.

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H7 recette** | Le taux de pointe divisé par quatre supprime les creux et divise l'oubli, sans perdre le gain. | (i) aucun tour du bras `closed` sous succès(tour 0) − 0,10 sur les graines retenues ; (ii) gain moyen de `closed` ≥ celui de v2 ; (iii) Δ BPB moyen de `closed` ≤ la moitié de celui de v2. Les trois, sinon échec. |

Le mode (c) (corps de `done` dès l'amorce, 13–17 % de malformées au tour 0) n'est pas visé
par v3 ; il relève de la taille du modèle et du budget de pas et sera traité à part.

## Amendement 6 — 2026-09-21, bras KLPO v2 arrêté après la graine 0, tour 1 ; v3 change ses hyperparamètres

**Observation.** Sous la grammaire compacte, le bras `closed-klpo` tombe à **0,000** dès le
tour 1 de la graine 0 (v1 : 0,517 puis 0,000) ; malformées 58 % / 57 %, perte KLPO
+1,0 → −10,3 en 60 pas, log p − log q moyen −0,83 nat/token. Le critère H5 ne peut plus
passer en v2 (même arithmétique que l'amendement 4) ; le bras est arrêté et rapporté.

**Diagnostic.** Deux runs, même forme : 60 pas d'AdamW à 5e-4 sur les **mêmes** 40–50
épisodes et leurs tirages auxiliaires enregistrés (« substitut empirique » du rapport),
avec β = 0,1. Le terme de KL est cent fois trop faible pour retenir une dérive de
0,8–1,4 nat par token ; le score centré pousse la politique vers des jetons que
l'échantillonneur n'aurait jamais tirés, et la grammaire ne rattrape pas une
distribution effondrée.

**Décision pour v3.** Le bras `closed-klpo` garde sa place mais avec **β = 1,0 et 20 pas
KLPO** (lr 5e-4 et M = 8 inchangés), plus le `--lr-scale 0.25` commun à tous les bras.
Son critère, H5, reste tel qu'écrit (gain ≥ `closed`, Δ BPB ≤ `closed`) et gagne une
condition de non-effondrement : aucun tour sous succès(tour 0) − 0,10. Deux variables
changent donc pour ce bras entre v2 et v3 ; la comparaison propre de KLPO est
« `closed-klpo` v3 contre `closed` v3 », pas contre v2.
