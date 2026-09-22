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

## Amendement 7 — 2026-09-21, pendant v3, avant tout run v4

**Observation.** Sous le taux divisé par quatre (v3), la graine 0 tient H7 sur ses trois
conditions (0,733 → 0,967 sans creux, BPB +0,060). Mais la graine 2 s'effondre au tour 1
(0,567 → **0,017**) d'une façon nouvelle, vue jeton par jeton : le modèle note la bonne
valeur puis **répète `note`** jusqu'au bout des quatre pas sans jamais appeler `done`, ou
meurt dans le corps de `done` (`"summ`). Les 14 trajectoires promues de ce tour sont
vérifiées mais bâclées (notes en double, pas malformés) ; le rendu retire les pas
malformés mais garde les doublons, et le modèle les imite. C'est le mode (b) de docs/32
sous une autre forme : le vérificateur de résultat promeut le processus bâclé.

**Bras `closed-clean`, une seule variable de plus.** Comme `closed`, mais la promotion
exige une trajectoire de **forme canonique** : aucun pas malformé, aucun `done` refusé,
aucun pas identique au précédent, et `done` en dernier. Tout le reste comme v3
(`--lr-scale 0.25`, grammaire compacte, mêmes graines, mêmes tours). Le vérificateur
exécutable reste le juge du résultat ; la forme n'ajoute qu'un filtre de processus, lisible
dans la quarantaine (`demoted_sloppy` par tour).

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H8 forme** | Promouvoir seulement les trajectoires canoniques supprime l'effondrement sans perdre le gain. | (i) aucun tour de `closed-clean` sous succès(tour 0) − 0,10 sur les graines retenues ; (ii) gain moyen de `closed-clean` ≥ celui de `closed` v3 ; (iii) Δ BPB moyen ≤ celui de `closed` v3 + 0,02. Les trois, sinon échec. |

## Amendement 8 — 2026-09-22, après v4, avant tout run v5

**Observation.** H7 et H8 échouent sur la même condition, la même graine et le même tour :
graine 0, tour 1, 0,600 < 0,633. Ce recul est le mode (c) de docs/32 §4 : après
`"args":{`, le modèle ouvre une clé (`"su`, `"{`) au lieu de fermer, et le span meurt. Le
schéma de `done` admettait une clé facultative `summary` que **rien ne lit** (ni le
vérificateur, ni le rendu, qui n'écrit jamais que `{"name":"done","args":{}}`) : encore
une continuation autorisée au décodage et jamais montrée à l'entraînement, comme les
blancs de l'amendement 3. 13–17 % de sorties malformées du modèle amorce viennent de là.

**Correctif, décodage seulement.** `done` n'a plus d'argument ; après `{`, seule `}` est
viable. Test : la grammaire refuse `{"name":"done","args":{"`.

**Pilote v5, une seule variable.** Bras `oracle`, `closed-clean`, `frozen`, taux ÷ 4,
grammaire compacte, mêmes amorces, mêmes tours, `done` sans argument. Le tour 0 change
(le banc du modèle amorce perd ses malformées de mode c), donc les gains se comparent à
leur propre tour 0 et le rendement à l'oracle v5.

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H9 `done`** | Sans continuation fantôme dans `done`, le bras fermé ne recule plus au premier tour et garde son gain. | (i) aucun tour de `closed-clean` sous succès(tour 0) − 0,10 sur les graines retenues ; (ii) part malformée du banc ≤ 5 % à chaque tour ; (iii) rendement gain(closed-clean) / gain(oracle) ≥ 0,8. Les trois, sinon échec. |

## Amendement 9 — 2026-09-22, pendant v5, avant tout run v6 : une seconde famille

**Pourquoi.** Cinq pilotes sur une seule famille (`lookup`) ont isolé et corrigé trois
défauts, une variable à la fois. Ce qui n'est pas établi : que la boucle fermée, avec cette
recette, apprend aussi une famille dont la **forme** est autre. `files` (« quel fichier
contient le mot X ? » : `grep` → `note` du nom de fichier → `done`) exige de lire une
observation à plusieurs lignes et d'en copier la bonne, et démarrait à 7 % avec les
trajectoires parfaites de docs/09 — une marge réelle, et le mur d'amorçage de §2 en vrai.

**Pilote v6.** Famille `files`, recette v5 sans changement (taux ÷ 4, grammaire compacte,
`done` sans argument, promotion canonique), bras `oracle`, `closed-clean`, `frozen`,
5 tours × 30 tâches, banc 2 × 30 (graines 7 et 11), BPB 200 documents. Graines : calibrées
comme à l'amendement 2 (tour 0 par graine, retenues si le départ est strictement entre 0
et 1, dans l'ordre 0, 1, 2, … jusqu'à trois retenues) ; si aucune graine ne démarre
au-dessus de 0, le résultat est « la boucle ne démarre pas à 7 M sur `files` », rapporté
tel quel avec le taux de départ.

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H10 généralité** | La recette qui marche sur `lookup` fait démarrer et progresser la boucle sur `files`. | (i) gain(closed-clean) > 0 avec un intervalle excluant zéro sur chaque graine retenue ; (ii) rendement gain(closed-clean) / gain(oracle) ≥ 0,6 ; (iii) aucun tour sous succès(tour 0) − 0,10. Les trois, sinon échec, et le taux de départ est rapporté. |

## Amendement 10 — 2026-09-22, calibration `files` : saturée à l'amorce

**Observation.** Avec la recette v5 (grammaire compacte, `done` sans argument), l'amorce de
100 trajectoires parfaites / 200 pas donne sur `files` un départ de **1,000 / 0,983 / 1,000**
(graines 0, 1, 2 ; banc 2 × 30). Les 7 % de docs/09 tenaient aux deux défauts de décodage
depuis corrigés. La calibration est arrêtée là : la règle « strictement entre 0 et 1 »
retiendrait la graine 1 avec une tâche de marge sur soixante, ce qui ne teste rien.

**Échelle d'amorce, pré-enregistrée.** Calibration du tour 0 sur les graines 0, 1, 2 pour
`files` à 50 / 100 et 25 / 50 (épisodes / pas), puis `count` à 100 / 200. Le pilote v6
tourne sur la **première** combinaison (famille, amorce) de cette liste où au moins deux
graines démarrent entre 0,10 et 0,90 ; les graines retenues sont celles-là. Si aucune
combinaison ne convient, le résultat est « à 7 M, aucune de ces familles n'offre de marge
à cette amorce », rapporté avec le tableau des départs. H10 reste tel qu'écrit.

**Résultat de l'échelle (2026-09-22).** `files` à 50 / 100 : départs **0,700 / 0,783 / 0,917**
(graines 0, 1, 2). Deux graines entre 0,10 et 0,90 : la combinaison est retenue à la
première marche, les graines 0 et 1 sont celles du pilote v6 ; la graine 2 (0,917) est hors
de la fenêtre et n'est pas retenue. Les marches suivantes (`files` 25 / 50, `count`) ne
sont pas explorées, comme écrit.

## Amendement 11 — 2026-09-22, après v6, avant tout run v7 : pas de pas répété au décodage

**Observation.** Sur `files` (v6), 5 échecs sur 6 du bras `closed-clean` sont une boucle
sur `note` : la bonne valeur est notée, puis `note` est réémis jusqu'au bout du budget sans
`done`. Les trajectoires promues (forme canonique) ne contiennent jamais deux pas
identiques ; le décodeur autorise pourtant de répéter l'action précédente. Troisième
désaccord entraînement / décodage du même type.

**Correctif, décodage seulement, activable.** `AgentConfig.no_repeat_action` (défaut
`False`) : quand il est actif, la grammaire du pas *i* exclut le **nom** de l'action du pas
*i − 1*. Les bras des pilotes l'activent ; l'ancien comportement reste le défaut du code.
Test : la boucle avec l'option ne peut pas enchaîner deux `note`.

**Pilote v7, une seule variable.** `files`, amorce 50 / 100, graines 0 et 1 (les mêmes
poids d'amorce), recette v6 plus `no_repeat_action`. Critère **H11** : H10 réévaluée sous
v7 ((i) intervalles excluant zéro, (ii) rendement ≥ 0,6, (iii) aucun tour sous t0 − 0,10).

## Amendement 12 — 2026-09-22, après v7, avant tout run v8 : le pointeur de copie explore

**Observation.** H11 tient (v7), et les 4 échecs restants de la graine 1 sur `files` sont
tous un pointeur de copie en retard d'un jeton (`chor_0.txt` pour `anchor_0.txt`). Les
cibles d'entraînement sont vérifiées justes (0 cible mal alignée sur 60 valeurs de
trajectoires parfaites rendues) : c'est le pointeur **appris** de cette amorce qui est
biaisé. Or, à la génération, la boucle prend l'**argmax** des deux pointeurs quelle que
soit la température : un pointeur systématiquement décalé ne produit jamais d'épisode
vérifié, et la boucle fermée ne peut pas corriger ce qu'elle ne réussit jamais. Le mur
d'amorçage de §2, au jeton près.

**Correctif, génération seulement, activable.** `AgentConfig.sample_copy` (défaut
`False`) : à la génération (température 0,7), le début et la fin du span copié sont tirés
de leur softmax tempérée au lieu de l'argmax ; le banc reste glouton. `--sample-copy` dans
le protocole, `SAMPLE_COPY=1` dans les scripts. Test : la fonction de choix est l'argmax à
température nulle et explore les candidats proches sinon, sans jamais mettre la fin avant
le début.

**Pilote v8, une seule variable.** `files` 50 / 100, graines 0 et 1, recette v7 plus
`sample_copy`. Critère **H12** : H11 réévaluée sous v8, plus (iv) sur la graine 1, gain ≥
celui de v7 (+0,117) et au plus 1 échec de copie décalée sur les 30 tâches du banc 7.

## Amendement 13 — 2026-09-22, après v8, avant tout run v9 : la graine dure de `lookup` sous la recette de référence

**Pourquoi.** La recette de référence est désormais celle de v7 (taux ÷ 4, grammaire
compacte, `done` sans argument, promotion canonique, pas de pas répété ; `sample_copy`
écarté par v8). Sur `lookup`, la graine 2 (départ 0,567) est la seule où le bras fermé n'a
jamais eu d'intervalle excluant zéro : +0,000 (v1), +0,033 (v2), +0,117 (v3), +0,267 (v4),
+0,100 (v5). Elle n'a pas été rejouée depuis l'amendement 11.

**Pilote v9, une seule variable.** `lookup`, graine 2, amorce 100 / 200 réutilisée (celle
de v3 à v5), bras `closed-clean` seul avec `no_repeat_action`, comparé à l'oracle v5 de la
même graine (+0,300) et au bras `closed-clean` v5 (+0,100 [−0,033, +0,233]).

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H13 graine dure** | La recette de référence fait passer la graine 2 au-dessus de zéro avec certitude. | (i) gain > 0 avec intervalle excluant zéro ; (ii) aucun tour sous 0,467 ; (iii) rendement contre l'oracle v5 ≥ 0,6. Les trois, sinon échec. |
