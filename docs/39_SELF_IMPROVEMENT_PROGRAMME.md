# 39 — Le programme d'auto-amélioration : les hypothèses développées, et la première sur CPU

**Statut : v0, 2026-09-23.** Ce document développe, pour la partie « s'améliore lui-même
sans plafond fixé d'avance » du but (docs/35, conditions C2 à C5), chaque hypothèse avec
son protocole, son critère, son coût et le lieu où elle tourne. **SI-1 est pré-enregistrée
ici et se lance sur CPU** ; les autres sont écrites au niveau où un pré-enregistrement
daté les figera avant leur lancement.

## 0. Ce que « s'améliorer sans plafond » peut mesurer ici

Trois courbes, tour après tour, à compute compté (docs/35 §2) :
- **(a)** le banc du générateur ;
- **(b)** un banc hors de sa distribution ;
- **(c)** l'ensemble jamais réussi.

« Sans plafond fixé d'avance » veut dire que (b) monte encore quand les tâches ne viennent
plus d'un humain. Ce qu'on sait déjà :
- (a) monte (+0,256 à 7 M, docs/32 §9) ;
- (c) baisse quand la reprise explore (18 → 6, §16) ;
- (b) n'a **jamais** été mesurée dans une boucle où le modèle propose ses tâches : le
  proposeur `lookup` ne démarre pas à 7 M (docs/32 §22–23).

## 1. Les hypothèses

| | Hypothèse | Mesure et critère | Où, coût | Dépend de |
|---|---|---|---|---|
| **SI-1** | **Une spécification à un seul champ fait démarrer le proposeur à 7 M, et proposer mène au-delà du générateur.** Famille `calc` : le modèle propose `{"expression": …}`, l'exécuteur calcule la réponse. | §2 ci-dessous | CPU, ≈ 3–4 h | rien |
| **SI-2** | La boucle fermée tient à 375M ce qu'elle fait à 7 M. | gain et rendement contre l'oracle (docs/31 H1–H4), ensemble jamais réussi, oubli, avec la recette de référence (docs/31 amendements 23–25) | A100, ≈ 8 tours × 3 familles, calibration comprise | programme 1 |
| **SI-3** | Le proposeur `lookup` démarre à 375M en format objet, et le pointeur apprend à désigner une clé si ses cibles en contiennent. | règle de docs/33 amendement 1 ; puis H20–H25 | A100 | SI-2 |
| **SI-4** | Le désaccord entre profondeurs signale l'erreur, et règle la reprise. | A4-0 (AUROC ≥ 0,70) puis, s'il passe, « reprendre plus profond » contre « reprendre pareil » à coût égal | A100, minutes puis ≈ 3 h | programme 1 |
| **SI-5** | Ce que la boucle apprend à 375M se transmet à un modèle plus petit sans perdre le gain. | épisodes promus par SI-2 → affinage du modèle embarqué ; part du gain gardée par paramètre | A100 | SI-2 |
| **SI-6** | Une mémoire sans gradient (registre, R03) garde ce que la boucle a appris sans oubli par omission. | docs/32 §15 : une famille absente d'un tour tombe à 0 ; avec registre, ≥ 0,9 de sa valeur | A100 (E2 ≈ 5 h) | R03 |
| **SI-7** | Élargir le vérifiable : des familles dont le programme sait vérifier plus (plusieurs opérations, tables, programmes courts). | chaque famille nouvelle entre par un banc hors distribution et une calibration (docs/31 amendement 20) | CPU pour la mécanique, A100 pour la mesure | SI-1 |
| **SI-8** | La pente (b) reste positive sur *n* tours sans tâche humaine. | pente de régression de (b) sur les tours ≥ 0 avec intervalle au-dessus de 0 ; nommer le mur quand elle s'annule | A100 | SI-1 à SI-3 |

## 2. SI-1, pré-enregistrée : proposer des calculs

**Pourquoi `calc`.** Le proposeur `lookup` échoue sur un format à plusieurs champs : deux
listes alignées, puis une clé à désigner (docs/32 §23). Une expression est un seul champ,
et sa réponse est calculée par l'exécuteur (`_safe_calc`), jamais fournie par le modèle.
Le générateur humain ne produit que `a op b` (deux entiers de 10 à 998, un opérateur) :
tout ce qui va au-delà, en nombre d'opérateurs ou en taille d'entiers, est hors de sa
distribution.

**Mécanique.**
- **Spécification** : `{"expression": "<e>"}`. **Règles** : entiers de 1 à 4 chiffres, 1
  à 3 opérateurs parmi `+ - *`, espaces simples, au plus 32 caractères.
- **Tâche dérivée** : le but du générateur (« Compute *e* with the calc tool, note the
  result, then finish. »), la réponse `_safe_calc(e)`, le vérificateur habituel.
- **Nouveauté (H25)** : au moins deux opérateurs, ou un entier hors de 10–998.
- **Banc hors distribution** : 2 × 30 expressions à **trois opérandes** (`a op b op c`,
  entiers de 10 à 998), graines 17 et 19. Le générateur n'en produit jamais, et aucun bras
  n'y a accès.

**Amorce et calibration.** Même recette que docs/33 : premier temps 100 trajectoires / 200
pas ; second temps *P* propositions parfaites et *P* trajectoires, en barreaux 50 / 50,
100 / 50, 100 / 25, 200 / 100. `calc` sature son banc dès l'amorce (docs/32 §0). La
fenêtre du solveur se lit donc sur le banc **hors distribution** : strictement entre
0,05 et 0,95, avec un succès canonique > 0 sur le banc du générateur. S'y ajoute une
sonde de 30 propositions, valides à ≥ 0,5. Le banc du générateur est jugé sur ce qu'il
garde (docs/31 amendement 20).

**Bras**, si un barreau passe : `closed-propose`, `closed-clean`, `oracle`, 5 tours,
graine 0, recette de référence (taux ÷ 4, rejeu 0,5, trois tentatives dont deux reprises
qui explorent, `no_repeat_emitted`).

**Hypothèses et critères**, fixés avant le lancement :

| | Critère |
|---|---|
| **H20c validité** | ≥ 50 % des propositions valides à chaque tour |
| **H21c bord** | sur ≥ 3 tours sur 5, les propositions valides sont résolues à un taux entre 0,2 et 0,8 |
| **H22c garde** | banc du générateur : aucune perte de plus de 0,05 entre le tour 0 et le tour 5, pour `closed-propose` |
| **H23c portée** (celle qui compte) | gain sur le banc hors distribution : `closed-propose` > `closed-clean`, intervalle (60 tâches) excluant zéro |
| **H24c oubli** | Δ BPB(`closed-propose`) ≤ Δ BPB(`closed-clean`) + 0,02 |
| **H25c nouveauté** | ≥ 50 % des propositions valides sont nouvelles |

**Lecture pré-écrite.**
- Si H23c passe, c'est la première mesure dans ce dépôt d'une boucle qui va au-delà de son
  générateur humain.
- Si H20c passe mais pas H25c, le proposeur recopie la forme de son amorce, et la portée ne
  peut pas venir de lui.
- Si aucun barreau ne passe, un champ unique ne suffit pas non plus à 7 M, et SI-1 passe à
  l'A100 avec SI-3.

## 3. Résultats

### SI-1, 2026-09-23 (graine 0, après l'amendement 1)

Calibration : le premier barreau (50 pas, *P* = 50) passe.

| Mesure | Valeur |
|---|---|
| banc du générateur | 1,0 |
| succès canonique | 1,0 |
| banc hors distribution | 0,783 |
| propositions valides | 28 sur 30 |
| propositions malformées | 0 |

Trois bras de 5 tours, ≈ 20 min de compute chacun. Script de lecture en brouillon ; il
réutilise `summarize_closed_loop.py`, et tous les chiffres sont ici.

| | Critère | Mesure | Verdict |
|---|---|---|---|
| H20c | ≥ 0,5 valides à chaque tour | 0,97 ; 0,97 ; 0,97 ; 0,93 ; 0,87 | **passe** |
| H21c | ≥ 3 tours sur 5 résolus entre 0,2 et 0,8 | 0,83 ; 1,0 ; 1,0 ; 0,93 ; 1,0 — aucun tour | **échoue** |
| H22c | banc du générateur, perte ≤ 0,05 | 1,0 aux six mesures | **passe** |
| H23c | gain hors distribution, `closed-propose` > `closed-clean`, intervalle > 0 | **−0,100** [−0,200 ; −0,017] (1 gagnée, 7 perdues) contre −0,033 [−0,083 ; 0] | **échoue** |
| H24c | ΔBPB ≤ témoin + 0,02 | +0,095 contre +0,095 | **passe** |
| H25c | ≥ 50 % de nouvelles | **4,3 %** (6 sur 141) | **échoue** |

Banc hors distribution tour par tour (60 tâches) :

| Bras | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| `closed-propose` | 0,783 | 0,767 | 0,717 | 0,717 | 0,717 | **0,683** |
| `closed-clean` | 0,783 | 0,733 | 0,767 | 0,750 | 0,750 | 0,750 |
| `oracle` | 0,783 | 0,717 | 0,750 | 0,717 | 0,733 | 0,750 |

**La lecture pré-écrite s'applique.** H20c passe, H25c échoue : le proposeur recopie la forme
de son amorce, deux opérandes à chaque proposition, et la portée ne peut pas venir de lui.
Proposer fait même **reculer** le banc hors distribution (−0,10, intervalle excluant zéro).
Les deux témoins restent dans le bruit (−0,033). S'entraîner sur des tâches à deux
opérandes, les siennes ou celles du générateur, n'apprend rien sur trois.

**Pourquoi le proposeur n'a jamais bougé : sa récompense n'a jamais été versée.**
- **Aucune** proposition n'a été résolue à une reprise en 5 tours : 134 résolues du premier
  coup, 7 jamais. La seule récompense du proposeur (docs/33 §2 : résolue à une reprise, pas
  du premier coup) ne s'est donc jamais déclenchée. Aucune proposition n'a été promue, et
  la distribution du proposeur est restée celle de l'amorce.
- Le diagnostic en lecture seule du banc hors distribution au tour 0 compte 13 échecs sur
  60, de deux sortes, **deux longueurs de copie** apprises sur le générateur :
  - (a) **expression tronquée**, 8 cas : `calc("41 - 250")` pour `41 - 250 - 56`. Le
    pointeur s'arrête après le deuxième opérande, la seule longueur qu'il ait vue ;
  - (b) **résultat tronqué**, 5 cas : la note `24557863` pour `245578632`. La copie de
    l'observation perd le dernier chiffre des résultats longs.
- Or l'exploration des reprises (docs/31 amendement 15) tire le **début** du span parmi
  les 3 meilleurs ; la **fin** reste gloutonne. Sous `--copy-explore observations`, elle
  ne joue même que sur les spans lus dans une observation, et l'expression de `calc` est
  copiée du but. Une reprise rejoue donc exactement la même troncature.

  Ce n'est pas un défaut silencieux : l'exploration a été conçue pour `lookup`, où
  l'erreur est la valeur désignée (le début). Mais sous cette recette, la récompense du
  proposeur `calc` ne pouvait pas être versée.

Ce que SI-1 établit à 7 M :
- un champ unique fait démarrer le proposeur (0 → 28 valides sur 30, contre au mieux 12
  pour `lookup`) ;
- la boucle de proposition tourne sans nuire au générateur ni à la langue ;
- elle **ne va pas au-delà du générateur**, parce que rien ne récompense un pas au-delà :
  un proposeur dont la récompense ne peut pas être versée n'apprend rien.

## Amendement 1 — 2026-09-23, après le premier barreau, avant tout tour entraîné : deux défauts

Le premier barreau (50 pas, *P* = 50, graine 0) est mort, et pour deux raisons sans rapport
avec la question posée. Les deux sont des défauts du dépôt, corrigés avec leurs tests. Le
barreau ne compte pas.

**1. La grammaire laissait un nom d'outil commencer un échappement.**
- La sonde : **30 propositions malformées sur 30**, toutes arrêtées sur le même span,
  `{"name":"propose_\udde`.
- Dans une chaîne, la grammaire admettait `\u` incomplet (docs/33 amendement 8). Mais dans
  un nom d'outil, aucun chiffre hexadécimal ne complète `\udde` en préfixe d'un nom
  existant. Le span était donc mort dès ce jeton, sous une grammaire qui le disait viable.
- **Correctif** : un nom d'outil ou une clé d'argument appartient à un ensemble fini que le
  rendu écrit sans échappement. La grammaire y refuse tout échappement, et continue de les
  admettre dans les valeurs (`_scan_string(identifier=True)`, test
  `test_grammar_refuses_escapes_in_tool_names_and_keys`).

**2. Les propositions apprenaient au solveur à ne plus copier.**
- Le banc du solveur est tombé de **1,0 à 0,0**, et le banc hors distribution de 0,767 à
  0,0.
- Rejoué avec la grammaire corrigée, le même checkpoint échoue encore sur les 30 tâches
  (graine 7), toutes de la même façon : il appelle `calc` avec **la même expression,
  `548 + 104`**, quelle que soit la tâche, et note 652. L'expression n'est plus copiée du
  but : elle est générée.
- **La cause.** Dans une trajectoire de proposition, l'expression est inventée : elle
  n'apparaît nulle part avant l'appel. `build_action_targets` entraînait donc la porte de
  copie à « ne pas copier » à cet endroit. Or :
  - le proposeur décode **sans copie** (`allow_copy = False`, docs/33 amendement 5) : cette
    supervision entraînait une porte qu'il n'interroge jamais ;
  - la clé est `expression`, **celle-là même** sous laquelle le solveur copie l'expression
    du but.

  Cinquante pas ont suffi à fermer la porte du solveur. **Lue directement** à l'endroit où
  le solveur l'interroge (10 tâches, graine 7), elle vaut :
  - après le premier temps : logit **+9,32** (9,30 à 9,32), 10 copies sur 10 ;
  - après le barreau : **−1,16** (−1,40 à −1,09), 0 copie sur 10.

  `lookup` ne partage aucune clé entre proposeur et solveur ; l'effet n'y a pas été mesuré.
- **Correctif** : `TrainConfig.gate_keys` indique, outil par outil, les clés où le
  décodeur interroge la porte. Ailleurs, les valeurs gardent la perte du modèle de langue
  mais n'entraînent ni la porte ni les pointeurs. Il n'entre dans le contrat
  d'entraînement que s'il est posé, pour que les checkpoints antérieurs se reprennent.
  `closed_loop.py` le dérive de la table `PROPOSE_COPY_KEYS`, que lit aussi le décodage
  des propositions : `propose_calc` n'a aucune clé, `propose_lookup` a `ask` sous
  `--propose-copy ask`. Tests :
  - `test_gate_keys_train_the_gate_only_where_the_decoder_asks_it` ;
  - `test_trainer_passes_gate_keys_and_keeps_old_contracts` ;
  - `test_train_rows_hands_the_gate_keys_to_the_trainer`.

**Ce qui est relancé, sans rien d'autre de changé.** L'échelle de §2, barreaux dans le même
ordre et mêmes critères, puis les trois bras si un barreau passe. Le premier temps de
l'amorce (100 trajectoires, 200 pas) est réutilisé : il ne contient aucune proposition. Ses
chiffres (banc 1,0, hors distribution 0,767) ont été lus sous l'ancienne grammaire. Chaque
barreau relit les siens.

**Lecture pré-écrite.** Si le premier barreau fait encore tomber le banc du solveur sous la
fenêtre, la porte n'était pas seule en cause : on s'arrête pour diagnostiquer, sans régler
de paramètre.

## Amendement 2 — 2026-09-23, après SI-1 : SI-1b, pré-enregistrée — payer la récompense du proposeur

**Ce qui manquait à SI-1** (§3) : une reprise ne pouvait rien rattraper, donc la récompense
du proposeur n'a jamais été versée. Il faut d'abord qu'une reprise puisse réussir là où le
premier essai échoue ; alors seulement on peut demander si le proposeur bouge.

**Le mécanisme, mesuré avant tout run** (sonde en lecture seule sur le checkpoint du tour
0 : les reprises de `generate_round`, deux essais qui explorent après un premier qui
n'explore pas) :

| Reprises | Banc hors distribution : rattrapées sur 13 échecs | Propositions jamais résolues : rattrapées sur 6 |
|---|---:|---:|
| début tiré parmi 3, spans des observations (recette de SI-1) | 0 | 0 |
| début tiré parmi 3, tous les spans | 0 | 0 |
| + fin tirée parmi 3 (`copy_explore_end`) | 1 à 2 | 0 |
| **+ fin restreinte aux fins de mot (`copy_end_boundaries = "explore"`)** | **4** | **4** |
| + les deux | 3 | 4 |

Les six propositions jamais résolues échouent toutes de la même façon : l'opérande à quatre
chiffres est coupé au troisième (`908 + 1263` → `calc("908 + 126")`). Les chiffres sont des
jetons isolés, et le générateur n'écrit jamais plus de trois chiffres. Aucun tirage parmi
les trois meilleures fins n'atteint le quatrième. La règle de docs/31 amendement 19 (une
valeur copiée est un mot entier) le fait, appliquée à la fin du span.

**Deux options nouvelles**, lues par le code et testées :
- `AgentConfig.copy_explore_end` et `choose_copy_span(end_topk=)` : la fin tirée parmi les
  *k* meilleures positions, sous `--copy-explore-end`. Mesurée ci-dessus, **pas retenue**.
- `AgentConfig.copy_end_boundaries` (`off`, `explore`, `always`) et `_word_end`, sous
  `--copy-end-boundaries`. En mode `explore`, la restriction ne joue que sur les essais qui
  explorent (`copy_topk` > 0), si bien que le banc glouton et le premier essai sont
  inchangés.

**Un second banc hors distribution** : `make_hard_calc_digits`, 2 × 30 tâches `a op b` à
deux opérandes de **quatre chiffres** (graines 17 et 19), que le générateur n'écrit jamais.
C'est l'axe qu'une reprise peut rattraper, là où le banc à trois opérandes en mesure un
autre. Il est évalué après le run par `scripts/bench_checkpoint.py`, qui relit la config
et les réglages de décodage du `protocol.json` du run.

Mesures a posteriori sur SI-1, **non pré-enregistrées**, qui servent de référence :
- tour 0 : 0,050 ;
- après 5 tours : `closed-propose` 0,167, `closed-clean` 0,217, `oracle` 0,150.

**SI-1b.** La seule variable qui change par rapport à SI-1 est `--copy-end-boundaries
explore`.
- Bras `closed-propose` et `closed-clean` : tous deux reprennent leurs échecs, donc la
  variable touche les deux.
- Le bras `oracle` de SI-1 est repris tel quel : il ne passe pas par les reprises.
- Même répertoire d'amorce que SI-1 (le barreau 50 / 50, déjà calibré ; le tour 0 est
  identique puisque le banc est glouton), même graine, 5 tours.

| | Critère |
|---|---|
| **M** (mécanisme) | des propositions résolues **à une reprise** sur ≥ 3 tours sur 5 : la récompense du proposeur est versée |
| H20c, H21c, H22c, H24c, H25c | inchangés (§2) |
| H23c | inchangé : banc à trois opérandes |
| **H26c portée en chiffres** | banc à quatre chiffres, du tour 0 au tour 5 : gain de `closed-propose` > gain de `closed-clean`, intervalle de `closed-propose` (60 tâches) excluant zéro |

**Lecture pré-écrite.**
- Si **M** échoue, ce qui rattrapait dans la sonde ne rattrape pas dans la boucle. On
  s'arrête pour comprendre.
- Si **M** passe et que la part de propositions nouvelles **monte** d'un tour à l'autre, le
  proposeur bouge quand on le récompense : c'est la première trace dans ce dépôt d'un
  modèle qui déplace lui-même la distribution de ses tâches.
- Si **H26c** passe, c'est la première mesure d'une boucle qui va au-delà de son
  générateur, sur l'axe que son exploration atteint.
- Si **H23c** passe aussi, le gain se transfère à un axe que rien n'a exploré.
- Si **M** passe mais pas **H26c**, les propositions récompensées ne donnent pas au solveur
  plus que la boucle propre.

### SI-1b, 2026-09-23 (graine 0, amendement 2) — **non reproduit** : une seule graine

Deux bras de 5 tours (≈ 21 min de compute chacun), sur l'amorce de SI-1. Le bras `oracle`
est repris de SI-1. Les bancs a posteriori viennent de `scripts/bench_checkpoint.py`. Pour
le banc à trois opérandes, rejoué a posteriori, les valeurs sont identiques à celles que le
run a mesurées au tour 5 (0,783 ; 0,750 ; 0,750) : la mesure a posteriori est la même
mesure.

| | Critère | Mesure | Verdict |
|---|---|---|---|
| **M** | récompense versée sur ≥ 3 tours sur 5 | propositions résolues à une reprise : **5 ; 1 ; 0 ; 0 ; 0** | **échoue** |
| H20c | ≥ 0,5 valides à chaque tour | 0,97 ; 0,90 ; 0,93 ; 0,87 ; 0,83 | passe |
| H21c | ≥ 3 tours résolus entre 0,2 et 0,8 | 1,0 à chaque tour | échoue |
| H22c | banc du générateur, perte ≤ 0,05 | 1,0 → 0,983 | passe |
| H23c | trois opérandes, `closed-propose` > `closed-clean`, intervalle > 0 | **0,0** [−0,10 ; +0,083] (4 gagnées, 4 perdues) contre −0,033 | échoue |
| H24c | ΔBPB ≤ témoin + 0,02 | +0,098 contre +0,095 | passe |
| H25c | ≥ 50 % de nouvelles (sur les 5 tours) | **34,8 %** ; par tour 5/29, 10/27, 8/28, 11/26, **13/25** | échoue |
| **H26c** | quatre chiffres, gain `closed-propose` > `closed-clean`, intervalle > 0 | **+0,917** [+0,833 ; +0,983], 55 gagnées, 0 perdue, contre +0,167 | **passe** |

Banc à quatre chiffres (60 tâches, graines 17 et 19), succès :

| tour 0 | `closed-propose`, tour 5 | `closed-clean`, tour 5 | `oracle`, tour 5 |
|---:|---:|---:|---:|
| 0,050 | **0,967** | 0,217 | 0,150 |

**Ce qui s'est passé, tour par tour.**
1. **Tour 1.** Les 5 propositions nouvelles, toutes à opérande de quatre chiffres,
   échouent au premier essai et sont rattrapées à la reprise : c'est la troncature que la
   restriction aux fins de mot corrige. Toutes les 5 sont promues. **La récompense du
   proposeur est versée pour la première fois**, et exactement sur les propositions qui
   sortent de la distribution.
2. **Tour 2.** Le solveur, entraîné sur ces reprises, résout les quatre chiffres du
   premier coup (26 sur 27, une reprise). La part de nouvelles double : 10 sur 27.
3. **Tours 3 à 5.** Plus rien n'est au bord, donc plus rien n'est promu. Mais les 6
   propositions promues sont rejouées à chaque tour, et la part de nouvelles monte
   encore, jusqu'à 13 sur 25. Le solveur résout tout ce qu'on lui propose.

**Lecture, par la lecture pré-écrite.**
- **H26c passe.** C'est la **première mesure dans ce dépôt d'une boucle qui va au-delà de
  son générateur**, sans tâche humaine : 0,05 → 0,967 sur des tâches que le générateur
  n'écrit jamais. Le témoin nourri par le générateur n'atteint que 0,217, l'oracle 0,150.
- **Sa portée est étroite, et le chiffre le dit.** L'axe gagné est celui que
  l'exploration atteint : la longueur des nombres. Rien ne se transfère au troisième
  opérande (H23c : 0,0).
- **M échoue sous sa forme écrite, et la raison compte.** La récompense a été versée deux
  tours, puis le bord a disparu : le solveur a rattrapé le proposeur, et le proposeur ne
  va pas plus loin. Sur 135 propositions valides, **aucune** n'a deux opérateurs, et les
  règles plafonnent les nombres à quatre chiffres. Le proposeur a épuisé le seul axe
  qu'il explore.

**Les deux murs, nommés** (ce que SI-8 devait trouver) :
1. **Le proposeur n'explore pas la structure.** Un second opérateur a une probabilité que
   le tirage des valeurs (top-5) n'atteint jamais.
2. **Une reprise ne sait pas ajouter ce qui manque.** Même proposée, une expression à
   trois opérandes échoue par un opérande omis, et la fin coupée après `250` est une fin
   de mot légitime : aucune règle de bornes ne la rattrape (sonde de l'amendement 2 : 4
   sur 13 au mieux sur ce banc).

## Amendement 3 — 2026-09-23, après SI-1b : réplication pré-enregistrée aux graines 1 et 2

SI-1b tient sur une graine. Avant de s'appuyer sur H26c, elle est rejouée **à l'identique**
aux graines 1 et 2 :
- premier temps de l'amorce propre à chaque graine ;
- échelle de calibration de §2, premier barreau qui passe ;
- `closed-propose` et `closed-clean`, 5 tours, `--copy-end-boundaries explore` ;
- bancs a posteriori à quatre chiffres et à trois opérandes.

Le bras `oracle` n'est pas rejoué : il ne porte sur aucun critère de la réplication. Les
bancs sont les mêmes pour toutes les graines (générateur 7 et 11, hors distribution 17 et
19). Seules changent les tâches d'amorce et de tour, et les graines d'entraînement.

**Critère de réplication** : H26c passe **à chacune** des deux graines. M, H23c et H25c sont
rapportés graine par graine. Si une graine ne passe pas un barreau de calibration, elle est
rapportée comme telle et ne compte ni pour ni contre.

### Réplication de SI-1b, 2026-09-23 (amendement 3) — **non reproduite**

Graines 1 et 2, même protocole. Les deux passent le premier barreau (banc 1,0 ; hors
distribution 0,80 et 0,667 ; 30 propositions valides sur 30, 1 et 0 nouvelles à la sonde).
Chaque run tient en ≈ 1 h 20 de CPU. La graine 1 a été interrompue par un redémarrage du
conteneur au tour 5 de `closed-propose`, puis reprise depuis son checkpoint du tour 4.

Banc à quatre chiffres (60 tâches). La dernière colonne donne la différence entre les deux
bras au tour 5, tâche par tâche : ce calcul n'est **pas pré-enregistré**, il est rapporté
parce que le critère écrit s'est révélé trop faible (voir plus bas).

| Graine | tour 0 | `closed-propose`, gain [IC 95 %] | `closed-clean`, gain | **H26c** | différence au tour 5 [IC 95 %] |
|---|---:|---|---|---|---|
| 0 (SI-1b) | 0,050 | **+0,917** [+0,833 ; +0,983] | +0,167 | passe | **+0,750** [+0,633 ; +0,850] |
| 1 | 0,017 | +0,183 [+0,083 ; +0,283] | +0,117 | passe (à la lettre) | +0,067 [−0,033 ; +0,167] |
| 2 | 0,283 | +0,083 [−0,050 ; +0,217] | +0,017 | **échoue** | +0,067 [−0,050 ; +0,183] |

Le mécanisme, graine par graine :

| Graine | propositions nouvelles, tour par tour | rattrapées à une reprise | promues |
|---|---|---|---|
| 0 | 5/29, 10/27, 8/28, 11/26, 13/25 | 5, 1, 0, 0, 0 | 6 |
| 1 | 0/30, **0/0, 0/0, 0/0**, 0/29 | 0 | 0 |
| 2 | 0/29, 0/29, 3/27, 0/29, 0/27 | 1 (une tâche non nouvelle) | 1 |

**Verdict pré-enregistré** : H26c doit passer à chacune des deux graines. Elle échoue à la
graine 2. **La réplication échoue.**

**Ce qu'on en lit.**
1. **Le résultat de la graine 0 tient pour cette graine.** L'écart entre les bras est
   +0,75, et son intervalle est loin de zéro.
2. **Il dépend d'un événement de hasard : que le proposeur écrive des propositions
   nouvelles dès le tour 1.** Aux graines 1 et 2, il n'en écrit presque aucune. Rien n'est
   donc à rattraper, la récompense n'est jamais versée pour une nouveauté, et le mécanisme
   ne démarre pas. Le mur n'est plus la reprise (levé par l'amendement 2) : **c'est
   l'exploration du proposeur.** Sa part de nouvelles au départ vaut 5/29 à la graine 0,
   0/30 et 0/29 aux graines 1 et 2.
3. **Le critère H26c était trop faible**, et on le rapporte comme tel. Il compare deux
   gains ponctuels, sans intervalle sur leur différence. À la graine 1, il passe alors que
   le proposeur n'a écrit aucune proposition nouvelle en 5 tours : l'écart de +0,067
   (4 tâches sur 60) est celui de deux bras sans mécanisme. Un critère futur portera sur
   l'intervalle de la différence entre bras.
4. **Un proposeur non récompensé peut oublier de proposer.** Graine 1, tours 2 à 4 :
   0 proposition valide sur 30. Le modèle, appelé à proposer, écrit l'action réservée
   `note`, comme le deuxième pas du solveur. Aucune trajectoire de proposition n'entre à
   l'entraînement quand rien n'est promu, et les solutions du tour 1 poussent le format
   dehors. Il revient seul au tour 5.

   **Hypothèse d'une fuite d'état testée, réfutée.** Le retour coïncidait avec le
   redémarrage du conteneur, d'où le soupçon d'un état hors des poids qui passerait d'une
   phase à l'autre. Mais sur les poids du tour 4, le modèle chargé à neuf, puis après un
   banc du solveur, propose 28 valides sur 30 dans les deux cas. Après 60 pas
   d'entraînement, en mémoire et rechargé : 23 et 23. C'est une dynamique
   d'entraînement, pas un défaut.
5. **Observation non pré-enregistrée, à confirmer.** À la graine 2, proposer aide le banc
   à trois opérandes : +0,100 contre −0,033, différence +0,133 [+0,033 ; +0,250].

## Amendement 4 — 2026-09-23, après la réplication : SI-1c, pré-enregistrée — tirer plus de propositions

**Le mur** : la part de propositions nouvelles au départ, qui décide si le mécanisme
démarre (réplication ci-dessus).

**Sonde en lecture seule** (checkpoints de calibration, `propose_round` comme le run) :

| Graine | tirages | 0,7 / top-5 (recette) : valides, nouvelles, à deux opérateurs | 1,0 / top-20 : valides, nouvelles, à deux opérateurs |
|---|---:|---|---|
| 0 | 30 | 27, 4, 0 | 23, 6, 0 |
| 1 | 30 | 29, 0, 0 | 22, 4, 2 |
| 2 | 30 | 26, 0, 0 | 22, 0, 0 |
| 1 | **120** | 112, **7**, **3** | 87, 10, 5 |
| 2 | **120** | 100, 1, 0 | 85, 3, 1 |

Élargir le tirage ajoute peu de nouveauté et coûte en validité (≈ −15 points ; à 1,5, la
moitié des propositions sont refusées). Tirer **plus** au même réglage multiplie le nombre
de propositions nouvelles, sans rien coûter en validité. C'est aussi à 120 tirages
qu'apparaissent les premières propositions à **deux opérateurs** : l'axe de la structure,
qu'aucune des 405 propositions valides des runs précédents n'avait touché.

**SI-1c.** La seule variable qui change par rapport à SI-1b est `--propose-n 120` : 120
propositions par tour, toutes validées. Toutes les propositions valides sont résolues,
avec trois tentatives dont deux reprises. Le reste est identique :
- graines 0, 1 et 2 ;
- les mêmes amorces et les mêmes barreaux, déjà calibrés ;
- 5 tours.

`closed-clean` ne lit pas `--propose-n` : ses runs de SI-1b et de la réplication servent
de témoin tels quels. Le compute est compté, et il est plus élevé pour `closed-propose`.

**Critères**, durcis d'après la réplication :

| | Critère, à chaque graine |
|---|---|
| **M′** (mécanisme) | au moins une proposition **nouvelle** promue (résolue à une reprise) sur les 5 tours |
| **H26c′** (celle qui compte) | banc à quatre chiffres au tour 5 : différence `closed-propose` − `closed-clean`, tâche par tâche, intervalle bootstrap à 95 % **excluant zéro** |
| H23c′ | la même différence sur le banc à trois opérandes (secondaire) |
| H20c, H22c, H24c | inchangés |

**SI-1c réussit si H26c′ passe à chacune des trois graines.**

**Lecture pré-écrite.**
- **H26c′ passe partout** : tirer davantage rend le mécanisme fiable à 7 M. La boucle va
  au-delà de son générateur, sans tâche humaine, sur l'axe que sa reprise rattrape.
- **H26c′ passe là où M′ passe, et seulement là** : le mur est quantitatif. Il faut un
  nombre minimal de propositions nouvelles pour que la récompense soit versée.
- **M′ passe mais pas H26c′** : la récompense versée ne suffit pas.
- **H23c′ passe** : les propositions à deux opérateurs ouvrent l'axe de la structure.

### SI-1c, 2026-09-23 (amendement 4) — critère de succès **échoué**, lecture pré-écrite confirmée

`closed-propose` avec 120 propositions par tour, graines 0, 1 et 2. Chaque graine tient en
≈ 50 min de CPU (27 à 28 min de compute compté). Les témoins `closed-clean` sont ceux de
SI-1b et de la réplication. Bancs a posteriori, 60 tâches. La différence entre bras est
calculée tâche par tâche au tour 5, avec un intervalle bootstrap à 95 %.

| Graine | **M′** : nouvelles promues | nouvelles par tour | quatre chiffres : `propose` / `clean` | **H26c′** : différence [IC 95 %] | H23c′ : trois opérandes | validité par tour |
|---|---:|---|---|---|---|---|
| 0 | **15** | 16, 26, 51, 57, 51 | 0,933 / 0,217 | **+0,717** [+0,600 ; +0,833] — passe | +0,050 [−0,033 ; +0,133] | 0,86 → 0,66 |
| 1 | **2** | 2, 0, 0, 2, 7 | 0,933 / 0,133 | **+0,800** [+0,700 ; +0,900] — passe | +0,100 [0,000 ; +0,200] | 0,98 ; **0,0** ; 0,92 ; 0,69 ; 0,61 |
| 2 | **0** | 3, 1, 0, 0, 1 | 0,300 / 0,300 | 0,000 [−0,117 ; +0,117] — échoue | +0,050 [−0,067 ; +0,183] | 0,88 → 0,90 |

Autres critères : le banc du générateur reste ≥ 0,983 partout (H22c). ΔBPB vaut +0,098,
+0,108 et +0,123, contre +0,095 à +0,126 pour les témoins (H24c). H20c échoue à la graine
1 (tour 2 : 0 valide).

**Verdict pré-enregistré** : SI-1c devait réussir à chacune des trois graines. Elle échoue
à la graine 2.

**La lecture pré-écrite qui s'applique** : « H26c′ passe là où M′ passe, et seulement
là : le mur est quantitatif ». C'est exactement ce qu'on observe. En comptant SI-1b, sa
réplication et SI-1c, **six runs** donnent la même règle, sans exception :

| Run | nouvelles promues | écart entre bras, quatre chiffres |
|---|---:|---|
| SI-1b graine 0 | 6 | +0,750 |
| SI-1c graine 0 | 15 | +0,717 |
| SI-1c graine 1 | 2 | +0,800 |
| SI-1b graine 1 | 0 | +0,067 (intervalle incluant zéro) |
| SI-1b graine 2 | 0 | +0,067 (intervalle incluant zéro) |
| SI-1c graine 2 | 0 | 0,000 |

**Ce que cela établit, à 7 M, sur `calc` :**
1. **Une boucle qui propose ses tâches va au-delà de son générateur dès qu'une seule
   proposition nouvelle est rattrapée et promue.** Deux suffisent (graine 1) : leurs
   épisodes rattrapés apprennent au solveur à copier des nombres de quatre chiffres, et
   le banc passe de 0,017 à 0,933, sans tâche humaine. Le témoin nourri par le
   générateur atteint 0,133.
2. **Que cela arrive reste une question de tirage.**
   - Tirer 120 propositions au lieu de 30 a fait démarrer la graine 1, qui ne démarrait
     pas.
   - La graine 2 n'a écrit que 5 propositions nouvelles en 5 tours, et la seule à quatre
     chiffres (`876 * 2463`) n'a pas été rattrapée.
3. **Les propositions à deux opérateurs apparaissent** (graine 2 : `386 * 266 + 535`,
   deux sur les 5 tours). Mais elles sont résolues du premier coup, donc jamais
   récompensées. La récompense « résolue à une reprise » ne pousse pas le proposeur sur
   un axe que le solveur sait déjà faire.
4. **Le proposeur noyé peut oublier de proposer** : graine 1, tour 2, 0 valide sur 120,
   comme dans la réplication. Au tour 1, il a été entraîné sur 117 solutions pour 3
   propositions.

**Les murs, nommés à ce point :**
- (a) la rareté des propositions nouvelles rattrapables au départ ;
- (b) une récompense qui s'éteint quand le solveur rattrape (graine 0, tours 4 et 5 : 0
  promue sur plus de 50 nouvelles) ;
- (c) l'oubli du format de proposition quand les épisodes du solveur dominent
  l'entraînement ;
- (d) le plafond des règles : quatre chiffres, et l'axe des opérateurs, que la
  récompense ne paie pas.

## Amendement 5 — 2026-09-23, après SI-1c : SI-8a, pré-enregistrée — relever le plafond des règles

**La question du but, posée directement** : quand les règles le permettent, la frontière
continue-t-elle d'avancer ? Aux graines 0 et 1, la boucle a appris les nombres de quatre
chiffres, **le plafond des règles** (`\d{1,4}`). Si ce plafond est relevé, une boucle qui
s'améliore elle-même devrait monter la marche suivante (cinq chiffres, puis six), toujours
sans tâche humaine.

**Trois ajouts**, chacun lu par le code et testé :
- `--calc-max-digits` (`calc_rules`, `validate(max_digits=)`) : le plafond des règles de
  proposition ; 4 par défaut, c'est-à-dire les règles de SI-1 exactement ;
- `make_hard_calc_digits(digits=)` et `make_bench` : les bancs `calc-digits5` à
  `calc-digits8`, deux opérandes de *d* chiffres, graines 17 et 19 ;
- `--extra-bench` : ces bancs mesurés **à chaque tour**, pour lire la marche tour par tour
  plutôt que sur le dernier checkpoint.

**SI-8a.** La seule variable est `--calc-max-digits 8`. Tout le reste reprend la recette de
SI-1c :
- 120 propositions par tour, fins de mot sur les reprises, amorces déjà calibrées ;
- mais **10 tours** au lieu de 5 : SI-8 porte sur la pente ;
- bancs à chaque tour : `calc-digits` (4 chiffres), `calc-digits5`, `calc-digits6`.

Deux bras par graine :
- **plafond 8** ;
- **plafond 4** (témoin) : le même run avec les règles d'avant. Il sépare « proposer des
  nombres de cinq chiffres » de « généraliser depuis quatre ».

Graine 0 d'abord, puis graine 1, les deux graines où le mécanisme a démarré en SI-1c.
Environ 2 h de CPU par bras.

**Critères**, à chaque graine :

| | Critère |
|---|---|
| **S1** (une marche) | banc à cinq chiffres au tour 10 : différence plafond 8 − plafond 4, tâche par tâche, intervalle bootstrap à 95 % excluant zéro |
| **S2** (deux marches) | la même chose sur le banc à six chiffres |
| **S3** (le proposeur monte) | au moins une proposition promue a un entier de plus de quatre chiffres |
| garde | banc du générateur ≥ 0,95 ; ΔBPB du bras plafond 8 ≤ celui du témoin + 0,02 ; validité ≥ 0,5 à chaque tour |

Sont rapportés en plus :
- le tour où chaque banc dépasse 0,5 ;
- la distribution des longueurs de nombres proposés, tour par tour.

**Lecture pré-écrite.**
- **S1 et S2 passent** : la boucle franchit l'ancien plafond sur deux marches, sans tâche
  humaine. Ce serait la première marche d'escalier mesurée dans ce dépôt.
- **S1 passe, S2 échoue** : une marche. On nomme ce qui arrête la seconde : la rareté
  des propositions, une reprise qui ne rattrape pas, ou une récompense qui s'éteint.
- **S1 échoue** : relever le plafond ne fait pas bouger la boucle. Soit le proposeur
  n'écrit pas de nombres de cinq chiffres, soit la reprise ne les rattrape pas ; le
  tableau de S3 le dira.

### SI-8a, graine 0, 2026-09-23 (amendement 5) — S1, S2 et S3 passent ; graine 1 en cours

Deux bras de 10 tours, ≈ 2 h 10 de CPU chacun (compte : 3 356 s et 3 066 s). Bancs
mesurés à chaque tour, 60 tâches ; différences au tour 10, tâche par tâche.

| Banc | tour 0 | plafond 8, tours 1 → 10 | plafond 4, tours 1 → 10 | différence au tour 10 [IC 95 %] |
|---|---:|---|---|---|
| 4 chiffres | 0,05 | 0,92 · 0,93 · 0,93 · 0,93 · 0,93 · 0,93 · 0,97 · 0,97 · 0,97 · **0,98** | 0,88 · 0,88 · 0,95 · 0,95 · 0,93 · 0,95 · 0,95 · 0,92 · 0,97 · **0,98** | 0,000 [−0,05 ; +0,05] |
| **5 chiffres** | 0,15 | 0,62 · 0,70 · 0,75 · **0,80** · 0,73 · 0,73 · 0,75 · 0,72 · 0,72 · **0,68** | 0,55 · 0,47 · 0,52 · 0,47 · 0,52 · 0,43 · 0,42 · 0,45 · 0,38 · **0,32** | **+0,367** [+0,217 ; +0,517] — **S1 passe** |
| **6 chiffres** | 0,03 | 0,48 · 0,38 · 0,45 · 0,38 · 0,37 · 0,35 · 0,32 · 0,33 · 0,37 · **0,37** | 0,35 · 0,28 · 0,32 · 0,30 · 0,27 · 0,27 · 0,25 · 0,25 · 0,25 · **0,23** | **+0,133** [+0,050 ; +0,217] — **S2 passe** |

**S3 passe.** Longueur (en chiffres) des propositions promues, tour par tour, pour le
plafond 8 :

| tour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| promues | 4 ×12 | 6 ×3, 5 | 7 ×4, 6, 4 | 7, 6 ×2, 5 | 7, 6 | 4 | 8, 7, 5 ×2 | 8, 7, 6 ×3 | 7, 5 ×2 | 7 ×2, 5, 3 |

Parmi les propositions valides, celles à trois chiffres passent de 87 sur 104 au tour 1 à
9 sur 105 au tour 10.

**Gardes.** Le banc du générateur finit à 0,95 (plafond 8), exactement le seuil, contre
0,983 au plafond 4. ΔBPB vaut +0,371, contre +0,377 au plafond 4. La validité reste
≥ 0,66 à chaque tour.

**Ce que la graine 0 montre.**
1. **La boucle franchit l'ancien plafond quand les règles le permettent, sans tâche
   humaine.** Le proposeur monte (4 → 6 → 7 → 8 chiffres promus), et le solveur garde
   sur cinq chiffres un gain que le témoin perd.
2. **Sans propositions au-delà, le progrès s'érode.** Au plafond 4, la généralisation
   du tour 1 (0,55 sur cinq chiffres) retombe à 0,32 : la boucle se spécialise sur ce
   qu'elle voit. Au plafond 8, elle tient entre 0,68 et 0,80.
3. **Deux murs.**
   - **La marche à six chiffres plafonne vers 0,35.** Le proposeur promeut des nombres
     de 6 à 8 chiffres, mais un seul opérande à la fois ; le banc en combine deux, et
     leurs produits vont jusqu'à 12 chiffres. Diagnostic à faire.
   - **L'oubli de la langue croît avec les tours** : ΔBPB +0,10 à 5 tours (SI-1c), +0,37
     à 10 tours, **dans les deux bras**. Ce n'est pas l'effet de proposer. C'est la
     condition C4 (docs/35) qui devient le mur d'une boucle longue.
