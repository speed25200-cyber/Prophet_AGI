# 33 — Pré-enregistrement, programme 3 : le modèle propose ses tâches

**Statut : v0, rédigé le 2026-09-22 avant toute ligne de code du mécanisme et avant tout
tour. Amendements datés en fin de document, jamais réécrits.**

## 0. La question

Le programme 2 (docs/31–32) a établi qu'un modèle apprend de ses propres épisodes quand un
programme les vérifie, et où cela s'arrête : quand toutes les tâches que le **générateur**
sait produire sont réussies, il n'y a plus rien à apprendre, et ce que le modèle rate
toujours, il ne l'apprend jamais seul (§14, §16). Le plafond est celui du générateur de
tâches, écrit par un humain.

Ce programme retire le générateur. Le modèle **propose** lui-même les tâches, sous une forme
que le programme sait vérifier ; le programme ne fournit plus que les **règles** : une
grammaire de propositions, une validation, et un vérificateur dérivé mécaniquement de
chaque proposition (la réponse d'une tâche proposée est calculée par le programme à partir
de la proposition, jamais fournie par le modèle). C'est la position du go : les règles
vérifient toutes les parties, personne ne les écrit une à une.

La question, mesurable : **à compute égal, une boucle dont le modèle propose ses tâches
progresse-t-elle autant qu'une boucle nourrie par le générateur sur le banc de celui-ci, et
va-t-elle plus loin que lui sur des tâches hors de sa distribution ?**

## 1. Ce qui existe, et ce qui est différent

Revue par recherche web du 2026-09-22 ; chaque ligne marquée `[S]` est un résumé de
recherche à confirmer sur le texte intégral avant citation.

| Travail | Proposeur | Vérificateur | Ce qu'il mesure |
|---|---|---|---|
| AlphaZero / KataGo | les règles du jeu | les règles du jeu | force de jeu |
| STaR, ReST-EM `[S]` | jeu de problèmes fixe | réponses de référence | précision sur des bancs |
| Absolute Zero Reasoner, 2505.03335 `[S]` | le modèle (programmes Python) | l'exécuteur Python | raisonnement sur code et maths, à 7B |
| Voyager `[S]` | le modèle (curriculum d'objectifs) | l'environnement (Minecraft) | compétences accumulées |
| **Ce programme** | **le modèle**, sous une grammaire de spécifications | **dérivé de la spécification** par un programme, sans réponse de référence | gain sur le banc du générateur **et** sur un banc hors distribution, à compute compté, avec l'oubli, à 7 M puis 375 M |

Différences : le proposeur et le solveur sont un seul modèle avec des têtes d'action
typées ; la validité d'une proposition est une règle de grammaire, pas un jugement ; la
mesure est celle du programme 2 (capacité par heure de compute, oubli, ensemble jamais
réussi) contre le générateur humain comme témoin ; et tout tourne à l'échelle où le
programme 2 a trouvé ses défauts avant de monter.

## 2. Mécanique

**Spécification** d'une tâche `lookup` : `{"file": "<nom>.json", "fields": {"<clé>": "<valeur>",
…}, "ask": "<clé>"}`. **Règles** (validation par programme, aucune n'est un jugement) : 2 à 6
champs ; clés et valeurs non vides, alphanumériques, ≤ 12 caractères ; clés distinctes ;
`ask` ∈ `fields` ; nom de fichier alphanumérique ≤ 16 caractères avec l'extension. Une
spécification valide devient une tâche par le programme : les fichiers, le but avec le
gabarit du générateur (« Read *f* and note the value of the field *k*, then finish. »), la
réponse `fields[ask]`, le vérificateur habituel (la réponse figure dans les notes). Une
proposition déjà vue (même fichier, mêmes champs, même clé) dans la quarantaine est
écartée.

**Épisode de proposition** : un outil `propose_lookup` dont le schéma est la spécification ;
but « Propose a new lookup task: a JSON file, its fields, and the field to ask for. » ;
un pas, échantillonné à la température de génération ; l'appel est rendu comme tout appel
d'outil, dans la grammaire compacte. L'amorce contient *P* épisodes de proposition parfaits,
construits à partir des tâches du générateur (la tâche devient sa spécification).

**Tour de la boucle `closed-propose`** : le modèle propose *N* tâches (invalides écartées
et comptées) ; le solveur les tente comme au programme 2 (trois tentatives, reprises qui
explorent, promotion canonique) ; les épisodes de solution vérifiés sont promus ; les
épisodes de **proposition** sont promus quand la tâche proposée a été résolue **à la
reprise et non du premier coup** (elle était au bord de ce que le solveur sait faire :
c'est la seule récompense du proposeur, dite « apprenabilité » chez AZR). Entraînement sur
le bassin cumulé (solutions et propositions), rejeu 0,5, taux ÷ 4.

**Bras**, tous à partir de la même amorce (100 trajectoires parfaites + *P* = 50
propositions parfaites, 200 pas) :

| Bras | Tâches du tour | Entraînement |
|---|---|---|
| `closed-propose` | *N* proposées par le modèle | solutions vérifiées + propositions au bord |
| `closed-clean` (témoin, programme 2) | *N* du générateur | solutions vérifiées |
| `oracle` | *N* du générateur, trajectoires parfaites | identique |

**Bancs**, gloutons, jamais entraînés : le banc du générateur (2 × 30, graines 7 et 11) et
un banc **hors distribution** produit par le programme (`lookup` à 5 champs dont deux clés
hors de {city, year, code}, 2 × 30, graines 17 et 19), auquel aucun bras n'a accès et que
le générateur ne produit jamais.

## 3. Hypothèses et critères, fixés avant le premier tour

| Hypothèse | Énoncé mesurable | Critère |
|---|---|---|
| **H20 validité** | Le modèle amorcé propose des tâches que les règles acceptent. | ≥ 50 % des propositions valides à chaque tour ; sinon le mécanisme ne démarre pas, rapporté avec le taux. |
| **H21 bord** | Les propositions se placent au bord de la compétence. | sur ≥ 3 tours sur 5, le taux de résolution des propositions valides est entre 0,2 et 0,8. |
| **H22 transfert** | Proposer vaut le générateur sur son propre banc. | gain(`closed-propose`) ≥ gain(`closed-clean`) − 0,05 sur le banc du générateur, intervalle de `closed-propose` excluant zéro. |
| **H23 portée** | Proposer va plus loin que le générateur. | gain sur le banc hors distribution : `closed-propose` > `closed-clean` avec un intervalle (60 tâches) excluant zéro. |
| **H24 oubli** | | Δ BPB(`closed-propose`) ≤ Δ BPB(`closed-clean`) + 0,02. |
| **H25 nouveauté** | Les propositions ne recopient pas l'amorce. | ≥ 50 % des propositions valides ont un nombre de champs ou une clé absents des tâches de l'amorce. |

Les cinq sont rapportées séparément ; le programme « passe » si H20, H22 et H23 passent.
H23 est celle qui compte : sans elle, proposer n'est qu'un générateur plus cher.

## 4. Échelle, budget, mesures

Pilote CPU : 7 M (docs/09), `lookup`, graine 0 puis 2 ; 5 tours × 30 tâches ; ≈ 30 min par
bras et par graine (les propositions coûtent un épisode d'un pas chacune). Mesures par tour :
celles de docs/31 §4, plus propositions (émises, valides, nouvelles, résolues du premier
coup, résolues à la reprise, jamais résolues), distribution du nombre de champs et des clés
proposées, et les deux bancs. Échelle A100 : `scripts/closed_loop_a100.sh` avec le bras
`closed-propose`, après le programme 2 à 375 M.

## 5. Ce qui n'est pas dans ce programme

- Des tâches jugées par un modèle : sans programme vérificateur, pas de promotion.
- L'invention de **nouvelles familles** (de nouveaux outils ou de nouveaux gabarits) : la
  grammaire des spécifications est écrite par un humain, une famille à la fois ; ce
  programme mesure ce que le modèle fait *dans* cette grammaire.
- Toute revendication de progrès « sans limite » : les plafonds de docs/32 §14 et §19
  restent ; ce programme déplace le plafond du générateur, pas celui du vérificateur.

## 6. Pièces (à écrire, dans cet ordre, chacune avec son test)

| Pièce | Fichier | Test |
|---|---|---|
| Spécifications, règles, tâche dérivée, banc hors distribution | `prophet/agent/propose.py` | `tests/test_propose.py` |
| Épisodes de proposition : outil, rendu, amorce parfaite | `prophet/agent/propose.py`, `scripts/closed_loop.py` | idem |
| Bras `closed-propose`, comptes des propositions, second banc | `scripts/closed_loop.py` | `tests/test_closed_loop.py` |
| Résumé : H20–H25 depuis les enregistrements | `scripts/summarize_closed_loop.py` | `tests/test_closed_loop_summary.py` |

## Amendement 1 — 2026-09-22, après la calibration du premier pilote, avant tout tour entraîné

**Ce qui s'est passé.** Amorce « 100 trajectoires + 50 propositions, 200 pas », graine 0 :
banc du générateur **0,067** (l'amorce solveur seule donnait ≈ 0,6 sur cette graine en
v3–v5), banc hors distribution 0,000. Diagnostic : 27 échecs sur 30, tous `read_file` puis
`done` refusé trois fois — les épisodes de proposition (un appel, puis fin) ont appris au
solveur « un appel, puis `done` ». Sonde de 30 propositions sur cette amorce : **30
malformées sur 30**. Cause mécanique : un appel `propose_lookup` fait 60 jetons et le
budget de la boucle pour un span d'action est de 64 (docs/09) ; la moindre variation
échantillonnée dépasse le budget. Le pilote est arrêté avant tout tour ; rien n'est
comptabilisé.

**Deux corrections, pré-enregistrées.**

1. **Budget d'action des épisodes de proposition** porté à 160 jetons (`propose_round`) ; le
   banc et le solveur gardent 64.
2. **Amorce en deux temps.** Premier temps : l'amorce du programme 2 (100 trajectoires
   parfaites, 200 pas), inchangée. Second temps : *P* = 50 propositions parfaites **et** 50
   trajectoires parfaites du solveur, 50 pas au taux ÷ 4 avec rejeu 0,5 du corpus ; le
   solveur revoit sa forme pendant que le proposeur apprend la sienne. Enregistré dans
   `seed.json` du répertoire d'amorce, partagé par les trois bras.

**Règle de calibration du programme 3**, avant tout tour : sur le banc du générateur,
départ strictement entre 0,30 et 0,95 avec succès canonique > 0 ; sur une sonde de 30
propositions échantillonnées (tour 0, enregistrée), validité ≥ 0,5 (H20 au tour 0).
Sinon, échelle : second temps à 100 pas, puis *P* = 25 ; si aucune amorce ne satisfait
les deux, le résultat est « le proposeur ne démarre pas à 7 M », rapporté avec les taux.

## Amendement 2 — 2026-09-22, après le second barreau de calibration, avant tout tour entraîné

**Ce qui s'est passé.** Amorce en deux temps (50 pas, 50 propositions) : solveur à 0,433 sur
le banc du générateur (canonique 0,367), banc hors distribution 0,100 — le solveur est
réparé. Proposeur : **30 propositions malformées sur 30**, budget de 160 jetons compris.
Trace jeton par jeton : le span d'action est échantillonné sous la grammaire ; le modèle
tire « k » (préfixe viable de `keys`), et au jeton suivant aucun des 64 meilleurs candidats
du modèle ne prolonge viablement « k » (« eys » n'est pas dans sa tête de distribution) :
le span meurt. Un modèle de 7 M échantillonné sur la *structure* d'un appel se perd dans
ses propres sous-mots.

**Deux corrections, pré-enregistrées.**

1. **Échantillonner les valeurs seulement** (`AgentConfig.sample_scope = "values"`) : dans un
   span contraint, les noms, clés et ponctuations sont décodés gloutonnement, seuls les
   jetons à l'intérieur d'une valeur chaîne sont tirés. La diversité des propositions vient
   des valeurs (noms de fichiers, clés, valeurs, clé demandée), pas de la structure.
2. **Élargir les candidats** (`AgentConfig.decoder_widen`) : quand aucun des 64 meilleurs
   jetons ne garde l'appel viable, vérifier tout le vocabulaire avant d'abandonner le span.

Les deux ne s'appliquent qu'aux épisodes de proposition ; le banc et le solveur gardent le
décodage de docs/09. Les tests couvrent l'état « dans une chaîne » de la grammaire, la
portée de l'échantillonnage et l'élargissement. La règle de calibration de l'amendement 1
est inchangée ; l'échelle repart au premier barreau (second temps 50 pas, 50 propositions).

## Amendement 3 — 2026-09-22, même calibration, avant tout tour entraîné : une clé déjà donnée

**Ce qui s'est passé.** Avec les valeurs seules échantillonnées, la trace montre un autre
mur : après `"ask":"code",` le modèle veut réécrire `"file"` ; la grammaire acceptait le
préfixe `"f` (préfixe d'un paramètre) et ne refusait le doublon qu'à la fermeture de la
clé, où plus rien n'est viable sauf un antislash. 30 malformées sur 30, pour cette raison.

**Correction, pré-enregistrée.** Un préfixe de clé n'est viable que s'il ouvre un paramètre
**pas encore donné** (`ActionGrammar`, test). C'est un défaut de la grammaire de décodage,
pas du proposeur ; il n'affectait pas le décodage glouton des pilotes précédents (qui ne
répète pas de clé) et ne change aucun de leurs chiffres. Calibration relancée sur la même
amorce, même règle.

## Amendement 4 — 2026-09-22, même calibration, avant tout tour entraîné : fermer l'appel, tirer près de la tête

**Ce qui s'est passé.** Après les amendements 2 et 3 : 30 propositions, 14 malformées,
16 invalides, 0 valide ; à température 0,3 ou 0,5, pas mieux. Deux causes lues dans les
sorties. (a) Structure : après le quatrième et dernier paramètre, le modèle émet une
virgule ; la grammaire l'acceptait alors qu'aucun paramètre ne reste, et le span mourait
sur une clé impossible (`"values":"code",",`). (b) Contenu : les valeurs tirées à 0,7 sur
la queue de la distribution d'un modèle de 7 M sont des sous-mots brouillés
(`cityical`, `codereserveloparameters`) ; `keys` n'est plus une liste de 2 à 6 jetons.

**Corrections, pré-enregistrées.** (a) La grammaire refuse la virgule quand tous les
paramètres sont donnés : la fermeture est forcée (test). (b) `AgentConfig.sample_topk` :
les valeurs sont tirées parmi les **5** jetons les plus probables (après le masque de
grammaire), la température inchangée ; les épisodes de proposition seuls. La règle de
calibration de l'amendement 1 est inchangée ; l'échelle repart au premier barreau.

## Amendement 5 — 2026-09-22, même calibration, avant tout tour entraîné : ne pas copier ce qu'on invente

**Ce qui s'est passé.** Après l'amendement 4 : 30 propositions, 9 malformées, 21 invalides,
0 valide. Les invalides sont de deux sortes : des listes de clés réduites à un mot brouillé
(`codeerslook`, `cityslo` : 7 sur 14 sondées) et des valeurs **copiées du prompt épinglé**
(`task: a file name, comma-separated`, `parameters`, `Read one file`) — la porte de copie
s'ouvre à un début de valeur et le pointeur recopie un fragment du schéma, puisque rien
dans le contexte ne contient la valeur à inventer.

**Correction, pré-enregistrée.** `AgentConfig.allow_copy = False` pour les épisodes de
proposition : les valeurs sont générées, jamais épissées (test). Les mots brouillés
relèvent de l'entraînement du proposeur, que l'échelle de l'amendement 1 (second temps à
100 pas) mesure ; s'ils persistent à 100 pas, le résultat est « le proposeur ne démarre pas
à 7 M », rapporté avec les taux. L'échelle repart au premier barreau avec cette correction.

## Amendement 6 — 2026-09-22, échelle en cours, avant tout tour entraîné : les clés dans l'ordre du schéma, et un barreau de plus

**Ce qui s'est passé.** Second temps à 100 pas : solveur à **0,733** (canonique 0,600), banc
hors distribution 0,433 — les cent pas de plus ont aussi profité au solveur. Proposeur :
30 propositions, **1** malformée, 29 invalides, 0 valide ; sous les amendements 4 et 5, les
appels sont bien formés mais le contenu est mal *placé* : la liste `city,year,code` dans
`ask`, un mot seul dans `keys`. Le modèle émet les clés dans un ordre libre (`file, ask,
keys, values` ou `file, values, ask, keys`), que la grammaire accepte, alors que le rendu
les écrit toujours dans l'ordre du schéma (`file, keys, values, ask`) : même écart que la
grammaire tolérante aux blancs de docs/32 (mode a), qui avait coûté 0,233 contre 0,550.

**Corrections, pré-enregistrées.** (1) `ActionGrammar(ordered=True)` : les clés d'un appel
doivent suivre l'ordre du schéma, y compris en préfixe ; `AgentConfig.ordered_keys`, activé
pour les épisodes de proposition seulement (test). (2) L'échelle de calibration de
l'amendement 1 reçoit un barreau de plus, après « second temps 100 pas » : **100
propositions, 200 pas**, puisque le passage de 50 à 100 pas a réduit les malformées de 30
à 1 et relevé le solveur. Ordre des barreaux : 50 / 50, 100 / 50, 100 / 25, 200 / 100. La
règle de calibration est inchangée.

## Amendement 7 — 2026-09-22, échelle reprise par une nouvelle session, avant tout tour entraîné : un premier run reconstruit, l'échelle entière rejouée

**Ce qui s'est passé.** La session qui a écrit l'amendement 6 a rejoué les barreaux 1 à 3
sous la grammaire ordonnée (`28b9e9a`) : 50 pas / 50 propositions, 7 malformées, 23
invalides, 0 valide ; 100 / 50, solveur 0,733, 17 malformées, 13 invalides, 0 valide ;
100 / 25, solveur 0,617, banc hors distribution 0,417, 8 malformées, 22 invalides, 0
valide. Elle a été archivée pendant le barreau 4 (200 / 100) : son verdict, le premier run
7 M, les amorces et les `rounds.jsonl` sont perdus avec son conteneur. Ces trois barreaux
ne sont connus que par ses messages : **non reproduits**, rapportés comme tels, jamais
comptés dans un verdict.

**Ce qui change : les poids, pas le protocole.** Le premier run est reconstruit dans le
nouveau conteneur par `scripts/first_run_cpu.py --stage all --minutes 45 --resume-minutes 15
--tokens 2.4e6` (docs/09), PyTorch 2.14 sur CPU. Son corpus est le texte que cette machine
contient, qui n'est pas celui de la première : 7 497 documents de prose au lieu de 6 745,
213 tenus à l'écart au lieu de 198. Les poids ne sont donc pas ceux des amendements 1 à 6,
et l'échelle est rejouée **en entier** sur eux, dans l'ordre de l'amendement 6 (second
temps 50 / 50, 100 / 50, 100 / 25, 200 / 100). Le premier temps (100 trajectoires
parfaites, 200 pas) est entraîné une fois, par le bras `closed-clean` à zéro tour, puis
copié pour chaque barreau. Chaque barreau est le bras `closed-propose` à zéro tour sur sa
copie : second temps, deux bancs, sonde de 30 propositions. Graine 0, options de
`scripts/closed_loop_cpu_pilot_propose.sh`, un seul processus torch à la fois.

**Enregistrement.** La sonde garde désormais chaque proposition dans `rounds.jsonl`
(`samples` : les arguments, la règle qui les refuse, ou le span d'un appel malformé, que la
boucle conserve désormais dans `StepRecord.span`). C'est une lecture, pas un critère : aucun
compte ne change (tests).

**Règle, inchangée (amendement 1).** Banc du générateur strictement entre 0,30 et 0,95 avec
succès canonique > 0, et validité ≥ 0,5 sur la sonde. Le premier barreau qui satisfait les
deux arrête l'échelle et lance les trois bras de §2 sur la graine 0. Si aucun ne les
satisfait, le résultat est « le proposeur ne démarre pas à 7 M », écrit dans docs/32 §22
avec les taux des quatre barreaux, et le CPU s'arrête pour les programmes 2 et 3
(docs/34 §7).
