# 09 — Premiers runs : un Prophet de 7M paramètres entraîné sur CPU, puis rendu agentique

> **Ce que ce document prouve :** que chaque étage du run A100 fonctionne sur des données
> réelles et produit un nombre vérifiable. **Ce qu'il ne prouve pas :** une capacité.
> À 7M paramètres et 2.4M tokens, aucune revendication n'est possible, et aucune n'est
> faite. Le poids résultant reste hors du dépôt (règle 4).

Aucun GPU dans cette session, et aucun accès réseau : le proxy sortant refuse
huggingface.co. Le corpus est donc le texte réel que la machine contient — la
documentation de ce dépôt et celle du système comme prose, les sources Python installées
comme code. `scripts/first_run_cpu.py` enchaîne les étages ; tout vit sous `--work`.

## Le chemin exercé

| Étage | Ce qui a tourné | Nombre |
|---|---|---:|
| Corpus | paragraphes de prose + fichiers Python, jeu tenu à l'écart découpé **avant** l'entraînement | 6745 + 2940 documents, 49.4 Mo ; 198 documents tenus à l'écart (1.04 Mo) |
| Tokenizer | Prophet-Tok v1, entraîneur incrémental | 3 584 fusions, vocabulaire 4 096, 2.87 octets/token |
| Décontamination | le jeu tenu à l'écart indexé en 13-grammes, tout document d'entraînement qui le répète est rejeté | 198 items indexés |
| Chargeur | deux phases (A 80 %, C 20 %), poids par source, reprise O(1) | 938 + 234 lots de 8 × 256 |
| Planning | WSD : 100 échauffement / 862 plateau / 210 décroissance ; Muon 0.02, AdamW 3e-3 | 1 172 pas, 2.40M tokens |
| Interruption | `--session-minutes 40` a arrêté le run sur un checkpoint propre au pas 1150 ; le second lancement a **repris** dans la même phase, sur le même flux, et fini le planning | 22 pas de reprise |
| Débit | balayage delta par blocs, 4 cœurs CPU | 2.12 s/pas, ≈ 967 tok/s |

## Le nombre

Bits par octet sur les 198 documents tenus à l'écart (375,630 tokens,
1,040,365 octets), même tokenizer, aucune position rembourrée comptée :

| | nats/token | bits/octet |
|---|---:|---:|
| Modèle non entraîné (uniforme sur 4 096 ids) | 8.347 | **4.348** |
| Après 1 172 pas | 4.192 | **2.184** |

Pour situer : l'entropie d'un modèle unigramme d'octets sur de l'anglais est d'environ 4.5
bits/octet, un trigramme d'octets descend vers 2.5–3, et les grands modèles publiés sont
entre 1.0 et 1.5 sur du texte général. 2.18 sur un mélange prose/code
avec 2.4M tokens dit : *le modèle apprend*, et rien de plus. C'est le premier chiffre du
projet mesuré par lui-même, sur ses propres outils.

## La courbe

Perte d'entraînement (nats/token, un lot de 2 048 tokens par point — donc bruitée) :

| Pas | Perte | LR | Phase |
|---:|---:|---:|---|
| 20 | 6.685 | 4.0e-03 | warmup |
| 100 | 6.287 | 2.0e-02 | plateau |
| 200 | 6.033 | 2.0e-02 | plateau |
| 300 | 3.757 | 2.0e-02 | plateau |
| 400 | 5.521 | 2.0e-02 | plateau |
| 500 | 5.355 | 2.0e-02 | plateau |
| 600 | 5.036 | 2.0e-02 | plateau |
| 700 | 5.016 | 2.0e-02 | plateau |
| 800 | 4.505 | 2.0e-02 | plateau |
| 900 | 4.801 | 2.0e-02 | plateau |
| 1000 | 4.478 | 1.2e-02 | decay |
| 1100 | 3.977 | 3.9e-03 | decay |
| 1160 | 3.420 | 6.3e-04 | decay |

## Ce que ce run a trouvé

**Le premier lancement a divergé au pas 280.** La perte était tombée de 8.35 à 4.56 au pas
260, puis NaN en un seul pas, avec l'écrêtage de gradient actif — et le run a ensuite
« entraîné » 900 pas de NaN en ayant l'air vivant. Reproduit sur la couche seule : avec
une porte d'oubli fermée (α < 1e-9), les gradients du balayage de référence sont finis,
ceux du balayage par blocs ne l'étaient pas. Deux causes, toutes deux dans la tenue de
livres en espace logarithmique du balayage par blocs :

1. il prenait `alpha.log()`, dont le gradient est 1/α — 1e30 dès qu'une porte se ferme —
   là où le balayage de référence multiplie par α et ne divise jamais ; il prend désormais
   le log-sigmoïde des logits de la porte, de gradient 1 − α, et le transmet aussi au
   noyau fusionné ;
2. il exponentiait toute la matrice des différences de log-décroissance cumulée **avant**
   de la masquer : au-dessus de la diagonale ces différences sont grandes et positives,
   leur exp est infini, et zéro fois l'infini est NaN dans le backward même si le forward
   a masqué la valeur. Le masque est désormais appliqué en espace log.

Tests de régression à −20, −40 et −90 sur le logit de porte. Et une garde dans le trainer :
un pas dont la perte ou la norme de gradient n'est pas finie n'atteint plus l'optimiseur
(le lot est consommé, le flux reste déterministe), et vingt de suite arrêtent le run avec
les nombres dans le message. Un run qui ne fait rien ne doit pas avoir l'air vivant.

C'est le **dixième défaut silencieux** du dépôt, et le premier trouvé par un entraînement
plutôt que par un test : les tests d'équivalence passaient, parce qu'aucun n'avait fermé
une porte.

## Ce que ce run ne dit pas

- Rien sur la qualité de l'architecture : à cette échelle, tout apprend.
- Rien sur le noyau fusionné `fla`, qui n'a toujours pas tourné ; `scripts/gpu_check.py`
  attend l'A100.
- Rien sur la halte apprise : `ponder/expected_depth` est resté entre 1 et 1.5 sur un
  planning de 1 172 pas, ce qui est le prior géométrique, pas un signal.

## Reproduire

```bash
python scripts/first_run_cpu.py --work /tmp/prophet-first-run --stage all \
    --minutes 40 --resume-minutes 8 --tokens 2.4e6
```

Le rapport (`report.json`) et les logs restent sous `--work`.

---

## Deuxième moitié : le premier chiffre agentique

`scripts/first_agent_run_cpu.py` reprend ce poids, active les têtes d'action typées
(`heads.action_head`, +49,697 paramètres) et l'entraîne sur 600 trajectoires parfaites
de la famille de tâches du benchmark à vérificateurs (`grep` du mot, `note` du fichier,
`done`), **rendues par le chemin qu'un épisode promu de la quarantaine emprunte**
(`prophet.agent.render`) — un épisode par ligne pour que chaque appel ait ses schémas en
contexte. 500 pas, 21 minutes, perte 8.16 → 0.030,
exactitude de sélection 100%, perte des têtes 2e-07, aucun pas non fini.
Puis le benchmark sur des tâches **jamais vues**, le vérificateur exécutable décidant.

Deux lancements, mêmes poids de départ, mêmes données, même graine ; une seule différence.

| | Succès (graine 7, 40 tâches) | Succès (graine 11, 40 tâches) | Valeurs copiées | Appels malformés |
|---|---:|---:|---:|---:|
| Têtes non entraînées | 0 % | — | 0 | 100 % |
| Run 1 : cibles de copie **absentes** (0 sur 40 valeurs) | 5.0% | 0.0% | 0 / 0 | 10.1% / 8.1% |
| Run 2 : cibles de copie **présentes** (40 sur 40) | **55.0%** | **32.5%** | 78 / 79 | 3.4% / 5.8% |

Le run 1 avait appris le flux d'entraînement parfaitement (sélection à 100 %, perte des
têtes ~1e-6) et ne transférait presque rien : un mot en prose est un token *avec son
espace* (« anchor»), le même mot en JSON est nu (« anchor »), donc aucune occurrence
verbatim ne commençait sur une frontière de token, le pointeur de copie n'a jamais eu une
seule cible, et le modèle devait *épeler* des valeurs qu'il ne pouvait que copier. Une
occurrence peut désormais commencer au token porteur d'espace, et le décodage retire cet
espace en épissant la valeur. C'est la trouvaille du run, et elle est de la classe des
dix autres : les tests d'équivalence passaient.

**Ce que ces nombres disent.** Que la mécanique du pilier — ancres, grammaire, sélection,
copie, portes du vérificateur — permet à un modèle de 7M paramètres d'apprendre une
famille de tâches depuis ses propres épisodes vérifiés, et que la copie d'arguments vaut
à elle seule la différence entre « rien » et « la moitié ». **Ce qu'ils ne disent pas :**
la famille de tâches est triviale, 40 tâches par graine donnent ±15 points, l'écart entre
graines (55 contre 32.5) est de cet ordre, et rien ici ne se compare à quoi que ce soit
d'existant. Le premier chiffre agentique du projet est un chiffre de mécanique, mesuré
sur ses propres outils, avec son intervalle.

Reproduire : `python scripts/first_agent_run_cpu.py --work /tmp/prophet-first-run --minutes 22 --seq-len 320`.
