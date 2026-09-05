# 10 — L'architecture suivante : économe en tokens, à apprentissage continu, à contexte borné-infini

> Trois propriétés, chacune ramenée à une quantité qu'on mesure sur les outils du dépôt,
> et à un mécanisme qui la déplace. Les nombres marqués **[CPU, 7M]** viennent des
> premiers runs (`09_FIRST_RUN.md`) ; ils sont vrais à cette échelle et à aucune autre.
> Les cases marquées **en cours** attendent une expérience qui tourne ou est planifiée.
> Rien ici n'est adopté dans une configuration livrée sans son ablation (règle 2).

## 0. Ce que les mots veulent dire, en quantités

| Propriété | La quantité | Le mécanisme qui la déplace | Où c'est mesuré |
|---|---|---|---|
| Contexte infini | octets de cache par token en fonction de la longueur de contexte : doit devenir **constante** | attention à registre (D3b) : fenêtre exacte + registre borné des KV évincés | `prophet.budget`, §1 |
| Apprentissage continu | compétence gagnée **contre** mémoire effacée : succès agentique sur tâches inédites et bits/octet tenus à l'écart, mesurés ensemble | chemin quarantaine → corpus, rejeu, consolidation dans le registre, état de session | `09_FIRST_RUN.md`, §2 |
| Économie de tokens | (a) bits/octet à tokens d'entraînement égaux ; (b) tokens par succès agentique | (a) profondeur bouclée à paramètres constants ; (b) actions typées et copie plutôt que génération | §3 |

## 1. Contexte infini : la courbe qui s'aplatit

Le seul endroit où la mémoire de la pile croissait avec le contexte était l'attention
globale. `LedgerAttention` (`mixer.global_memory="ledger"`) attend exactement dans une
fenêtre et écrit chaque paire (clé, valeur) évincée dans un registre à clés-produit adressé
par la clé ; chaque requête relit le registre à sa propre adresse, par une porte par tête.
Un registre vide lit zéro : dans la fenêtre, la couche *est* une couche à fenêtre glissante
(testé à 1e-6). Les registres sont persistés avec la session.

| Contexte | Cache Prophet-main (Go) | Avec registre (Go) |
|---:|---:|---:|
| 131 072 | 0.548 | 0.062 |
| 1 048 576 | 4.307 | 0.062 |
| 8 388 608 | 34.371 | **0.062** |

(bf16, fenêtre 4 096, 16 384 emplacements par couche globale, 48 paramètres ajoutés.)

**Ce que ça coûte.** Au-delà de la fenêtre, le rappel devient associatif — un nombre borné
d'emplacements, adressés doucement. `scripts/needle_cpu.py` le mesure : trois modèles
identiques (attention complète comme contrôle d'apprenabilité, fenêtre seule, fenêtre +
registre) entraînés sur un rappel clé→valeur, exactitude par distance dans et au-delà de
la fenêtre. Premier lancement **[CPU]** : inconclusif — 64 clés, 1 500 pas, aucun bras
n'apprend même dans la fenêtre (≈ 2 %, le hasard à 1/64), donc rien n'est séparé. Second
lancement (16 clés, 3 000 pas, bras de contrôle) : **en cours**. L'ablation sur texte réel
(deux runs de 100M, BPB et rappel multi-clés à 32k) est dans `prophet.plan` avec son
critère d'échec : BPB dégradé de plus de 0.5 % ou rappel au hasard au-delà de la fenêtre,
et le registre reste à `"none"`.

## 2. Apprentissage continu : la hauteur du mur, puis la première parade

Le mécanisme existe de bout en bout : épisode → quarantaine (provenance, tiers) →
promotion → rendu dans le flux à ids de contrôle → source du chargeur → cibles des têtes
d'action. Il a été exercé sur de vrais poids. Ce qu'il coûte est maintenant un nombre.

| Fine-tune agentique, 500 pas, 600 épisodes **[CPU, 7M]** | Succès tâches inédites | Bits/octet tenus à l'écart |
|---|---:|---:|
| Avant | 0 % | 2.18 |
| Après, sans rejeu | 55 % / 32.5 % (deux graines) | **7.36** (un modèle vierge : 4.35) |
| Après, avec 50 % de rejeu du corpus de base | **en cours** | **en cours** |

Le gradient seul, sur le flux d'expérience seul, efface tout le reste : le mur C tel que
`07_WALLS.md` le décrit, en un nombre. Le rejeu est la première parade et elle se mesure au
même endroit ; la consolidation sans gradient (registre de sortie, écriture sur surprise)
et l'état de session porté entre épisodes sont les suivantes, et le benchmark rapporte la
courbe par bloc pour les voir plier — ou pas.

## 3. Économie de tokens

**(a) À l'entraînement.** Le pari central — la profondeur bouclée achète de la profondeur
sans paramètres — n'avait jamais été mesuré par le projet. À iso-paramètres, iso-tokens
(2.4M), iso-données, iso-graine et iso-planning **[CPU, 7M]** :

| Profondeur du cœur | E[k] | Passes de cœur par token | Bits/octet tenus à l'écart |
|---|---:|---:|---:|
| bouclée, *k* ∈ {1..4} log-uniforme, halte apprise | 2.14 | 2.14 × 2 blocs | 2.184 |
| constante, *k* = 1 | 1 | 1 × 2 blocs | **2.179** |

À cette échelle la boucle n'achète **rien** et coûte 2.1× les FLOPs du cœur : l'écart
(0.005 bits/octet) est dans le bruit, dans le mauvais sens. C'est le résultat que R04
annonçait — la récurrence sous-performe une pile simple à 135M et ne gagne qu'à partir de
~360M — et la raison pour laquelle la porte R04 du plan est à ≥ 350M. Le pari central
reste un pari ; ce nombre dit seulement qu'il ne se gagne pas petit, et qu'un run qui le
testerait en dessous de l'échelle où il peut gagner produirait exactement ce faux négatif.

**(b) À l'inférence.** Un agent qui *copie* un argument le paie un pas au lieu de douze,
et un agent qui appelle un outil au lieu de raisonner en tokens paie l'appel. Le benchmark
compte désormais chaque token traité par épisode et rapporte **tokens par succès** ; les
runs en cours le remplissent pour les cinq familles de tâches (`prophet/agent/tasks.py`).

## 4. Les datasets, prêts

| | Script | État |
|---|---|---|
| Corpus de pré-entraînement | `scripts/prepare_corpus.py` : sources du mélange en shards locaux, filtres, plafonds, provenance, reprise | prêt ; `--dry-run` sans réseau ; les ids restent à vérifier (`scripts/verify_datasets.py`) |
| Jeux de test à décontaminer | `scripts/fetch_benchmarks.py` | prêt ; ids marqués à vérifier ; jeu vide refusé |
| Corpus agentique | `scripts/build_agent_dataset.py` sur cinq familles vérifiables | prêt et généré à volonté, sans licence ni fuite possible |
| Tokenizer | `scripts/train_tokenizer.py` | prêt (incrémental) |

## 5. Ce qui reste hors de portée ici

Un GPU et le réseau. Tout ce qui précède tourne sur 4 cœurs et 7M paramètres, ce qui
suffit à prouver la mécanique et à trouver ses défauts (deux de plus en une journée,
`CLAUDE.md`), et à rien d'autre. Les nombres qui comptent sortiront de `prophet.plan`,
dans l'ordre qu'il donne.
