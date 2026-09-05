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
lancement (16 clés, 3 000 pas, 165k paramètres, hasard à 6.25 %) :

| Distance de la paire à la question | attention complète | fenêtre seule | fenêtre + registre (tel que construit) |
|---|---:|---:|---:|
| ≤ fenêtre | 24.3 % | 13.6 % | **4.9 %** |
| 1 à 2 fenêtres | 19.9 % | 10.2 % | 5.6 % |
| > 2 fenêtres | 17.8 % | 7.5 % | 6.2 % |

Le contrôle apprend ; la fenêtre seule apprend dedans et décroît dehors ; le registre tel
que construit **n'a rien appris, même dans la fenêtre** où il devrait être une couche à
fenêtre. Deux causes, dans mon implémentation : la porte s'ouvrait à ½ dès le pas zéro,
donc les lectures d'une mémoire non entraînée noyaient le signal d'attention ; et l'ordre
des blocs à l'entraînement faisait lire au bloc *j* ce que les blocs ≤ *j*−2 seulement
avaient écrit. Corrigés (porte initialisée quasi fermée, σ(−4) = 0.018, testée ; écriture
du bloc *j*−1 avant la lecture du bloc *j*), le bras « registre » relancé donne 4.9 % /
9.3 % / 8.1 % : toujours le hasard dans la fenêtre. Mais le bras « fenêtre seule » n'était
pas le bon témoin — une couche SWA à RoPE, là où l'hôte du registre est une couche NoPE à
fenêtre — donc ce que ces 4.9 % disent, c'est qu'une couche NoPE à fenêtre n'apprend pas
ce rappel dans ce budget, registre ou pas. Le témoin propre — la couche à registre avec
sa porte clouée fermée, qui ne diffère du bras « registre » que par le terme de lecture —
a parlé :

| Distance de la paire à la question | attention complète | fenêtre RoPE seule | hôte NoPE, porte clouée fermée | hôte NoPE + registre (porte libre) |
|---|---:|---:|---:|---:|
| ≤ fenêtre | 24.3 % | 13.6 % | 12.6 % | **4.9 %** |
| 1 à 2 fenêtres | 19.9 % | 10.2 % | 10.2 % | 9.3 % |
| > 2 fenêtres | 17.8 % | 7.5 % | 10.6 % | 8.1 % |

Le témoin propre apprend dans sa fenêtre (12.6 %, comme la fenêtre RoPE) ; ouvrir la porte
au registre **nuit** à son hôte dans la fenêtre (4.9 %) et n'ajoute rien au-delà (9.3 et
8.1 contre 10.2 et 10.6). À cette échelle (165k paramètres, 3 000 pas, rappel synthétique),
le verdict est celui que le critère d'échec du plan prévoyait : **mémoire constante
prouvée, rappel au-delà de la fenêtre non démontré, coût dans la fenêtre mesuré** — D3b
reste à `"none"`. Ce que l'expérience ne dit pas : si un hôte plus large ou plus de pas
apprendraient à *lire* un registre qu'ils savent déjà écrire ; c'est l'ablation sur texte
réel (100M, dans `prophet.plan`) qui le dira, avec le même critère. L'ablation sur texte réel
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
| Après, avec 50 % de rejeu du corpus de base (mêmes 600 épisodes, même graine, 500 pas) | 15 % / 25 % | **2.61** |
| Après, cinq familles (800 épisodes, 600 pas) **avec** 50 % de rejeu | voir §3 (b) | **2.74** |

Le gradient seul, sur le flux d'expérience seul, efface tout le reste : le mur C tel que
`07_WALLS.md` le décrit, en un nombre. Le rejeu est la première parade et elle se mesure au
même endroit. Comparaison contrôlée — mêmes 600 épisodes, même graine, mêmes 500 pas, une
ligne sur deux tirée du corpus de base : l'effacement passe de +5.2 à **+0.43 bits/octet**,
et le succès sur tâches inédites de 55 % / 32.5 % à **15 % / 25 %**, parce que la moitié
des pas ne portent plus d'épisode (27 valeurs copiées contre 78). Sur les cinq familles à
la fois, même dose de rejeu : +0.55. Le rejeu est donc un **cadran**, pas une solution :
il échange la compétence contre la mémoire à coût linéaire — la moitié des tokens de
l'entraînement agentique servent à ne pas oublier — et c'est précisément la place des
mécanismes sans gradient (registre de sortie, écriture sur surprise, état de session porté)
que le benchmark mesure par la courbe par bloc, pour la voir plier — ou pas.

**L'état de session porté entre épisodes, mesuré.** Le mécanisme existe (R03 appliqué à
l'agent : l'état récurrent borné de l'épisode précédent restauré au début du suivant). Sur
le checkpoint agentique à 57.5 %, 40 tâches inédites, poids gelés, seule variable l'état
porté **[CPU, 7M]** :

| État au début de chaque épisode | Succès | Malformés |
|---|---:|---:|
| vierge | **57.5 %** | 4.7 % |
| porté de l'épisode précédent (tâche sans rapport) | **0 %** | 1.9 % |

Les appels sont bien formés et faux : l'état récurrent porte le contenu d'une autre tâche,
et le modèle n'a jamais vu un état porté à l'entraînement — chaque séquence y démarre d'un
état vierge. Même classe de trouvaille que les plafonds de profondeur par token : ce qui
n'est pas dans la distribution d'entraînement est indéfini à l'inférence, et le mécanisme
le plus correct du monde ne le sauve pas. La recette qui rendrait l'état porté utile est
un entraînement sur des *séquences d'épisodes* (l'état d'un épisode devient l'init du
suivant, avec des tâches liées et d'autres non) ; elle n'est pas construite. La
consolidation dans le registre de sortie entre blocs d'épisodes n'a pas été mesurée non
plus : la config CPU n'a pas de registre. Ces deux-là sont les prochaines lignes de la
courbe, pas des résultats.

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
compte chaque token traité par épisode et rapporte **tokens par succès**. Un modèle de 7M
paramètres, 600 pas sur 160 épisodes par famille, 30 tâches inédites par famille et par
graine **[CPU, 7M]** :

| Famille | Ce qu'il faut pour réussir | Succès (deux graines) | Tokens par succès | Malformés |
|---|---|---:|---:|---:|
| `calc` | copier l'expression du but, noter le résultat de l'outil | **60 % / 80 %** | 504 / 378 | 0 % |
| `lookup` | lire un JSON, copier la valeur d'un champ | 47 % / 27 % | 537 / 971 | 11–16 % |
| `files` | choisir `grep`, copier le mot, copier le nom de fichier | 7 % / 7 % | 4 732 / 4 643 | 12–21 % |
| `count` | choisir `count`, copier le mot, copier le nombre | 3 % / 0 % | 9 353 / ∞ | 48 % |
| `replace` | **générer** un fichier entier modifié | 0 % / 0 % | ∞ | 38 % |

Deux lectures. La première est celle de la conception : les familles où chaque valeur se
*copie* réussissent, celle où la valeur se *génère* est à zéro, et `count`, dont la sortie
d'outil est un seul token, échoue sur le choix d'outil (48 % de malformés) — à 160
épisodes par famille, un modèle de 7M n'a pas appris à distinguer cinq schémas. La seconde
est celle des tokens : un succès `calc` coûte 378 à 504 tokens, appel d'outil et copie
compris, contre un coût infini là où il faut générer ; à cette échelle, l'économie de
tokens d'un agent est d'abord la différence entre copier et générer. `files` à 7 % contre
55 % dans le run à une seule famille dit ce que 160 épisodes valent contre 600.

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
