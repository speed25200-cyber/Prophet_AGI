# 10 — L'architecture suivante : économe en tokens, à apprentissage continu, à contexte borné-infini

> Trois propriétés, chacune ramenée à une quantité qu'on mesure sur les outils du dépôt,
> et à un mécanisme qui la déplace. Les nombres marqués **[CPU, 7M]** viennent des
> premiers runs (`09_FIRST_RUN.md`) ; ils sont vrais à cette échelle et à aucune autre.
> Les cases marquées **en cours** attendent une expérience qui tourne ou est planifiée.
> Les nombres agentiques d'avant la cinquième mesure de `09_FIRST_RUN.md` (55 % / 32.5 %)
> portent deux décalages train/boucle corrigés depuis ; ils sont gardés pour l'écart qu'ils
> mesurent, pas comme référence.
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
au registre semblait **nuire** à son hôte dans la fenêtre (4.9 %) et ne rien ajouter au-delà.
Mais ces trois protocoles avaient un défaut commun, trouvé après coup : **le contrôle
n'apprenait pas** — 24 % pour l'attention complète là où le hasard est à 6 %, et aucun
bras ne se sépare d'un contrôle qui ne sépare rien. La cause est arithmétique. Avec
QK-norm, requête et clé sont normalisées et le logit d'attention est borné par
√`head_dim` × les gains appris : **4** à `head_dim` 16. Une clé parmi 160 ne peut donc
recevoir que e⁴ / (e⁴ + 159) = 26 % de la masse, et 24–25 % est exactement le plateau
observé. (`design_warnings()` le signale désormais sous `head_dim` 32 ; à 64, la borne est
e⁸ ≈ 3 000 clés et ce sont les gains appris qui doivent la dépasser — un point à
surveiller sur tout contexte long.) Protocole corrigé — QK-norm coupée, un saut au lieu de
deux, moitié des exemples dans la fenêtre, six paires interrogées par séquence au lieu
d'une — 16 clés, 6 paires, 3 000 pas, 165k paramètres, ~1 200 questions par ligne :

| Distance de la paire à la question | attention complète | fenêtre RoPE seule | hôte NoPE, porte clouée | hôte NoPE + registre |
|---|---:|---:|---:|---:|
| ≤ fenêtre | 45.3 % | 31.7 % | 43.8 % | 43.7 % |
| 1 à 2 fenêtres | 51.7 % | 37.3 % | 49.7 % | 49.5 % |
| > 2 fenêtres | 47.3 % | 36.3 % | 48.0 % | 49.2 % |

Trois lectures. Le contrôle apprend (perte 1.06 et en baisse : pas à saturation, mais
loin du hasard). Le registre **ne coûte rien** dans la fenêtre : le « coût » de 12.6 → 4.9
était un artefact du protocole borné. Et au-delà de la fenêtre, l'hôte à porte clouée fait
**aussi bien que l'attention complète** (48.0 contre 47.3) : à 6 paires, c'est l'état borné
du cœur delta qui rappelle, pas la fenêtre — la tâche tient dans l'état, et rien ne peut
départager un registre d'un état qui suffit. La fenêtre RoPE seule, elle, perd 12 points
partout : la couche NoPE est le meilleur hôte. Le test qui compte est donc celui où la
tâche **dépasse l'état**. Grossir la tâche ne marche pas : à 64 clés et 24 paires, le
contrôle à attention complète est au hasard après 4 000 pas (8.7 / 10.7 / 12.0 %, perte
2.69 pour 2.77 au hasard), comme au tout premier protocole. Rétrécir l'état, si : cœur
delta à une tête de 8 dimensions (~8 paires) contre 12 paires de 16 clés, même fenêtre,
6 000 pas — **en cours** (`--state-heads 1 --state-dim 8 --pairs 12`). Le verdict D3b reste
suspendu à ce nombre et à l'ablation sur texte réel (deux runs de 100M, BPB et rappel
multi-clés à 32k, dans `prophet.plan`) avec son critère d'échec : BPB dégradé de plus de
0.5 % ou rappel au hasard au-delà de la fenêtre, et le registre reste à `"none"`.

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

**L'état de session porté entre épisodes, mesuré — deux fois.** Le mécanisme existe (R03
appliqué à l'agent : l'état récurrent borné de l'épisode précédent restauré au début du
suivant, attention vide). Première mesure, sur le checkpoint à 57.5 % : **57.5 % → 0 %**
dès que l'état est porté, appels bien formés et faux. La lecture d'alors — « le modèle n'a
jamais vu d'état porté à l'entraînement » — était vraie mais pas première : en relançant
la recette, deux décalages train/boucle plus graves sont apparus (position de requête du
pointeur de copie, span de réflexion jamais rendu ; `09_FIRST_RUN.md`, cinquième mesure),
et une fois corrigés, le même budget (600 épisodes, 500 pas, 7M) donne — 40 tâches
inédites par graine, deux graines **[CPU, 7M]** :

| Recette | État vierge | État porté (40 épisodes à la suite) | Tokens par succès (vierge / porté) |
|---|---:|---:|---:|
| première (55 % / 32.5 %) | 55 % / 32.5 % | 0 % | 529 / ∞ |
| corrigée, 1 épisode par ligne | **95 % / 97.5 %** | **87.5 % / 90 %** | 294 / 321 |
| corrigée, 3 épisodes par ligne (concaténés) | 97.5 % / 97.5 % | 82.5 % / 75 % (courbe 1.00 → 0.50 le long de la session) | 284 / 376 |
| corrigée, 3 épisodes par ligne, **attention masquée par épisode** | 95 % / 100 % | 82.5 % / **92.5 %** (courbes 0.50 → 1.00 et plate) | 292 / 341, 276 / 301 |

Trois faits. La compétence de la recette était cachée par ses décalages : 55 % → 95–97.5 %
sans un paramètre de plus, avec 279 tokens par épisode au lieu de 304 (le span de
réflexion se ferme en un token). Un état porté, même jamais vu à l'entraînement, ne coûte
plus que 5 à 8 points : le « mur » de 57.5 → 0 était d'abord celui des défauts. Et la
recette naïve des séquences — épisodes concaténés dans une ligne — **aggrave** l'état
porté au lieu de l'améliorer, avec une courbe qui décroît le long de la session : le
modèle apprend à s'appuyer sur ce que l'attention lui montre de l'épisode précédent,
absente à l'inférence, et se dégrade dès que l'état s'éloigne de trois épisodes. Le
décalage restant est nommé et câblé : `ProphetModel.forward(segment_ids=)` masque
l'attention à chaque `<|bos|>` d'une ligne et laisse passer l'état récurrent — une ligne de
trois épisodes ainsi masquée **est** le chemin de la session portée, à 1e-4 (test
`tests/test_segments.py`), pas une approximation. Mesuré : le masque **supprime la
décroissance** (75 % décroissant jusqu'à 0.50 → 92.5 % plat sur la graine 11 ; 82.5 % avec
une courbe qui monte de 0.50 à 1.00 sur la graine 7) et ne coûte rien à l'état vierge
(95 % / 100 %). Il ne fait pas mieux, en moyenne, qu'un épisode par ligne (87.5 % contre
87.5 % / 90 %) : à 7M et 500 pas, l'état porté coûte encore 5 à 10 points quelle que soit
la recette, et c'est ce reste-là que mesurera l'échelle. La consolidation
dans le registre de sortie entre blocs d'épisodes n'a pas été mesurée : la config CPU n'a
pas de registre.

**Ce que l'état porté peut acheter : ne pas relire.** Des suites de tâches `lookup` sur le
même fichier (`make_related_tasks`) où un fichier lu par l'épisode précédent est répondu
sans le relire — moins de tokens à succès égal si l'état retient ce qu'il a lu, réponse
fausse sinon. Entraîné trois épisodes par ligne, **concaténés** (sans masque), 500 pas,
39 tâches inédites **[CPU, 7M]** :

| Banc (deux graines) | Succès | Fichiers non vus : succès, lectures/épisode | Fichiers vus : succès, lectures/épisode |
|---|---:|---:|---:|
| état vierge | 97.4 % / 100 % | 100 % / 100 %, 1.0 | 91.7 % / 100 %, 0.92 / 1.0 |
| état porté | **5.1 % / 2.6 %** (45 % / 39 % de malformés) | 7.4 % / 3.4 %, 1.0 | 0 % / 0 %, 1.0 |

À l'état vierge, le modèle **relit** un fichier « vu » (0.92 lecture par épisode, et la
seule fois où il ne relit pas, il se trompe) : la décision de ne pas relire est bien
conditionnée à l'état, pas à la position. À l'état porté — le seul cas où ne pas relire
aurait un sens — tout s'effondre, fichiers vus et non vus, 45 % d'appels malformés :
entraîné sur des lignes où l'attention voit l'épisode précédent, ce modèle s'appuie sur
elle et rien d'autre, et l'état seul lui est étranger. Même cause que la ligne du bras à
trois épisodes ci-dessus, en plus fort. Avec le masque par épisode
(`--related --segment-attention`), même budget :

| Banc (deux graines) | Succès | Fichiers non vus : succès, lectures/épisode | Fichiers vus : succès, lectures/épisode |
|---|---:|---:|---:|
| état vierge | 97.4 % / 100 % | 100 % / 100 %, 1.0 | 91.7 % / 100 %, 1.0 |
| état porté | **10.3 % / 5.1 %** (0 / 1.3 % de malformés) | 11.1 % / 3.4 %, **0.04 / 0.03** | 8.3 % / 10 %, **0.0 / 0.0** |

Le masque a fait son travail : plus un appel malformé, et la décision de ne pas relire est
prise **sur l'état seul** — dès qu'un état est porté, le modèle saute la lecture (0.04
lecture par épisode). Mais il la saute pour *tous* les fichiers, vus ou non, et répond
faux : l'état borné du cœur delta, à 7M paramètres et 500 pas, porte le fait qu'une lecture
a eu lieu, pas son contenu, ni le nom du fichier qui permettrait de distinguer « ce
fichier » d'« un autre ». Le gain de tokens qu'il achèterait est visible (218 contre 226
par épisode) et inutile tant que la réponse est fausse. C'est le nombre honnête de la
mémoire de travail entre épisodes à cette échelle : la **mécanique** (masque, état porté,
décision conditionnée) est prouvée, la **capacité** ne l'est pas, et c'est elle que
l'échelle et le registre de sortie (absent de la config CPU) doivent apporter.

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

**(a′) La profondeur latente contre les tokens de chaîne de pensée.** Le test le plus
direct du pari : mêmes poids, `k` passes du cœur et un seul token de réponse, contre une
passe et la chaîne de pensée émise token par token (`scripts/depth_cpu.py`, opérations
séquentielles modulo 10, monoïde non résoluble, donc pas de raccourci logarithmique à
apprendre). Premier lancement à 6 opérations, 150k paramètres, 3 000 pas **[CPU]** : k=1
31.2 %, k=2 30.7 %, k=4 31.2 %, chaîne de pensée 28.9 % — hasard 10 %. **Insensible** :
la chaîne, qui n'a qu'une opération par token à faire, n'apprend pas non plus, donc rien
n'est séparé (même QK-norm à `head_dim` 16 que le needle). Balayage sans QK-norm, 3 000
pas par bras, 1 024 expressions d'évaluation **[CPU, 150k]** :

| Opérations | k=1, réponse directe | k=2 | k=4 | k=1, chaîne (un token par opération) |
|---:|---:|---:|---:|---:|
| 1 | **100 %** | 100 % | 100 % | 100 % |
| 2 | 30.7 % | 30.9 % | 30.7 % | **45.3 %** |
| 3 | 32.1 % | 32.0 % | 32.5 % | 41.7 % |
| 4 | 29.1 % | 29.4 % | 30.6 % | 40.1 % |
| 6 | 30.1 % | 32.0 % | 31.4 % | 34.3 % |

Une opération s'apprend à 100 % dans tous les bras : le contrôle passe. Dès la deuxième,
la réponse directe tombe à ~31 % **quelle que soit la profondeur** — k=1, 2 et 4 finissent
à la même perte à la troisième décimale (1.351) : quatre passes du cœur n'apprennent
pas une composition de plus qu'une seule, à ce budget. La chaîne émise achète 10 à 15
points, pas la solution : sa fiabilité par pas est d'environ 57 % (chaîne exacte 19 % à
trois opérations), là où l'opération isolée est à 100 % — c'est l'entraînement qui
manque, pas la mécanique. Le pari « la profondeur latente remplace les tokens de
chaîne » est donc, à 150k paramètres et 3 000 pas, perdu des deux côtés à égalité. À
12 000 pas, le budget et la capacité se séparent :

| Opérations (12 000 pas) | k=1 direct | k=4 direct | k=1, chaîne |
|---:|---:|---:|---:|
| 2 | 29.6 % | 29.3 % | **100 %** (chaîne exacte 100 %) |
| 3 | 31.3 % | 30.9 % | 42.1 % (chaîne exacte 20 %) |

Le verdict est net et va **contre** le pari, à cette échelle : quatre fois plus de pas ne
déplacent pas d'un point la réponse directe, à k=1 comme à k=4 (perte 1.348 dans les deux
cas), tandis que la chaîne émise apprend la composition de deux opérations **à 100 %**
avec deux tokens. Deux tokens de chaîne achètent ce que quatre passes latentes du même
cœur n'apprennent pas du tout. À trois opérations la chaîne stagne à son tour (42 %, ~58 %
par pas, inchangé entre 3 000 et 12 000 pas) : un plafond d'apprentissage de ce modèle de
150k paramètres sur la localisation, pas une propriété de la chaîne, et hors sujet ici.
Ce que ce nombre dit du projet : le pari central de Prophet — la profondeur latente
remplace les tokens de raisonnement — n'a **aucun** support à petite échelle, ni en
bits/octet (§3a) ni en composition ; il ne se joue qu'au-dessus de la porte R04 (≥ 350M),
et tout ce dépôt peut faire d'ici là est de garder la boucle réversible (elle l'est :
`train_loop_min = train_loop_max = 1`, `halting = "none"`) et le test prêt.

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
