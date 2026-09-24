# 35 — Carte des hypothèses : vers un modèle plus petit qui s'améliore lui-même

**Statut : v1, 2026-09-23.** Tous les tests CPU de la liste de §3 sont faits ou écartés avec leur raison ; ce qui reste tranche à l'A100. Relie chaque pari du dépôt au but, avec ce qui est établi,
réfuté ou ouvert, et le test qui tranche. Chaque chiffre renvoie à sa section d'origine. Les
chiffres marqués `[S]` viennent de résumés de recherche que les docs sources n'ont pas
vérifiés sur le texte intégral. Ce document ne remplace aucun pré-enregistrement : il dit
lequel lancer ensuite, et pourquoi.

## 0. Le but, mis en conditions mesurables

Le but du dépôt, dit par son auteur : **une IA avec moins de paramètres, qui s'améliore
elle-même sans fin, au-delà de la frontière**. Avec les contraintes de CLAUDE.md (un A100,
≈ 300 heures, inférence sur 5090, Mac, iPhone), il se décompose en six conditions. Chacune
est nécessaire ; aucune ne suffit seule.

| | Condition | Question mesurable |
|---|---|---|
| **C0** | Le compute est ce qu'il est | Que peut-on acheter avec ≈ 300 A100-h ? |
| **C1** | Plus de capacité par paramètre | Un cœur bouclé (profondeur au lieu de largeur) vaut-il un modèle plus gros, à FLOPs égaux ? |
| **C2** | Apprendre de soi | Le modèle progresse-t-il sur ses propres épisodes vérifiés par un programme ? |
| **C3** | Se donner du travail | Invente-t-il des tâches au-delà du générateur écrit par un humain ? |
| **C4** | Ne pas s'user | Ce qu'il apprend efface-t-il ce qu'il savait (langue, autres familles) ? |
| **C5** | Déplacer le plafond | Qu'est-ce qui élargit l'ensemble de ce qu'on sait vérifier ? |

**« Sans fin » et « au-delà de la frontière », pris au sérieux.** Quatre bornes sont
établies ou démontrées dans le dépôt. Aucune revendication ne peut les ignorer.

1. **Le vérificateur borne l'auto-amélioration.** Ce que le modèle rate toujours, il ne
   l'apprend jamais seul (docs/32 §14). Et s'améliorer contre son propre juge n'ajoute
   aucune information que le juge n'a pas (08 A4 §1.2, « sharpening »). En dessous
   d'environ 7B, l'écart génération–vérification est ≤ 0 `[S]`.
2. **Des poids fixes ne retiennent pas tout.** « Zéro oubli pour toujours » est impossible
   à poids fixes (08 §1.2 ; W3 §1.2). 229M paramètres tiennent ≈ 57 Mo de faits (W3 §1.2).
3. **Le compute.** 300 A100-h à 35 % de MFU font 1,18e20 FLOPs : **73×** sous SmolLM2-360M
   et **3 114×** sous Qwen3-1.7B (00 §0).
4. **La profondeur constante n'achète aucune classe de complexité** ; seule une profondeur
   qui dépend de l'entrée le peut (W1 §2.6, W2 §2.1 `[S]`).

« Au-delà de la frontière » ne peut donc vouloir dire qu'une chose mesurable : **plus de
capacité par paramètre et par heure de compute, sur des familles de tâches vérifiées par
programme, qui monte de tour en tour sans tâches écrites par un humain, sans oubli**. La
mesure est la pente de cette courbe (§2), pas un mot. « Sans fin » veut dire : la pente
reste positive quand on retire le générateur (H23, docs/33), et chaque plafond atteint est
nommé.

## 1. Le registre, condition par condition

Statuts : **établi** (chiffre reproduit ou démontré), **réfuté** (critère pré-enregistré
échoué), **ouvert** (test défini, pas lancé), **non testé** (idée sans test défini).

### C1 — Capacité par paramètre (profondeur au lieu de largeur)

| Hypothèse | Statut | Chiffres | Réf. |
|---|---|---|---|
| Boucler achète de la profondeur à 7 M | **réfuté à 7 M** | bouclé *k* ∈ {1..4}, E[k] = 2,14 : 2,184 bit/octet ; *k* = 1 : **2,179** ; 2,1× les FLOPs du cœur pour rien | 10 §3a |
| Cœur partagé (374,7M, *k* = 4) ≈ pile non partagée (920,7M) à 20 blocs exécutés | **ouvert, graine 0 un peu défavorable** | +0,3015 % de BPB pour **59,30 % de paramètres en moins** ; +0,0116 nat/token [+0,0083, +0,0149] au pas 4 096 ; le signe a changé en cours de route ; graines 1 et 2 jamais entraînées | 17 |
| Plus de boucles à l'inférence aident un modèle entraîné à *k* = 4 | **réfuté** | *k* = 8 : **+16,16 %** de BPB | 17, 23 |
| Entraîner à profondeur variable (2..6) rend *k* = 6 meilleur que *k* = 4 | **réfuté (512 pas de reprise)** | *k*6/*k*4 = **1,0043** (critère ≤ 0,995) | 23 |
| Réinjection apprise | **réfuté** (4 critères sur 4) | *k*6 2,45 % pire que son *k*4 | 25 |
| La norme de l'état explose avec les boucles | **observé, cause ouverte** | RMS 1 058 → 7 248 en 8 boucles | 24 ; 29 §7 |
| La profondeur latente compose (plusieurs sauts sans jetons émis) | **réfuté en petit** (150k paramètres, CPU) | meilleure des 7 variantes **50,7 %** à 2 opérations ; chaîne émise **100 %** | 10 §3a″ |
| β ∈ (0, 2) rend la règle delta capable de parité | **établi en petit** | hasard → **0,996** | 07, W2 |
| La conversion d'un donneur garde sa qualité | **réfuté** | meilleur récupéré 4,297 contre 3,158 nats pour le donneur ; ARC-Easy 35 % contre 61 % ; élaguer 8 couches (5,188) bat tout partage | 18–22 |
| Récurrence sur donneur retenu, identité exacte à *k* = 1 | **établi (identité)**, qualité **non testée** | logits identiques au bit près | 28 |
| Registre d'attention à mémoire constante | **établi (mémoire)**, rappel faible | 34,4 Go → 0,062 Go à 8,4M tokens ; rappel +5,2 ± 1,2 points (NoPE) | 10 §1–2 |
| Le cœur bouclé compose des consultations, et *k* lié aux sauts porte au-delà (H29–H31, miniature d'H4 et de W2-A5/A6) | **non conclusif** (contrôle à 1 saut raté par 7 modèles sur 8) ; relance pré-enregistrée | le seul modèle qui consulte (GDN, *k* lié) ne compose pas : 0,21 à 2 sauts | 36 |
| Boucler remplace la taille d'état pour le rappel (H32, F4 de W2 contredite ?) | **non décidé** (contrôle avec attention raté) ; relance du contrôle en cours | descriptif : à *d_k* = 8, *k* = 4 rattrape +0,34 ; à *d_k* ≥ 16, rien | 37 |
| Nos couches d'attention apprennent une consultation à petite taille | **établi sous condition** : à *d* = 64, `init_std` 0,02 + embeddings liés + GQA la rendent aléatoire (1 graine sur 4) ; corrigés, 4 sur 4 en ≈ 800 pas, comme un transformeur minimal. Les miniatures à attention du dépôt (docs/10, 36, 37) sont à reprendre | 38 |

**Le test qui tranche C1 : le programme 1** (docs/29 ; ≈ 70 + 6 A100-h). Il compare, à
blocs exécutés égaux, un cœur GDN à état borné, un cœur attention et une pile de 920,7M,
sur trois graines : mémoire (H1), qualité (H2, ≤ 1,02), courbe de profondeur (H3),
composition à 3–4 sauts (H4, **le premier test de composition à l'échelle**) et
quantification (H5). Il est prêt et n'a jamais été lancé. Sans lui, C1 reste un pari.

### C2 — Apprendre de ses propres épisodes vérifiés

| Hypothèse | Statut | Chiffres | Réf. |
|---|---|---|---|
| La boucle fermée apprend | **établi à 7 M** (v4) | +0,256 de gain moyen, intervalles > 0 sur 3 graines ; rendement **0,92** contre l'oracle | 32 §9 |
| L'oubli vient de la recette | **établi** | +0,44 → **+0,07** bit/octet au taux ÷ 4, sans perte de gain | 32 §8 |
| Le vérificateur contredit les erreurs systématiques | **réfuté** | graine dure : **+0,000** [−0,100, +0,100] contre +0,300 pour l'oracle ; 18 tâches jamais réussies sur 60 | 32 §14 |
| La reprise qui explore le pointeur crée la contradiction | **établi, 1 graine, 1 famille** | 0,583 → **0,817** ; dose-réponse 0 / 3 / 13 épisodes → +0,000 / +0,133 / +0,233 | 32 §16 |
| … aussi quand la bonne valeur est loin dans le pointeur | **réfuté** | rang 11 et 15 (`files`) ; 5 tours ne suffisent pas | 32 §17–18 |
| … aussi quand l'étape manque (`done` prématuré) | **réfuté** (v10d) | 18 échecs sur 30, tous `read_file` puis `done` refusé | 32 §21 |
| Le `done` prématuré est une étape que la politique n'émet pas | **réfuté : un défaut de décodage** (H28) | `no_repeat_action` interdisait le substitut `verify`, pas le `done` émis ; corrigé : banc 0,667 → **0,967**, hors distribution 0,267 → 0,617, ce mode 16 → 0 | 32 §24 |
| Un signal négatif sur les échecs (KLPO) | **réfuté à 7 M**, 3 fois | effondrement à β = 0,1, oscillation à β = 1,0 | 32 §6, §8 |
| Plusieurs familles dans une boucle | **établi : oubli par omission total** ; bassin cumulé protège | une famille absente d'un tour : 1,000 → **0,000** | 32 §15, §19 |

**Tests ouverts, CPU :** ~~un a priori de décodage contre le `done` prématuré~~ (fait, H28,
32 §24) ; un curriculum qui rend atteignable l'ensemble jamais réussi (32 §14,
voie 3) ; KLPO avec β entre 0,1 et 1,0 ou des tirages frais à chaque pas (32 §8). **Test
qui tranche à l'échelle :** `scripts/closed_loop_a100.sh` sur le checkpoint du programme 1.
Il n'a jamais tourné sur GPU.

### C3 — Se donner du travail (le modèle propose ses tâches)

| Hypothèse | Statut | Chiffres | Réf. |
|---|---|---|---|
| H20 : le modèle amorcé propose des tâches valides (≥ 50 %), format en deux listes | **réfuté à 7 M** | au mieux **9 sur 30** (0,30), solveur à 0,950 | 32 §22 |
| Le décodage admettait ce que JSON refuse | **défaut trouvé et corrigé** | 27 malformées sur 30 au premier barreau à cause de lui | 33 amend. 8 |
| H26 : les champs en objet, dans la forme que le solveur lit | **réfuté sur le critère**, verrou des champs levé | 12 valides sur 30 (0,40), solveur 0,933 dans la fenêtre ; champs justes 0 → 23 sur 29 à dose égale | 32 §23 |
| H27 : la clé demandée copiée par le pointeur | **réfuté** | 6 valides sur 30 : le pointeur copie la valeur au lieu de la clé | 32 §23 |
| H21–H25 : bord de compétence, transfert, **portée hors distribution**, oubli, nouveauté | **jamais lancées** pour `lookup` | la calibration n'est jamais passée | 33 §3 |
| **SI-1 : un champ unique (`calc`) fait démarrer le proposeur** | **établi à 7 M** | 28 valides sur 30 dès le premier barreau, 0,87 à 0,97 à chaque tour | 39 §3 |
| SI-1, H23c : proposer va au-delà du générateur | **réfuté à 7 M** | banc à trois opérandes −0,10 [−0,20 ; −0,017], témoin −0,033 ; 4,3 % de propositions nouvelles | 39 §3 |
| Pourquoi : la récompense du proposeur n'a jamais été versée | **établi** | 0 proposition résolue à une reprise en 5 tours : l'exploration tire le début du span, et les échecs sont des fins coupées | 39 §3 |
| SI-1b, H26c : quand une reprise peut rattraper, la boucle qui propose va au-delà du générateur | **établi à la graine 0, non reproduit** | graine 0 : 0,05 → **0,967** (témoin 0,217), écart entre bras +0,75 [+0,63 ; +0,85] ; graines 1 et 2 : écart +0,067, intervalles incluant zéro | 39 §3, amend. 2–3 |
| Pourquoi la réplication échoue : le proposeur n'écrit pas de nouveauté au départ | **établi** | propositions nouvelles au tour 1 : 5/29 à la graine 0, 0/30 et 0/29 aux graines 1 et 2 ; le mécanisme ne démarre pas | 39 §3 |
| SI-1c : 120 propositions par tour rendent le mécanisme fiable | **réfuté** (2 graines sur 3) | graines 0 et 1 : écart +0,717 et **+0,800** (la graine 1 ne démarrait pas avec 30) ; graine 2 : 0,000, aucune nouvelle promue | 39 §3 |
| **Règle des six runs** : la boucle dépasse son générateur si et seulement si au moins une proposition nouvelle est rattrapée et promue | **établi à 7 M** (`calc`) | 3 runs avec ≥ 2 promues : +0,72 à +0,80 ; 3 runs sans : 0,000 à +0,067 | 39 §3 |
| **SI-8a : plafond des règles relevé, la boucle monte les marches suivantes** | **établi à la graine 0**, graine 1 en cours | cinq chiffres : +0,367 [+0,217 ; +0,517] contre le plafond 4 ; six : +0,133 [+0,050 ; +0,217] ; propositions promues jusqu'à 8 chiffres | 39 §3 |
| Sans propositions au-delà, le progrès s'érode | **observé** (graine 0) | cinq chiffres au plafond 4 : 0,55 (tour 1) → 0,32 (tour 10) ; au plafond 8 : de 0,68 à 0,80 | 39 §3 |
| Une récompense échantillonnée (AZR) paie la frontière à 7 M | **réfuté** (sonde) | échecs gloutons rattrapés par un tirage : 2 sur 18, 1 sur 38 ; la frontière est systématique ; porte pré-enregistrée pour 375 M | 39 amend. 6 |
| SI-1d : une part fixe du flux pour les propositions empêche l'oubli du format | **établi** (graine 1, où l'oubli arrivait 3 fois sur 3) | validité au tour 2 : 0,0 → **0,967** ; quatre chiffres +0,717 contre le témoin ; coût −0,083 contre SI-1c | 39 §3 |
| SI-1b : le proposeur bouge quand sa récompense est versée | **observé**, critère M échoué | récompense versée aux tours 1–2 (sur les 5 puis 1 propositions nouvelles) ; part de nouvelles 17 % → 52 % ; puis plus de bord : le solveur a rattrapé | 39 §3 |
| SI-1b, H23c : le gain se transfère à un autre axe (troisième opérande) | **réfuté à 7 M** | 0,0 (4 gagnées, 4 perdues) ; aucune proposition à deux opérateurs sur 135 | 39 §3 |
| Deux défauts qui tuaient SI-1 : échappement dans un nom d'outil ; porte de copie entraînée là où le proposeur ne l'interroge pas | **trouvés et corrigés** | 30 malformées sur 30 ; logit de la porte +9,32 → −1,16, banc du solveur 1,0 → 0,0 | 39 amend. 1 |

H23 est celle qui compte pour le but. **Proposer va plus loin que le générateur** sur un
banc hors de sa distribution. Sans elle, proposer n'est qu'un générateur plus cher.

### C4 — Ne pas s'user (oubli, mémoire)

| Hypothèse | Statut | Chiffres | Réf. |
|---|---|---|---|
| Un affinage agentique sans rejeu détruit la langue | **établi** | 2,18 → **7,36** bit/octet (un modèle vierge : 4,35) ; avec rejeu 2,61 mais succès 15 %/25 % | 10 §2 |
| Avec rejeu 0,5, l'oubli croît avec la longueur de la boucle | **observé** (SI-8a graine 0, les deux bras) | ΔBPB +0,10 en 5 tours, **+0,37 en 10 tours** ; le mur d'une boucle longue | 39 §3 |
| SI-8b : un rejeu de 0,75 tient la langue sur 10 tours | **réfuté** | ΔBPB +0,276 au lieu de +0,371 (−26 %), la dérive accélère toujours ; il dilue les propositions (4 tours à 0 valide) | 39 §3 |
| L'oubli vient de la relance d'un planning complet à chaque tour | **établi** (SI-8c, graine 0) | taux ÷ √r : ΔBPB +0,371 → **+0,125** en 10 tours, accroissement constant (+0,017 par tour) au lieu d'accélérer ; la marche intacte (+0,417 contre le plafond 4) | 39 §3 |
| Porter l'état entre épisodes aide | **réfuté** sans entraînement dédié | 57,5 % → **0 %** au banc, poids gelés | CLAUDE.md (tableau des défauts) |
| Mémoire à deux niveaux, écriture « sommeil » sans gradient (R03, E2) | **ouvert** | ≈ 5 h + évaluations ; 20 h financées | 07, 06 |
| Distiller dans les poids (W3), ratio σ transfert / rappel | **non testé** | W3-E0 à E6 : 13,5 A100-h | W3 §7 |
| La consolidation se compose (W4) | **non testé** | porte 0 : ≈ 1 h ; arrêt si acc(*k* = 16) − acc(*k* = 2) < 3 points | W4 |

### C5 — Déplacer le plafond (ce qu'on sait vérifier)

| Hypothèse | Statut | Chiffres | Réf. |
|---|---|---|---|
| Un vérificateur appris peut promouvoir | **refusé par principe** | AUROC 0,80 → ≈ 30 % de fausses réponses admises ; seul le tier 0 (un programme) promeut | 08 §4 |
| Le désaccord entre profondeurs signale l'erreur (A4-0) | **ouvert**, « gratuit » | < 5 min, une passe à *k* = 8 sur ≈ 1 500 exemples ; arrêt si AUROC < 0,65 | 05, 08 |
| Familles vérifiables existantes | **cinq** (`calc`, `lookup`, `files`, `count`, `replace`) | premier run agentique : `count` 3 %/0 %, `replace` 0 %/0 % | 09 |
| Des tâches comme programmes, l'exécuteur comme juge (façon AZR) | **non testé** | élargirait l'ensemble vérifiable sans juge appris `[S]` | 33 §1 |

## 2. À quoi ressemblerait « s'améliorer sans fin », en chiffres

L'auto-amélioration se mesure par trois courbes, tour après tour, à compute compté :

1. le gain sur le banc du générateur (le programme 2 l'a : +0,256 en cinq tours) ;
2. **le gain sur un banc hors distribution**, que le générateur ne produit jamais (docs/33
   le définit ; **mesuré pour la première fois en SI-1b** : 0,05 → 0,967 sur des opérandes
   de quatre chiffres, contre 0,217 pour le témoin, une graine, docs/39 §3) ;
3. **la taille de l'ensemble jamais réussi**, qui doit baisser (18 → 6 sur une graine en
   §16).

« Sans plafond fixé d'avance » veut dire : la courbe 2 monte encore quand le générateur est
retiré. Chaque fois qu'une courbe s'aplatit, le dépôt a nommé le mur. Aujourd'hui, à 7 M :

| Mur | Où il est | Ce qui le déplace (hypothèse) |
|---|---|---|
| **Amorçage** : il faut déjà réussir parfois | `calc` passe de 0 à 100 % entre 40/80 et 100/200 (31 §2) | une amorce calibrée famille par famille (fait) |
| **Vérificateur** : il filtre, il ne contredit pas | graine dure : +0,000 (32 §14) | la reprise qui explore (+0,233, 32 §16) ; un curriculum (non testé) |
| ~~**Étape absente** : `done` prématuré~~ | 32 §21 | **levé** : c'était un défaut de décodage (H28, 32 §24) |
| **Proposeur** : il ne démarre pas | 0,40 de validité au mieux pour `lookup` (32 §22–23) | **levé pour `calc`** : un champ unique, 28 valides sur 30 (39 §3) ; pour `lookup`, cibler des clés à l'entraînement du pointeur, l'échelle 375 M |
| ~~**Récompense du proposeur jamais versée**~~ | SI-1 : 0 proposition résolue à une reprise en 5 tours (39 §3) | **levé** : fins restreintes aux fins de mot sur les reprises (SI-1b, récompense versée aux tours 1–2) |
| **Le proposeur n'explore pas la structure** | SI-1b : 0 proposition à deux opérateurs sur 135 ; les nombres plafonnent à quatre chiffres, le bord disparaît au tour 3 (39 §3) | une nouveauté tirée sur la structure, pas seulement sur les valeurs (non testé) |
| **Une reprise n'ajoute pas ce qui manque** | un opérande omis finit sur une fin de mot légitime ; 4 sur 13 rattrapés au mieux (39 amend. 2) | une reprise qui relit le but, ou qui pense avant d'agir (non testé) |
| **Familles écrites par un humain** | 33 §5 | des tâches-programmes jugées par un exécuteur (non testé) |
| **Capacité** : 7 M ne boucle pas utilement | 10 §3a | le programme 1 (375 M, A100) |

## 3. L'ordre des tests : le plus d'information par heure de compute

**Sur CPU, maintenant, gratuit :** (mis à jour : les miniatures de W2 se font sur CPU, docs/36 et 37 ;
W2-A2, le comptage modulo 3, et W2-A7, la troncature du gradient, aussi ; W2-A3, S₅, demande
d'abord un mélangeur à produits de Householder, non implémenté)

1. ~~H26, le format objet, puis H27, la clé copiée~~ : faits (32 §23). Le proposeur ne
   démarre pas à 7 M ; le CPU s'arrête pour le programme 3. H21 à H25, et donc H23, le
   cœur du but, attendent l'A100.
2. ~~Le `done` prématuré~~ : fait (H28, 32 §24). C'était un défaut de décodage ; l'option
   corrigée entre dans la recette A100 (31, amendement 25).
3. ~~A4-0 à 7 M~~ : **écarté**. Son protocole (docs/research/A4 §8) exige un point de
   contrôle ≥ 350 M, où la récurrence sert ; à 7 M boucler n'achète rien (2,184 contre
   2,179), un AUROC n'y dirait rien de l'échelle visée. Il tourne en minutes sur le
   checkpoint du programme 1.
4. ~~SI-1, SI-1b, sa réplication, SI-1c~~ : faits (39 §3). La boucle dépasse son
   générateur quand au moins une proposition nouvelle est rattrapée et promue : c'est
   arrivé dans 3 runs sur 6. Quatre murs sont nommés (39 §3, SI-1c) : la rareté des
   nouvelles au départ, la récompense qui s'éteint, l'oubli du format, le plafond des
   règles.

**Sur A100, dès qu'une session est disponible** (docs/34 §9 pour le choix de la carte) :

1. `scripts/gpu_check.py` sur chaque carte proposée (≈ 5 min chacune).
2. **Le programme 1** (≈ 76 h) : tranche C1 à l'échelle, dont la composition (H4).
3. **La boucle fermée à 375 M** sur son checkpoint : tranche C2 à l'échelle.
4. **Le programme 3 à 375 M** : C3, avec H23.
5. La mémoire (R03 E2 ≈ 5 h ; W3-E0/E1 ≈ 1,5 h ; W4 porte 0 ≈ 1 h) : C4.

## 4. Ce que ce dépôt refuse d'écrire

Pas de « sans limite », « AGI » ni « Turing-complet » (31 §5, 33 §5, W2 §2.2). Pas de
comparaison de capacité générale avec un modèle frontière : 73× à 3 114× moins de compute
(00 §0). Pas de chiffre `[S]` comme base d'une dépense. Chaque plafond atteint est un
résultat, publié tel quel.
