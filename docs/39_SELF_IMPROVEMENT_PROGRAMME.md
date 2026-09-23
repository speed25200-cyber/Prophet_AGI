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

*(à venir)*
