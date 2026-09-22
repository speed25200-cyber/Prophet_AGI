# 37 — Pré-enregistrement : le rappel contre la profondeur, en miniature

**Statut : v0, 2026-09-23, avant tout lancement.** Résultats sous « Résultats », jamais
réécrits au-dessus.

## 0. La question

Si « réfléchir plus longtemps » remplaçait des paramètres, boucler un cœur à état borné
devrait aussi compenser un état trop petit quand il faut retenir beaucoup de choses à la
fois. W2 (docs/research/W2 §3.3, prédiction F4) prédit le contraire. Le rappel multi-clés
serait borné par la taille de l'état (`linear_head_dim`), avec un coude, et le nombre de
passes *k* ne le déplacerait pas. Si c'est vrai, la profondeur est un cadran pour composer
(docs/36), pas pour mémoriser, et la mémoire exacte doit venir de l'attention ou d'un état
plus grand. C'est une borne directe de la condition C1 de docs/35. W2 classe cette mesure
comme la plus informative par heure de calcul de sa liste (W2-A4).

## 1. Tâche, modèles, bras

- **Tâche** (`scripts/recall_cpu.py`) : rappel associatif à requêtes multiples (MQAR).
  *m* paires clé → valeur (64 clés, 16 valeurs possibles), un séparateur, puis 8 clés
  demandées parmi elles ; la valeur est prédite juste après chaque clé. Hasard : 1/16.
- **Bras « état »** : aucun bloc d'attention. Prélude, cœur bouclé et coda sont des blocs
  GDN. *d_k* ∈ {8, 16, 32} × *k* ∈ {1, 4}, soit 6 modèles.
- **Contrôle « disposition »** : la disposition du programme 1 (prélude et coda à attention,
  cœur GDN), *d_k* = 16, *k* ∈ {1, 4}, soit 2 modèles. L'attention retrouve n'importe
  quelle paire.
- **Recette** : entraînement sur *m* ∈ {4, 8, 16, 32} en alternance, mesure à chaque *m* ;
  4 000 pas, lots de 32, AdamW 1e-3 ; graine 0 ; normalisation QK coupée (docs/36
  amendement 1).

## 2. Hypothèses et critères, fixés avant le lancement

| | Énoncé | Critère |
|---|---|---|
| **H32 (i) la profondeur ne remplace pas l'état** (F4) | Boucler n'améliore pas le rappel d'un modèle sans attention. | pour chaque *d_k* : moyenne sur *m* de acc(*k* = 4) − acc(*k* = 1) ≤ +0,05 |
| **H32 (ii) la taille d'état compte** | Le rappel croît avec *d_k*. | moyenne sur *m* de acc(*d_k* = 32, *k* = 1) − acc(*d_k* = 8, *k* = 1) ≥ +0,10 |
| **Contrôle** | L'attention retrouve tout. | les bras « disposition » ≥ 0,9 à chaque *m* ; sinon la recette est trop faible et les bras « état » ne se lisent pas |

**Lecture pré-écrite.** Si (i) et (ii) passent, réfléchir plus longtemps ne remplace pas la
mémoire. Le pari « moins de paramètres » doit alors garder de l'attention (ou un état plus
grand) pour tout ce qui se retient, et réserver la profondeur à ce qui se compose. Si (i)
échoue, boucler un état borné achète du rappel, contre la prédiction de W2, et cela vaut
une réplication à plus grande échelle. Un test de mécanisme, à 10⁵ paramètres, sur une
graine.

## 3. Résultats

*(à venir)*
