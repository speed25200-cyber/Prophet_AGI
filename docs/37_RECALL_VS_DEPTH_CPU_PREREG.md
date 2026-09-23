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

## 3. Résultats — exécution du 2026-09-23 : le contrôle échoue, H32 n'est **pas** décidée

Précision du rappel (hasard 1/16), 256 séquences × 8 requêtes par *m*, 4 000 pas, graine 0.

| Bras | *m* = 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| état, *d_k* = 8, *k* = 1 | 0,837 | 0,682 | 0,445 | 0,237 |
| état, *d_k* = 8, *k* = 4 | 0,999 | 0,979 | 0,902 | 0,691 |
| état, *d_k* = 16, *k* = 1 | 1,000 | 0,995 | 0,977 | 0,908 |
| état, *d_k* = 16, *k* = 4 | 1,000 | 0,995 | 0,973 | 0,908 |
| état, *d_k* = 32, *k* = 1 | 1,000 | 0,996 | 0,961 | 0,839 |
| état, *d_k* = 32, *k* = 4 | 1,000 | 0,997 | 0,979 | 0,901 |
| **contrôle** disposition, *k* = 1 | 0,310 | 0,250 | 0,184 | 0,154 |
| **contrôle** disposition, *k* = 4 | 0,314 | 0,247 | 0,191 | 0,160 |

**Verdict, selon la règle écrite avant le lancement.** Le contrôle échoue : les bras avec
attention n'atteignent pas 0,9. Les bras « état » ne se lisent donc pas comme un verdict,
et **H32 n'est ni passée ni échouée**. À titre descriptif seulement : à *d_k* = 8, boucler
quatre fois rattrape l'essentiel d'un état trop petit (+0,34 en moyenne) ; à *d_k* = 16 et
32, rien (−0,00 et +0,02). C'est le contraire de F4 pour un petit état, et l'accord avec F4
pour un état suffisant. Il faut le confirmer avec un contrôle qui passe.

**Pourquoi le contrôle échoue : un diagnostic, pas un défaut.** Même tâche à 8 paires
fixes, 4 000 pas, même optimiseur, graine 0 (script de brouillon, chiffres ici) :

| Modèle | Précision | Perte aux pas 500 / 1 000 / 2 000 / 3 000 |
|---|---:|---|
| transformeur minimal écrit à la main (positions apprises, `nn.MultiheadAttention`), 82k paramètres | **1,000** | 2,006 / 0,027 / 0,000 / 0,000 |
| `ProphetModel`, attention seule (RoPE, 2 couches), 79k | 0,955 | 1,959 / 1,814 / 1,272 / 0,147 |
| `ProphetModel`, disposition des sondes (fenêtre, NoPE, *sink*, cœur GDN), 154k | 0,889 | 1,828 / 1,431 / 0,830 / 0,378 |
| `ProphetModel`, GDN seul, 156k | 0,978 | 0,896 / 0,293 / 0,127 / 0,111 |

Nos couches d'attention **apprennent** la consultation : elles ne sont pas cassées. Mais
elles la trouvent environ 3× plus tard qu'un transformeur minimal, et plus tard que la règle
delta. Sur la tâche mélangée des sondes (longueurs de 4 à 32 paires), 4 000 pas ne leur
suffisent pas. Même lecture pour la sonde des sauts (docs/36), dont aucun bras attention
n'a appris le saut simple en 6 000 pas. L'écart de vitesse avec un transformeur minimal
(positions, GQA, initialisation) est lui-même une question ouverte à petite échelle ; rien
ne dit qu'il existe à 375M.

## Amendement 1 — 2026-09-23, après l'exécution, avant toute relance : le contrôle d'abord

La recette passe à **12 000 pas** ; tout le reste est inchangé. **Le contrôle est relancé
seul d'abord** (les deux bras « disposition »). S'il passe (≥ 0,9 à chaque *m*), les six bras
« état » sont relancés sous la même recette, et H32 est jugée sur cette seconde exécution.
Sinon, H32 reste non décidée sur CPU, et les chiffres descriptifs ci-dessus sont tout ce que
cette miniature donne.
