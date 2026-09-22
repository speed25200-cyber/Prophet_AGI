# 36 — Pré-enregistrement : la profondeur comme cadran, en miniature

**Statut : v0, 2026-09-23, avant tout lancement.** Résultats dans ce document, sous
« Résultats », jamais réécrits au-dessus.

## 0. La question

La thèse centrale du dépôt (docs/35, C1) : un cœur bouclé « réfléchit plus longtemps au
lieu d'être plus gros ». Deux choses doivent être vraies pour qu'elle serve le but :

1. le cœur **compose** : *k* passes font *k* étapes qu'une passe ne fait pas ;
2. *k* est **un cadran** : lié à la difficulté déclarée, il porte au-delà des difficultés
   vues à l'entraînement.

Rien ne l'a encore montré dans ce dépôt :
- à 7 M, boucler n'achète rien (2,184 contre 2,179, docs/10 §3a) ;
- à 150k paramètres, la profondeur latente ne compose pas une chaîne arithmétique
  (50,7 % au mieux, docs/10 §3a″) ;
- à 375M sur A100, *k* = 8 fait +16 % de BPB (docs/17).

Le programme 1 (docs/29, H4) posera la question à l'échelle, sur des tables à 1–4 sauts.
Ce document en fait la version miniature sur CPU, telle que W2-A5 et W2-A6 la proposent
(docs/research/W2 §9) : **consulter une table en *h* sauts**.

## 1. Tâche, modèle, bras

- **Tâche** (`scripts/hops_cpu.py`) : une table mélangée de 8 paires `a b` (une permutation
  de 8 nœuds pris parmi 16), puis `Q départ H =`. La réponse est le nœud atteint après *h*
  applications de la table. Hasard : 1/8.
- **Entraînement et mesure** : entraînement sur *h* ∈ {1, 2, 3}, mesure sur *h* = 1 à 6, donc
  4 à 6 jamais vus.
- **Modèle** : la disposition du programme 1 en miniature (≈ 150k paramètres, largeur 64).
  Un prélude attention, un cœur bouclé d'un bloc, une coda de deux blocs attention. Le cœur
  est **GDN** (état borné) ou **attention**.
- **Bras** : cœur ∈ {`gdn`, `attn`} × calendrier ∈ {`fixed` : *k* = 4 à l'entraînement et à
  la mesure ; `tied` : *k* = *h*, fixé par la difficulté déclarée}.
- **Recette** : 6 000 pas, lots de 32, AdamW 1e-3, chauffe 100, cosinus ; graines 0 et 1 ;
  512 exemples de mesure par *h*. Balayage des mêmes poids à *k* = 1..6 pour *h* = 2, 3, 4.

## 2. Hypothèses et critères, fixés avant le lancement

| | Énoncé | Critère (sur les deux graines) |
|---|---|---|
| **H29 composition** (H4 de docs/29 en miniature) | Le cœur attention bouclé compose ; le cœur à état borné non. | `attn-tied` ≥ 0,9 à *h* = 2 et 3, et `attn-tied` − `gdn-tied` ≥ 0,3 à *h* = 3 |
| **H30 cadran** (W2-A6 en miniature) | Lier *k* aux sauts porte au-delà des sauts entraînés. | moyenne sur *h* ∈ {4, 5, 6} : `attn-tied` − `attn-fixed` > 0,20 |
| **H31 courbe de profondeur** | Mêmes poids, plus de passes, plus de sauts réussis, pour l'attention seulement. | à *h* = 3 : acc(*k* = 3) − acc(*k* = 1) > 0,3 pour `attn-tied`, ≤ 0,1 pour `gdn-tied` |

**Contrôle** : à *h* = 1, tous les bras devraient réussir, puisque la coda attention fait un
saut seule. Sinon, la recette est en cause, pas la thèse.

**Lecture pré-écrite**
- Si H29 échoue pour l'attention, le cœur bouclé ne compose pas même des consultations à
  150k paramètres, comme la chaîne arithmétique de docs/10. H4 à 375M reste le test qui
  compte.
- Si H29 passe et H30 échoue, la profondeur compose dans ce qu'elle a vu mais n'est pas un
  cadran au-delà.
- Rien ici n'est une revendication de capacité. C'est un test de mécanisme, à 150k
  paramètres, sur deux graines.

## Amendement 1 — 2026-09-23, avant tout résultat : la normalisation QK coupée

La première exécution a été arrêtée au pas 800 de son premier modèle, sans aucune mesure.
La configuration gardait la normalisation QK par défaut, et `design_warnings()` le
signalait : à *head_dim* 16, elle plafonne le logit d'attention à 4. Une consultation
exacte parmi 28 positions en souffre, précisément ce que les bras attention doivent faire.
docs/10 §1 l'avait appris sur le banc de l'aiguille, et `depth_cpu.py` la coupe. Elle est
coupée ici aussi (test). Rien d'autre ne change : mêmes bras, même recette, mêmes critères.

## 3. Résultats

*(à venir)*
