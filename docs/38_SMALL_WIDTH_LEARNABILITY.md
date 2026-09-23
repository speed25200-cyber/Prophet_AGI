# 38 — Pourquoi l'attention des miniatures n'apprenait pas : largeur, initialisation, embeddings liés

**Statut : diagnostic, 2026-09-23.** Aucune hypothèse pré-enregistrée n'est jugée ici. Ce
document explique pourquoi les contrôles des miniatures de docs/36 et docs/37 ont échoué,
et ce que cela implique pour toutes les miniatures CPU du dépôt. Scripts de diagnostic en
brouillon ; tous les chiffres sont ici.

## 0. Le constat

Dans les deux miniatures, les bras avec attention n'apprennent pas une consultation que la
règle delta apprend :
- docs/36 : aucun bras à cœur attention n'apprend le saut simple, en 6 000 puis 20 000 pas ;
- docs/37 : le contrôle avec attention reste sous 0,5, en 4 000 puis 12 000 pas.

## 1. Sur plusieurs graines, l'écart est réel

Même tâche (rappel multi-requêtes, 8 paires), même optimiseur (AdamW 1e-3, chauffe 100),
3 000 pas. On compte les graines dont la perte passe sous 0,5 :

| Modèle, *d* = 64 | Graines qui apprennent | Pas pour y arriver |
|---|---:|---|
| transformeur minimal écrit à la main (positions apprises, LayerNorm, biais, init PyTorch, embeddings déliés) | **4 sur 4** | 746 – 961 |
| `ProphetModel`, attention seule (RoPE, GQA 4/2, RMSNorm, sans biais, `init_std` 0,02, embeddings liés) | **1 sur 4** | 2 092 |

## 2. Ingrédient par ingrédient, depuis le transformeur minimal

On ajoute un seul ingrédient de notre pile à la fois, sur 4 graines :

| Variante | Graines qui apprennent | Pas pour y arriver |
|---|---:|---|
| transformeur minimal (attention réécrite à la main) | 4 | 792 – 1 640 |
| + RoPE | 4 | 686 – 1 027 |
| + RMSNorm | 4 | 712 – 1 443 |
| − biais | 4 | 633 – 968 |
| + norme finale | 4 | 880 – 2 036 |
| + embeddings liés | 4 | 1 536 – 1 961 (≈ 1,5× plus lent) |
| **+ `init_std` 0,02** | **0** | — |

## 3. Dans notre modèle

Balayage de `init_std` (GQA et embeddings liés conservés) :

| `init_std` | 0,02 | 0,04 | 0,06 | 0,09 | 0,125 |
|---|---:|---:|---:|---:|---:|
| graines qui apprennent (sur 4) | 1 | 0 | 2 | 1 | 1 |

L'initialisation seule ne suffit pas. À `init_std` 0,06, en retirant les autres ingrédients :

| Variante | Graines qui apprennent | Pas pour y arriver |
|---|---:|---|
| base (GQA 4/2, embeddings liés) | 2 | 1 043 – 1 603 |
| sans GQA (4/4) | 3 | 573 – 1 630 |
| embeddings déliés | **4** | 786 – 1 287 |
| **sans GQA + embeddings déliés** | **4** | **725 – 823**, comme le transformeur minimal |

## 4. Ce que cela établit

1. **À *d* = 64, trois ingrédients de notre pile rendent l'apprentissage d'une consultation
   aléatoire** : `init_std` 0,02 (0,16/√*d*, six fois sous l'usage), les embeddings liés
   (sortie et entrée partagées quand le vocabulaire est un jeu de symboles), et, dans une
   moindre mesure, le GQA. Corrigés ensemble, notre modèle apprend comme un transformeur
   minimal. Rien n'est cassé : c'est une question de largeur.
2. **Le chemin critique n'est pas concerné.** À *d* = 1 792 (programme 1), 0,02 vaut
   0,85/√*d*, l'échelle usuelle, et les runs R04 sur A100 ont appris normalement. Que les
   embeddings liés retardent les têtes de consultation à 375M n'est pas mesuré.
3. **Les miniatures CPU à *d* = 64 du dépôt en portent la trace.** Leurs bras avec
   attention ont été entraînés dans le régime où la consultation s'apprend mal :
   - la composition latente de docs/10 §3a′–a″ (dont le bras « attention dans le cœur »,
     45,5 %) ;
   - le banc de l'aiguille de docs/10 §1 ;
   - docs/36 ;
   - le contrôle de docs/37.

   Leurs conclusions sur **l'attention** à cette largeur sont à reprendre avec la recette
   corrigée. Celles sur la règle delta (GDN), qui apprend vite dans les deux régimes,
   tiennent.
4. **Un champ jamais lu.** `FeedForwardConfig.activation` (`swiglu`, `geglu`, `relu2`) n'était
   lu par rien : toute FFN est un SwiGLU. Un diagnostic de ce document l'a montré (GeGLU
   donnait des chiffres identiques au bit près). `validate()` refuse désormais toute autre
   valeur, test à l'appui. Aucun config du dépôt ne demandait autre chose que `swiglu`.

## 5. Ce qui change dans les outils

- `ProphetConfig.design_warnings()` signale une attention à *d* ≤ 128 avec `init_std` <
  0,5/√*d* et des embeddings liés (test). Une miniature ne pourra plus confondre ce régime
  avec un résultat.
- La recette de miniature qui apprend : `init_std` ≈ 0,5/√*d* (0,06 à *d* = 64),
  embeddings déliés, pas de GQA. docs/36 est relancé avec elle (amendement 3).
