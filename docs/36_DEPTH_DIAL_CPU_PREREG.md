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

## 3. Résultats — exécution du 2026-09-23 (après l'amendement 1) : **non conclusif**

Précision par nombre de sauts, 512 exemples par *h*, chaque bras mesuré à son propre *k*.
Hasard : 0,125. Environ 150k paramètres, 6 000 pas, 3,5 à 7,3 min par modèle.

| Bras | *h* = 1 (contrôle) | 2 | 3 | 4 | 5 | 6 | Perte finale |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gdn-fixed`, graine 0 | 0,145 | 0,225 | 0,238 | 0,373 | 0,242 | 0,459 | 2,074 |
| `gdn-tied`, graine 0 | **1,000** | 0,211 | 0,311 | 0,256 | 0,422 | 0,252 | **1,295** |
| `attn-fixed`, graine 0 | 0,102 | 0,232 | 0,240 | 0,375 | 0,240 | 0,469 | 2,084 |
| `attn-tied`, graine 0 | 0,109 | 0,232 | 0,244 | 0,377 | 0,240 | 0,471 | 2,080 |
| `gdn-fixed`, graine 1 | 0,219 | 0,229 | 0,260 | 0,330 | 0,236 | 0,410 | 2,020 |
| `gdn-tied`, graine 1 | 0,203 | 0,230 | 0,273 | 0,359 | 0,232 | 0,479 | 2,012 |
| `attn-fixed`, graine 1 | 0,133 | 0,244 | 0,289 | 0,375 | 0,225 | 0,518 | 2,079 |
| `attn-tied`, graine 1 | 0,123 | 0,244 | 0,289 | 0,377 | 0,225 | 0,518 | 2,075 |

**Verdict : le contrôle échoue sur 7 modèles sur 8.** À 1 saut, une seule consultation que
la coda attention fait seule, un seul modèle apprend la tâche (`gdn-tied`, graine 0 : 1,000,
avec une chute de perte entre les pas 2 000 et 4 000). Comme la lecture pré-écrite le
prévoyait, c'est la recette qui est en cause : **H29, H30 et H31 ne sont pas décidables**
sur ces chiffres, et aucune n'est déclarée passée ni échouée.

**Ce qui se lit quand même.**
1. Les sept modèles qui n'apprennent pas la consultation apprennent un raccourci : répondre
   à peu près le nœud de départ. Leur précision suit la probabilité qu'une permutation de 8
   ramène au départ en *h* pas : haute aux sauts pairs (0,37 à 4, 0,41–0,52 à 6), basse aux
   impairs. C'est la signature d'un modèle qui n'a pas trouvé la consultation, pas d'un
   modèle qui compose mal.
2. Le seul modèle qui consulte (`gdn-tied`, graine 0) ne compose pas : 0,21 à 2 sauts,
   0,31 à 3. Au balayage des mêmes poids, *k* = 1 à 6, 3 sauts restent entre 0,27 et 0,34.
   C'est ce que H4 prédit pour un cœur à état borné. Mais sur un seul modèle, et sans bras
   attention qui ait appris la base à comparer, ce n'est **pas** un résultat.
3. L'apprentissage de la consultation est une transition brusque, à la date aléatoire,
   comme pour l'apparition des têtes d'induction. À 6 000 pas, elle est rarement franchie à
   cette taille.

**Suite (amendement 2, à écrire avant toute relance).** Allonger la recette, et vérifier
d'abord sur un seul bras (`attn-tied`, graine 0) que le contrôle passe (≥ 0,9 à 1 saut)
avant de relancer les huit modèles. Si le contrôle ne passe toujours pas, la question reste
à H4 du programme 1, à 375M.

## Amendement 2 — 2026-09-23, après l'exécution non conclusive, avant toute relance : une recette qui apprend la base

**Ce qui change**, la recette seulement. Bras, tâche, mesures et critères H29 à H31 sont
inchangés.
- **20 000 pas** au lieu de 6 000.
- **Les 2 000 premiers pas sur 1 saut seul** (`--warm-hops1 2000`), puis le mélange
  {1, 2, 3}. À 1 saut, répondre le nœud de départ ne rapporte presque rien (1/8 des
  nœuds sont fixes). Seule la consultation paie, alors que les lots à 2 sauts du mélange
  récompensaient le raccourci que sept modèles sur huit ont appris.
- Plafond de 40 min par modèle.

**Vérification avant la relance.** Un seul modèle, `attn-tied` en graine 0. S'il dépasse
0,9 à 1 saut, les huit modèles sont relancés sous cette recette (≈ 2,5 h de CPU) et H29 à
H31 sont jugées dessus. Le modèle de vérification compte comme l'un des huit, puisque même
bras, même graine, même recette. Sinon, H29 à H31 restent non conclusives à cette échelle,
et la question reste à H4 du programme 1.

## Résultat de l'amendement 2 — 2026-09-23 : la vérification échoue, H29 à H31 restent non conclusives

`attn-tied`, graine 0, 20 000 pas dont 2 000 sur 1 saut seul : **0,104** à 1 saut
(0,234 / 0,240 / 0,367 / 0,244 / 0,469 de 2 à 6), une perte restée entre 2,33 et 2,06 du
pas 2 000 au pas 16 000. Pas même pendant l'échauffement sur 1 saut, le modèle n'a trouvé
la consultation. Comme l'amendement le prévoyait, les huit modèles ne sont pas relancés :
**H29, H30 et H31 restent non conclusives sur CPU**, et la composition par la profondeur
sera mesurée par H4 du programme 1, à 375M. Le même obstacle arrête le contrôle de docs/37 :
à cette taille, nos couches d'attention apprennent mal une consultation qu'un transformeur
minimal apprend en 700 pas (docs/37, diagnostic).

## Amendement 3 — 2026-09-23, avant toute relance : la recette de petite largeur (docs/38)

**Cause trouvée.** Les échecs du contrôle n'étaient pas une question de longueur. À *d* =
64, `init_std` 0,02, les embeddings liés et le GQA rendent l'apprentissage d'une
consultation aléatoire (docs/38). Corrigés, notre attention apprend un rappel à 8 paires en
725 à 823 pas sur 4 graines sur 4, comme un transformeur minimal.

**Relance** avec `--small-width-recipe` : `init_std` 0,06, embeddings déliés, 4 têtes KV.
C'est la seule différence avec la recette d'origine, qu'on reprend telle quelle : 6 000
pas, sans échauffement, huit modèles (deux cœurs × deux calendriers × graines 0 et 1). Le
contrôle et H29 à H31 sont **inchangés**. Si le contrôle échoue encore (< 0,9 à 1 saut),
H29 à H31 restent non conclusives, cette fois sans cause connue.

## Résultats de l'amendement 3 — 2026-09-23 : le contrôle passe pour le cœur GDN, échoue pour le cœur attention ; H29 à H31 restent non conclusives

Recette de petite largeur, 6 000 pas, 512 exemples par *h*.

| Bras | *h* = 1 (contrôle) | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| `gdn-fixed`, graine 0 | **1,000** | **0,994** | **0,926** | 0,273 | 0,256 | 0,236 |
| `gdn-tied`, graine 0 | **0,996** | 0,361 | 0,270 | 0,336 | 0,275 | 0,402 |
| `attn-fixed`, graine 0 | 0,109 | 0,219 | 0,236 | 0,357 | 0,240 | 0,451 |
| `attn-tied`, graine 0 | 0,107 | 0,230 | 0,244 | 0,365 | 0,229 | 0,459 |
| `gdn-fixed`, graine 1 (rejoué) | **0,994** | 0,227 | 0,385 | 0,320 | 0,314 | 0,361 |
| `gdn-tied`, graine 1 (rejoué) | **1,000** | 0,219 | 0,348 | 0,279 | 0,449 | 0,244 |
| `attn-fixed`, graine 1 | 0,127 | 0,236 | 0,289 | 0,359 | 0,219 | 0,482 |
| `attn-tied`, graine 1 | 0,117 | 0,236 | 0,287 | 0,369 | 0,227 | 0,514 |

**Incident.** Les deux bras GDN de la graine 1 ont d'abord été tronqués, à 880 et 180 pas,
par un diagnostic lancé en parallèle, contre la règle d'un seul processus torch à la fois.
Leurs chiffres tronqués sont écartés, et ils ont été rejoués seuls. Les six autres modèles
ont fait leurs 6 000 pas.

**Verdict.** Le contrôle passe pour les quatre modèles à cœur GDN et échoue pour les quatre
à cœur attention. H29 à H31 comparent justement les deux cœurs : elles restent **non
conclusives**. La cause de l'échec du cœur attention n'est pas l'enveloppe récurrente
(docs/38 §6). Elle tient à la pile de blocs d'attention elle-même à cette taille, et n'est
pas isolée.

**Ce qui se lit sur le cœur GDN.** Il apprend la consultation simple à coup sûr (4 sur 4),
mais ne compose que rarement. Un seul modèle sur quatre fait 2 et 3 sauts (`gdn-fixed`,
graine 0 : 0,994 et 0,926), et aucun ne porte au-delà des sauts entraînés. Sa composition,
quand elle a lieu, peut venir des trois blocs d'attention hors de la boucle, qui suffisent
à trois sauts. Au-delà, là où le cœur devrait contribuer, rien. C'est compatible avec H4 à
l'échelle, qui prédit qu'un cœur à état borné ne compose pas. Ce n'est pas une preuve.
