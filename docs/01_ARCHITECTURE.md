# 01 — Architecture Prophet

> Spécification et **registre de décisions**. Chaque choix cite le track de recherche qui
> le fonde, et chaque conflit entre tracks est arbitré explicitement plutôt que moyenné.
>
> Statut : v0.2 — les configurations sont produites par `scripts/design_search.py`, pas
> posées à la main. Les décisions marquées **[ABLATION]** ne sont pas acquises : elles
> attendent leur expérience de validation.

---

## 1. La forme du modèle

```
             ┌──────────────────────── prélude (attention) ───────────────────────┐
tokens ─────▶│  bloc SWA  │  bloc attention complète (NoPE)  │ ...                │
             └───────────────────────────────┬────────────────────────────────────┘
                                             │  h₀ (injecté à chaque tour)
                              ┌──────────────▼───────────────┐
                              │   CŒUR PARTAGÉ (récurrent)   │  ◀── bouclé k fois
                              │   GDN + FFN, poids uniques   │      k réglé à l'exécution
                              └──────────────┬───────────────┘
                                             │
             ┌───────────────────────────────▼────────────────────────────────────┐
             │  coda (attention)  │  norme  │  tête LM  │  tête MTP  │  confiance  │
             └────────────────────────────────────────────────────────────────────┘
```

Profondeur effective = `prélude + cœur × k + coda`.
Profondeur payée en mémoire = `prélude + cœur + coda`.

---

## 2. Décision 1 — L'attention ne va pas dans le cœur bouclé

**Le problème.** Une couche d'attention bouclée *k* fois a besoin de *k* caches KV
distincts : à l'itération *i*, les requêtes diffèrent de celles de l'itération *j*.
Boucler l'attention multiplie donc la mémoire KV par *k*. R04 signale que
l'implémentation de référence de Huginn fait précisément cela et appelle ça
« le piège de déploiement ».

**Deux solutions existent.** R04 propose le *partage du KV entre étapes* : calculer K,V à
la première itération seulement, recalculer les requêtes ensuite. Nous retenons une
solution plus simple et strictement bornée : **le cœur bouclé ne contient que des
mélangeurs à état borné** (gated DeltaNet), et l'attention est confinée au prélude et au
coda, où elle s'exécute exactement une fois.

**Vérification** (`tests/test_modeling.py::test_attention_cache_does_not_grow_with_recurrence_depth`) :

| | k = 1 | k = 8 |
|---|---:|---:|
| Octets de cache d'attention | 10 240 | **10 240** |
| Octets d'état récurrent | 11 264 | 90 112 |
| État récurrent à contexte × 8 | — | **inchangé** |

L'attention ne coûte rien de plus en profondeur ; l'état récurrent croît avec *k* mais
reste une matrice de taille fixe, indépendante de la longueur du contexte.

> **[ABLATION A-KV]** Le partage du KV entre étapes de R04 permettrait de l'attention
> dans le cœur. À comparer une fois le socle validé — mais pas avant, car cela
> réintroduit un couplage entre profondeur et mémoire.

### 2bis. Révision — ce que D1 a coûté

*Ajouté après le track W1. Cette décision était présentée comme une élégance ; c'est un
arbitrage, et il faut l'écrire ainsi.*

Le chain-of-thought rend deux services distincts : de la **profondeur sérielle**, et un
**bloc-notes relisible**. Le cœur bouclé remplace le premier. Il ne remplace pas le second :
un état de taille fixe est un résumé, pas un tampon adressable.

W1 donne à cette distinction une forme forte : *les boucles à état borné ne peuvent pas
décider les problèmes P-complets sous réductions logspace, là où un CoT de longueur
polynomiale le peut.* Empiriquement, sur les mêmes tâches, zéro token de mémoire échoue
systématiquement et huit suffisent.

**Le point qui nous concerne directement : D1 — « pas d'attention dans le cœur bouclé »,
prise pour empêcher le cache KV de croître avec *k* — est exactement la décision qui a
supprimé le bloc-notes.** Le gain mémoire (cache d'attention identique à l'octet près entre
k=1 et k=8) et la perte de capacité sont **le même choix**, pas deux choix indépendants.

Nous conservons D1, pour une raison qui reste valable : la mémoire d'inférence est notre
contrainte dure. Mais le coût est maintenant nommé, et deux conséquences en découlent :

1. **Prophet ne doit pas prétendre remplacer le CoT.** L'objectif honnête est de le
   *comprimer* — atteindre l'essentiel de la qualité du CoT complet en émettant nettement
   moins de tokens — et non de s'en passer.
2. **Un tampon latent persistant est la réparation candidate.** W1 spécifie un petit nombre
   d'emplacements latents portés d'un token décodé au suivant, écrits et lus par une
   attention bornée, dont le coût est indépendant de la longueur du contexte. C'est le
   bloc-notes restitué sans le cache KV. Non implémenté ; c'est le premier candidat si une
   porte de décision libère du budget.

### 2ter. Révision — la boucle à *k* constant n'achète aucune classe de complexité

Seconde correction de W1, plus sévère. Boucler *k* fois avec *k* **constant** laisse la
profondeur bornée par une constante : aucun changement de classe de complexité. Le gain est
un facteur constant, pas un changement de nature.

Seule une profondeur **dépendant de l'entrée** sort le modèle de sa classe — c'est-à-dire
un mécanisme de **halte entraîné**. Notre schéma de configuration expose déjà
`recurrent.halting`, mais nous le traitions comme une option agréable. C'est en réalité la
seule voie vers ce que la boucle est censée acheter, et sa priorité doit être relevée en
conséquence.

---

## 3. Décision 2 — Taille du modèle : arbitrage R04 contre R07

**Le conflit.** R04 spécifie 9.48B paramètres au total. R07 calcule que l'état statique
d'entraînement d'un MoE de 10B occupe **97 Gio contre 77 Gio utilisables** sur un A100 —
et recommande ≤ 4B au total. Les deux ne peuvent pas être vrais.

**Arbitrage : R07 l'emporte**, pour deux raisons indépendantes.

1. *Contrainte dure contre préférence.* La mémoire d'un A100 n'est pas négociable ; la
   taille cible de R04 l'est.
2. *Le compute-optimal est déjà en dessous.* Avec 300 heures-A100, notre planificateur
   place l'optimum vers 1B de paramètres pour 20B tokens. À 1.3B actifs on ne peut
   acheter que ~12B tokens, soit 0.5× Chinchilla — nous serions dans un régime de
   sous-entraînement sévère avant même de parler de mémoire.

**Résultat de `scripts/design_search.py`** — configurations satisfaisant *simultanément*
la mémoire d'entraînement, le budget de tokens, la mémoire de l'appareil cible et
l'absence de mauvaise allocation :

| Configuration | Total | Actifs | Prof. eff. | Tokens | Tok/actif | Entraînement | Appareil |
|---|---:|---:|---:|---:|---:|---:|---:|
| **prophet-main** `d1536 p4c4×4o4 e128` | **3.79B** | **369M** | 24 | 24.6B | 31 | 38.1 GB | 2.59 GB (5090) |
| **prophet-mini** `d1280 p3c4×2o3` | **229M** | 211M | 14 | 58.3B | 173 | 3.4 GB | 0.67 GB (iPhone) |

Le rapport de sparsité de 10.3× est ce qui rend la chose intéressante : la capacité de
connaissance d'un modèle de 3.8B pour le coût par token d'un modèle de 369M.

---

## 4. Décision 3 — La pile de mélangeurs

Fondée sur R02, qui recense les architectures hybrides ayant réellement été livrées.

| Section | Motif | Justification |
|---|---|---|
| Prélude | `SWA(2048)` puis `attention complète (NoPE)` | La fenêtre glissante capte la structure locale ; une couche globale sans encodage positionnel donne l'extrapolation en longueur. |
| Cœur (bouclé) | `GDN` uniquement | État borné → boucler ne coûte pas de mémoire de contexte (§2). |
| Coda | `SWA(2048)` puis `attention complète (NoPE)` | Rappel exact avant la lecture finale. |

Paramètres d'attention (identiques sur toutes les couches d'attention, ce qui les rend
**compatibles avec un donneur** Llama-3.2/Qwen3 — voir §7) :

| Champ | Valeur | Raison |
|---|---|---|
| `head_dim` | 128 | Correspond aux donneurs ; bon pour les noyaux. |
| `n_kv_heads` | `n_heads / 8` (GQA) | Réduction du cache KV de 8×, point sûr éprouvé. |
| `qk_norm` | activé | Stabilise sans plafonnement de logits, **et** supprime les valeurs aberrantes d'activation qui ruinent la quantification des petits modèles (R08). |
| Puits d'attention | 1 par couche | Sans lui, l'attention fenêtrée s'effondre en streaming. |
| Positions | RoPE sur SWA, **NoPE** sur les couches globales | R02 : l'extension de contexte devient quasi gratuite. |

**Risque principal identifié par R02, à surveiller** : l'effondrement du rappel
multi-clés. Le meilleur mélangeur linéaire de 2026 atteint 89.8 % sur aiguille unique
mais **37.8 % sur aiguilles multiples**. L'hybridation corrige partiellement ; le
correctif en cas d'échec est **plus de couches globales, pas une fenêtre plus large**.

---

## 5. Décision 4 — La profondeur comme cadran d'exécution

Le pari central (R04). Un cœur à poids partagés appliqué *k* fois.

| | iPhone 17 Pro | Mac Studio | RTX 5090 |
|---|---:|---:|---:|
| `k` par défaut | 2 | 4 | 4–8 |
| Profondeur effective (main) | 16 | 24 | 24–40 |

**Mécanismes retenus :**

- **Injection de l'entrée à chaque tour.** Sans elle, une boucle profonde dérive loin du
  prompt. Effet secondaire vérifié par test : elle rétablit un chemin de gradient vers le
  prélude même sous troncature agressive — c'est une propriété voulue, pas une fuite.
- **Init aléatoire de l'état à l'entraînement, déterministe à l'inférence.** L'init
  aléatoire force la boucle à converger vers un attracteur indépendant du point de
  départ, ce qui est *précisément* ce qui rend *k* réglable après coup. Mais elle doit
  s'arrêter à l'inférence : mesuré, deux appels identiques différaient de 0.67 en logits
  et l'équivalence préremplissage/décodage était rompue.
- **Rétropropagation tronquée** aux 3 dernières itérations : entraîner une boucle
  profonde coûte la mémoire d'activation d'une pile courte.
- **Profondeur échantillonnée par pas** (log-uniforme) : le modèle reste utilisable à
  *toute* profondeur.

> **[ABLATION A1]** R04 est explicite : *Mixture-of-Recursions sous-performe le vanilla à
> 135M et ne gagne qu'à partir de 360M.* Nos ablations à 130M peuvent donc produire un
> **faux négatif**. Le test go/no-go doit être conduit à ≥ 350M.

---

## 6. Décision 5 — Ce que nous ne construisons pas en v1

| Composant | Décision | Fondement |
|---|---|---|
| Frontend octet / sans tokenizer | **Non en v1** | R01 : à l'échelle mini le frontend octet tourne à **0.44× le débit**. Vocabulaire BPE de 32 768 avec repli octet, chiffres isolés, tokens d'indentation. Le frontend octet reste un *retrofit* ultérieur, sous ablation. |
| Multimodalité | **Non en v1, ancrages oui** | R12 : la vision n'ajoute aucun point aux benchmarks qui décident de notre victoire, mais les ancrages coûtent ~0.5M paramètres et zéro FLOP. |
| Audio | **Non** | R12 : ASR en pipeline (modèle externe distillé), pas de modélisation audio native. |
| Mémoire persistante | **Ancrée, activable** | R03 : conception à deux étages retenue, mais le verrou est prouvé par une ablation dédiée avant intégration. |

**Ancrages multimodaux implémentés en v1** (R12, hooks H1–H10) : 256 identifiants de
vocabulaire réservés, embeddings typés par modalité, RoPE à dimensions multiples
(dégénéré en 1-D aujourd'hui), masque d'attention comme objet de première classe
supportant les segments bidirectionnels, chemin `inputs_embeds`, cache dimensionné à
l'exécution.

---

## 7. Décision D10 — tranchée : la voie mixte

**Quatre tracks indépendants sont arrivés à la même conclusion**, sans se coordonner :

| Track | Formulation |
|---|---|
| R01 | Le budget ne permet pas un frontend octet entraîné de zéro ; retrofit d'un checkpoint. |
| R02 | « 300 heures-A100 ≈ 5.6B tokens à 1.2B actifs — nous ne pouvons pas surpasser Qwen3 en pré-entraînement. La conversion de donneur est le chemin réaliste. » |
| R04 | « Nous sommes ~2 500× sous le compute de pré-entraînement de Qwen3-1.7B. La récurrence achète de la profondeur, pas de la connaissance. » |
| R05 | « 466× — le multiple de compute nécessaire pour égaler Qwen3-1.7B depuis une init aléatoire. » |
| R07 | ≤ 4B au total, atteint par *upcycling* d'un checkpoint dense plutôt que depuis une init aléatoire. |

**Décision retenue : les deux, sur deux modèles distincts.**

| Modèle | Origine | Rôle |
|---|---|---|
| **Prophet-mini** (229M) | **Poids aléatoires** | Preuve scientifique honnête de l'architecture. Cible iPhone. Ne doit rien au pré-entraînement de personne. |
| **Prophet-main** (~970M) | **Conversion d'un donneur Apache-2.0** | Modèle compétitif. Hérite de la connaissance ; nous n'achetons que l'architecture. |

Les deux partagent l'architecture, le tokenizer d'entrée près, les données et l'évaluation.
Seule l'initialisation diffère.

### Ce que la conversion fait concrètement

Implémentée dans `prophet/convert/`. Résultat mesuré pour Qwen3-1.7B :

| | Donneur | Prophet converti |
|---|---:|---:|
| Paramètres | 1.72B | **0.97B** |
| Couches | 28 | 12 paramétrées, **28 effectives** (k=5) |
| Couverture paramétrique | — | **89 %** |

- **Prélude et coda** prennent les premières et dernières couches du donneur par **copie directe**. La configuration Prophet est générée *depuis* le donneur (`head_dim`, `n_kv_heads`, largeur, largeur FFN) précisément pour que la copie soit directe et non une interpolation.
- **Le cœur partagé** est initialisé par la **moyenne** des couches médianes du donneur. Des couches consécutives d'un transformeur entraîné calculent des mises à jour similaires ; leur moyenne est un point de départ défendable pour un bloc appliqué en boucle. C'est une initialisation, pas une équivalence.
- **Les couches à delta gated** n'ont pas d'équivalent chez le donneur. Leurs projections q/k sont amorcées depuis l'attention (les deux projettent le flux résiduel dans un espace où un produit scalaire signifie « similarité »), et la projection de sortie place les poids du donneur dans la première moitié avec **des zéros dans la seconde** — la capacité élargie démarre inerte, donc la fonction initiale de la couche est aussi proche de l'attention du donneur qu'un mélangeur à état borné peut l'être.

### Garde-fous

- **Licence.** `assert_donor_is_usable` **refuse** un donneur dont les conditions suivent la dérivée. Llama-3.2 est rejeté par défaut : sa licence contraindrait le nom du modèle produit et interdirait une publication sous Apache-2.0.
- **Couverture minimale.** En dessous de 50 % de paramètres hérités, `scripts/convert_donor.py` refuse : à ce niveau, la « conversion » est en réalité du pré-entraînement à départ chaud et doit être budgétée comme tel.
- **Vérification.** Les chiffres d'architecture des donneurs ont été écrits alors que le Hub était injoignable. Tous sont marqués `verified=False` et la conversion refuse de s'exécuter tant que `scripts/verify_donors.py` n'a pas confronté chaque champ au `config.json` du Hub. Un `head_dim` erroné n'échoue pas bruyamment : il produit des incompatibilités de forme silencieusement laissées en initialisation fraîche.

### Conséquence acceptée

Prophet-mini utilise Prophet-Tok v1 (32k) ; Prophet-main hérite du vocabulaire du donneur
(~152k). **Ils ne partagent donc pas de vocabulaire**, et mini ne peut pas servir de
modèle brouillon pour le décodage spéculatif de main — une propriété que R01 valorisait.
Le remplacement est déjà dans l'architecture : les têtes de prédiction multi-tokens de
main assurent la spéculation sans modèle externe.

## 8. Traçabilité des décisions

| # | Décision | Track | Statut |
|---|---|---|---|
| D1 | Cœur bouclé récurrent uniquement, attention hors boucle | R02, R04 | **Acquis** (test) |
| D2 | ≤ 4B total / ~370M actifs | R07, planificateur | **Acquis** (mémoire) |
| D3 | Hybride GDN 3:1 avec SWA + globale NoPE | R02 | **Acquis** |
| D4 | Profondeur réglable à l'exécution | R04 | **[ABLATION A1] à ≥ 350M** |
| D4b | Halte entraînée, pour une profondeur dépendant de l'entrée | W1 | **Requis** — sans elle, la boucle n'achète qu'un facteur constant (§2ter) |
| D1b | Bloc-notes latent persistant, pour réparer ce que D1 a coûté | W1 | **Candidat** — non implémenté (§2bis) |
| D5 | Init d'état déterministe à l'inférence | interne | **Acquis** (test) |
| D6 | BPE 32k à repli octet, pas de frontend octet en v1 | R01 | **Acquis** |
| D7 | Ancrages multimodaux seulement | R12 | **Acquis** |
| D8 | Mémoire persistante à deux étages | R03 | **[ABLATION E2]** |
| D9 | Tête de confiance + abstention | R09 | **[ABLATION A1-R09]** |
| D10 | Zéro contre conversion de donneur | R01/R02/R04/R07 | **Tranché : voie mixte — §7** |
