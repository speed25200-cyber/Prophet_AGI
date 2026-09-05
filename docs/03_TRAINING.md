# 03 — Méthode d'entraînement

> Recette issue du track R07, contrainte par le matériel réel : **un A100 80GB Ampere
> (sm_80), sessions interruptibles**. Tout ce qui suppose du FP8 matériel, FlashAttention-3
> ou du multi-GPU est hors périmètre — non par choix, mais parce que le A100 n'a ni FP8
> ni support FA3, et que nous n'avons qu'une carte.

---

## 1. L'optimiseur : Muon sur les matrices, AdamW sur le reste

Muon orthogonalise le momentum avant de l'appliquer. L'intuition : la descente de
gradient sur une matrice pousse répétitivement dans quelques directions dominantes ;
orthogonaliser étale la mise à jour sur tout le spectre, donc chaque pas déplace la
matrice dans des directions qu'elle n'a pas déjà apprises.

**Mesuré dans nos tests** : cinq itérations du quintique réglé font passer le nombre de
conditionnement d'une matrice mal conditionnée de ~1000 à ~6. Elles n'orthogonalisent pas
parfaitement — ce n'est pas leur but.

| Groupe | Optimiseur | Weight decay | Raison |
|---|---|---|---|
| Matrices cachées 2-D | **Muon**, lr 0.02 | 0.1 découplé | Le cas pour lequel l'orthogonalisation est définie. |
| Embeddings, tête LM | AdamW, lr 3e-3 | 0.1 | Objets indexés, pas multipliés. |
| Normes, biais, portes | AdamW | **0** | Les décroître rétrécit la représentation au lieu de régulariser. |
| **Routeur MoE** | AdamW | 0.1 | **Délibérément hors Muon** : orthogonaliser ses mises à jour combattrait l'équilibrage de charge qui empêche l'effondrement des experts. |

**Attente réaliste.** R07 : le gain réel est de **1.1–1.4×** face à un AdamW *correctement
réglé*, pas 2×, et il décroît avec l'échelle. Le gain plus important est la mémoire :
**2 octets/paramètre d'état contre 4–8 pour AdamW**. Sur une seule carte de 80 Go, c'est
souvent ce qui décide si une configuration s'entraîne du tout.

C'est pourquoi le bake-off optimiseur est une **porte de décision** au seuil de rentabilité
explicite : en dessous de 1.058× d'accélération, la comparaison coûte plus qu'elle ne
rapporte et on garde AdamW.

---

## 2. Le planning : WSD, pas cosinus

Un planning cosinus enfouit la longueur totale du run dans chaque pas : il faut fixer le
nombre d'étapes d'avance, et s'arrêter tôt laisse le modèle à un taux élevé, en pleine
descente et sous-entraîné. Trois propriétés nous imposent WSD :

1. **La longueur du run n'a pas à être connue.** Le budget Colab réel se découvre en
   route ; le plateau s'étend ou se coupe librement.
2. **Les checkpoints de plateau sont réutilisables.** Plusieurs recuits peuvent être
   branchés depuis un même checkpoint et fusionnés — le meilleur rapport qualité/prix du
   plan, puisque la phase de décroissance est une petite part des tokens mais l'endroit
   où les scores se font.
3. **L'interruption est survivable.** Une session qui meurt pendant le plateau ne coûte
   que les pas écoulés.

Forme retenue : **2 % warmup / 80 % plateau / 18 % décroissance en `1 − √progress`**.
Cette forme passe plus de temps à taux utile avant la chute qu'une rampe linéaire.

```
lr │      ┌──────────────────────────────┐
   │     ╱                                ╲
   │    ╱                                   ╲
   │   ╱                                       ╲___
   └──┴────────────────────────────────────────────▸ steps
     warmup            plateau              décroissance
                          ▲
                    point de branchement : recuits multiples + fusion
```

---

## 3. Survivre à Colab

Le processus **sera tué sans préavis**, des dizaines de fois sur un run de plusieurs
semaines. La conception en découle.

**Rotation atomique à deux emplacements.** L'écriture va dans un fichier temporaire, est
fsyncée, hachée en SHA-256, puis renommée dans un emplacement (le renommage est atomique
sous POSIX). Les emplacements alternent, donc une écriture tronquée ne peut endommager
que la copie *la plus ancienne* — la plus récente est toujours intacte. Le chargement
vérifie le hachage et retombe sur l'autre emplacement en cas d'échec.

Testé : après corruption délibérée du checkpoint courant, le chargement récupère
silencieusement le précédent ; après corruption des deux, il **lève une erreur** au lieu
de rendre des poids douteux.

**Reprise exacte.** Sont sauvegardés *ensemble* : poids, états d'optimiseurs, position
dans le planning, curseur de données, tampon de tokens partiels, et l'état du générateur
aléatoire qui échantillonne la profondeur de récurrence. Le test
`test_resumed_run_matches_uninterrupted_run` vérifie qu'un run interrompu au pas 10 puis
repris produit des poids **identiques** à un run de 20 pas jamais interrompu.

La sélection de source dans le chargeur dérive d'un **hachage sans état de (graine, pas)**,
donc reprendre au pas 40 000 est en O(1) et ne peut pas dériver.

**Jalons préservés.** Le point de branchement du plateau est copié hors rotation : il sert
de départ à plusieurs recuits.

---

## 4. Stabilité

Un run qui diverge à 70 % du budget est la pire perte possible. Les mesures retenues, par
ordre de rapport bénéfice/coût :

| Mesure | Coût | Ce qu'elle empêche |
|---|---|---|
| **QK-norm** | négligeable | Explosion des logits d'attention. **Et** les valeurs aberrantes d'activation qui ruinent la quantification des petits modèles (R08) — c'est deux problèmes pour le prix d'un. |
| **z-loss** (1e-4) | négligeable | Dérive de magnitude des logits, précurseur habituel d'instabilité en basse précision. |
| **Scaling résiduel** 1/√(2·profondeur) | nul | Variance non bornée du flux résiduel. La profondeur qui compte est l'**effective** : une boucle profonde ajoute autant de variance que des couches distinctes. |
| **Clipping de gradient** (1.0) | négligeable | Pics isolés. |
| **Rétropropagation tronquée** (3 itérations) | réduit le coût | Gradients qui explosent ou s'évanouissent à travers *k* boucles. |
| **muP** | une passe de réglage | Un LR trouvé sur un proxy étroit transfère à la largeur finale sans balayage — à notre budget, c'est la différence entre un run de réglage et une douzaine. |

---

## 5. Précision

**Ne pas pré-entraîner en basse précision** (R08). Le A100 est sm_80 : pas de FP4/FP8
matériel, donc un QAT serait *simulé* — tout le coût, zéro accélération. De plus, la
précision d'entraînement compute-optimale se situe autour de 7–8 bits, au-dessus de ce
qu'un QAT agressif viserait.

À la place : **architecture consciente de la quantification dès le pas 0** (dimensions
constructibles en Hadamard, QK-norm, pas de biais, routeur en FP32), une sonde PTQ
continue pendant le pré-entraînement pour voir la dégradation arriver, puis un recuit QAT
sur les derniers ~12 % de tokens.

Entraînement : **autocast BF16, paramètres FP32** (ils *sont* la copie maîtresse), TF32
activé, activations recalculées en arrière (`activation_checkpointing`). Les états
d'optimiseur sont en FP32 : 4 octets/paramètre pour Muon, 8 pour AdamW — le calcul de
budget utilise la répartition réelle, pas un optimiseur 8 bits qui n'existe pas.

> **Ce qui n'est pas là.** Le noyau delta fusionné (`flash-linear-attention`) n'a jamais
> été exécuté dans ce dépôt — pas de GPU. Le scan de référence retient chaque état pour
> la rétropropagation, ce qui coûte ~144 Go pour 8k tokens sur la config principale.
> `scripts/train.py` **refuse** un run non-`--smoke` tant que `fla` n'est pas installé,
> et un test d'équivalence GPU contre `_scan` (sortie *et* état final) est requis avant
> qu'il ne porte un entraînement.

---

## 6. Ce qui reste à valider

| Élément | Statut |
|---|---|
| Muon > AdamW à notre échelle | **Porte de décision**, 14 h, seuil 1.058× |
| Profondeur récurrente > profondeur simple | **Porte de décision**, 24 h, à ≥ 350M paramètres |
| MFU réel des noyaux hybrides | **Porte de décision**, 2 h |
| Recuit multiple + fusion | Non testé ; peu risqué, gain attendu ~0.5–1 point |
