# 07 — Les murs

> Analyse des verrous que la génération actuelle de modèles ne franchit pas. Ce document
> se distingue de [`00_PROBLEM_LANDSCAPE.md`](00_PROBLEM_LANDSCAPE.md) par son objet :
> le premier recense les problèmes attaquables sous notre budget, celui-ci cherche le
> **mécanisme** de trois ou quatre limitations plus profondes, y compris celles que nous
> ne savons pas franchir.
>
> Règle de lecture : on distingue partout la **phénoménologie** (« les modèles oublient »)
> du **mécanisme** (« pourquoi la descente de gradient détruit ce qu'elle ne met pas à
> jour »). Une phénoménologie ne se corrige pas ; un mécanisme, parfois.

---

## Mur A — Le chain-of-thought est un canal étroit, et c'est aussi ce qui le stabilise

### A.1 L'arithmétique — corrigée par le track W1

Le raisonnement verbalisé est traité comme un coût de latence. C'est d'abord un
**goulot d'information**. Mais la première version de ce document énonçait ce goulot de
façon **fausse**, et la correction est instructive.

**Ce que j'avais écrit :** un flux résiduel de 2 048 dimensions en bf16 porte 32 768 bits ;
un token d'un vocabulaire de 32 768 en porte 15 ; donc « 99.95 % de l'état calculé est jeté
à chaque étape », soit un rapport de 2 180 : 1.

**Ce qui ne va pas, et pourquoi c'est important.** Rien n'est jeté. Par l'inégalité de
traitement de données, toute activation est une fonction déterministe du préfixe : le cache
KV conserve l'intégralité de ce qui a été calculé, et n'apporte aucune information nouvelle.
Parler de perte d'information était une erreur de cadrage.

**Le vrai mécanisme**, tel que W1 le formule : le token émis est **le seul chemin qui
rentre à nouveau à la couche 0**. Toute autre dépendance entre positions plafonne à `L`
sauts, où `L` est la profondeur. C'est précisément pourquoi les *filler tokens* ne
produisent pas de raisonnement sériel : ils ajoutent des positions sans ajouter de
ré-entrée. Le goulot n'est pas une perte d'information, c'est un **goulot de re-circulation**.

Les deux corrections quantitatives tiennent malgré tout, et vont dans le même sens :

| Grandeur | Valeur nominale | Valeur réalisée |
|---|---:|---:|
| Entropie d'un token en CoT | 15 bits | **1.0 – 2.0 bits** |
| Rang effectif du flux résiduel | 2 048 | une petite fraction de `d` |
| **Rapport** | ≈ 2 180 : 1 | **≈ 250 – 750 : 1** |

Un raisonnement de 1 000 tokens transporte donc de l'ordre de **150 octets** de contenu
décisionnel réel. `prophet/analysis/bandwidth.py` implémentait déjà les deux corrections ;
ce qu'il mesure reste valable, c'est l'interprétation qui a changé.

### A.2 La discrétisation stabilise — mais elle n'est pas nécessaire

L'hypothèse : projeter sur le vocabulaire agirait comme une correction d'erreur, ramenant
l'état sur la variété de ce que le modèle a déjà vu, et ce serait pourquoi le raisonnement
latent dérive là où le CoT en tokens tient sur des milliers d'étapes.

**Verdict de W1 : le phénomène est confirmé, le mécanisme est probablement faux.**

L'ablation directe existe et va dans notre sens — retirer les ancres symboliques fait
revenir l'hallucination. Mais plusieurs travaux stabilisent le raisonnement latent par une
**supervision par étape**, sans aucune projection au moment de l'inférence. La discrétisation
est donc **suffisante mais pas nécessaire** : ce qui stabilise, c'est qu'un signal
d'entraînement contraigne chaque étape, pas que l'état soit discret.

Conséquence pratique, et elle est meilleure que l'hypothèse d'origine : le substitut bon
marché de l'ancrage discret est une **perte de décodabilité** — exiger que l'état latent
reste décodable en tokens, sans jamais le décoder réellement à l'inférence. On garde la
contrainte, on paie le goulot seulement à l'entraînement.

### A.3 Profondeur ≠ bloc-notes — et c'est désormais un théorème

Le CoT rend **deux** services, et la littérature les confond :

| Service | Ce que c'est | Remplaçable par la récurrence ? |
|---|---|---|
| **Profondeur sérielle** | Chaque token ajoute un pas de calcul séquentiel | **Oui** — c'est ce qu'achète un cœur bouclé, sans passer par le goulot |
| **Bloc-notes** | Les tokens émis sont **relisibles** : mémoire externe adressable | **Non** — un état de taille fixe est un résumé, pas un tampon |

W1 confirme la distinction et lui donne une forme forte : **les boucles à état borné ne
peuvent pas décider les problèmes P-complets sous réductions logspace, là où un CoT de
longueur polynomiale le peut.** Empiriquement, sur les mêmes tâches, zéro token de mémoire
échoue systématiquement et huit suffisent à réussir.

**Et voici le coût que nous n'avions pas vu.** Notre décision D1 —
« pas d'attention dans le cœur bouclé », prise pour empêcher le cache KV de croître avec
*k* — **est exactement la décision qui a supprimé le bloc-notes**. Le gain mémoire et la
perte de capacité sont le même choix, pas deux choix indépendants. Nous l'avions présenté
comme une élégance ; c'est un arbitrage, et il faut l'écrire ainsi.

**Une seconde correction, plus sévère encore.** Boucler *k* fois avec *k* **constant** ne
change aucune classe de complexité : la profondeur reste bornée par une constante. La
boucle n'obtient donc **aucun crédit asymptotique**. Seule une profondeur dépendant de
l'entrée — c'est-à-dire un mécanisme de **halte** — sort le modèle de sa classe.

**Conséquence : la halte est passée d'option à exigence, et elle est implémentée.**
Une tête scalaire par position et par itération produit une distribution de temps d'arrêt ;
la perte de ponderation combine la perte de modélisation espérée sur ces temps d'arrêt et
une divergence vers un prior géométrique. Le prior n'est pas décoratif : sans lui, la tête
apprend à toujours réfléchir aussi longtemps qu'on l'y autorise, puisque calculer davantage
ne dégrade jamais la perte.

Deux détails que l'implémentation a fait apparaître :

- Sans sa propre perte, la tête de halte **ne reçoit aucun gradient** — la distribution
  d'arrêt n'entre pas dans les logits. Une tête non entraînée donne une profondeur qui
  *paraît* dépendre de l'entrée et n'est que du bruit. Le `Trainer` reprend donc
  automatiquement le poids depuis la configuration du modèle.
- Évaluer le coda à chaque itération pour scorer les points d'arrêt candidats écrivait
  *k* fois dans le **même** emplacement de cache. Le décodage incrémental produisait alors
  une sortie fluide, plausible et fausse, sans rien pour le signaler. Les passes de sonde
  sont désormais sans cache ; le vrai coda ne s'exécute qu'une fois. Vérifié par test :
  10 positions de cache pour 10 tokens, pas 40.

## Mur B — La profondeur fixe borne la classe de calcul

Un transformeur à `L` couches applique un nombre **constant** d'étapes séquentielles,
quelle que soit la difficulté de l'entrée. Les résultats de complexité placent les
transformeurs à profondeur fixe et précision logarithmique dans une classe de circuits à
profondeur constante : il existe des problèmes intrinsèquement séquentiels qu'aucune
largeur ne compense.

Le CoT est l'échappatoire de l'industrie : *t* étapes de CoT ≈ *t* pas séquentiels
supplémentaires. Mais on vient de voir à quel prix — 2 000 : 1 d'information par pas.

**La récurrence achète la même profondeur sérielle sans le goulot de re-circulation.**
C'est la thèse centrale de Prophet, et elle a une conséquence testable : à profondeur
sérielle égale, un cœur bouclé *k* fois devrait égaler *k* tokens de CoT sur les tâches de
pure composition, et les battre sur celles qui exigent de transporter beaucoup d'état entre
les pas.

**Réserve, apportée par W1 (§A.3) :** avec un *k* constant, ce gain est un facteur constant,
pas un changement de classe. Pour que la boucle achète réellement de la profondeur au sens
de la complexité, il faut que *k* dépende de l'entrée — donc un mécanisme de halte
entraîné, pas un cadran fixé par l'appelant.

**Ce que nous avons peut-être surestimé.** Notre pile est majoritairement à état borné
(delta gated). Ces mélangeurs ont leurs propres limites d'expressivité, différentes de
celles de l'attention. Le pari de la conception est qu'une minorité de couches d'attention
complète suffit à les compenser. C'est un pari, pas un théorème — le track W2 est chargé
de le contredire s'il est faux, et le rapport R02 signalait déjà l'effondrement du rappel
multi-clés comme le point de rupture le plus probable.

---

## Mur C — L'apprentissage continu, et pourquoi le gradient ne peut pas le résoudre seul

### C.1 Le mécanisme

« Les modèles oublient » est une phénoménologie. Le mécanisme est le suivant : dans un
réseau dense, **chaque poids participe à tout**. Une mise à jour par gradient pour
apprendre X déplace des poids qui encodent Y. Pour apprendre X sans détruire Y, il
faudrait que la mise à jour soit orthogonale au sous-espace qui encode Y — et dans un
réseau dense, ce sous-espace est l'espace entier.

D'où le résultat que R03 a trouvé et sur lequel repose notre conception :

| Méthode | Connaissance antérieure perdue |
|---|---:|
| Fine-tuning complet | 89 % |
| LoRA | 71 % |
| **Mise à jour mémoire creuse** | **11 %** |

La sparsité n'est pas une optimisation, c'est **la condition d'orthogonalité rendue
approximativement vraie**. Écrire dans quelques emplacements est la seule variante qui ne
détruit pas ce qui était là, parce que c'est la seule où la mise à jour touche un
sous-espace petit.

### C.2 Le troisième étage que nous n'avons pas construit

Nous avons deux étages : état de session, et registre à écriture directe. La théorie des
systèmes d'apprentissage complémentaires en décrit **trois** : un magasin épisodique rapide
et creux, un système sémantique lent et distribué, et un **transfert du premier vers le
second** pendant le sommeil.

Nous avons construit le magasin rapide. Nous n'avons pas construit la distillation vers les
poids.

**W3 valide la lacune mais corrige la destination, et la correction change la conception.**
Deux points :

1. *Le registre ne croît pas sans borne.* Son nombre d'emplacements est fixe. Mon
   argument initial était faux sur ce point.
2. *Les faits ne doivent pas migrer vers les poids.* À ~2 bits par paramètre, un tronc de
   229M contient environ 57 Mo de connaissance extractible — **déjà entièrement dépensés**.
   Le registre, lui, offre 50 Mo en int4 pour **zéro FLOP marginal**. Distiller des faits
   du registre vers les poids consiste donc à déplacer de l'information d'un stockage
   gratuit vers un stockage saturé. C'est le mauvais sens.

Ce qui doit être distillé n'est pas le **contenu** mais la **règle** : ce qui rend le
registre atteignable et généralisable, pas ce qu'il contient. W3 propose une échelle à
trois barreaux dont seul le dernier touche des poids, et encore : ~3.2M paramètres, rien à
l'intérieur du cœur récurrent, tiers inférieur gelé.

**Avertissement que W3 remonte et qu'il faut prendre au sérieux :** la seule étude 2026 de
consolidation mémoire répétée observe une utilité qui **monte puis redescend sous le
niveau sans mémoire**, un contrôle purement épisodique restant compétitif. La consolidation
répétée n'est pas gratuite.

**La mesure qui tranche.** W3 la formule proprement : le **ratio de compétence
σ = gain de transfert / gain de rappel**, les deux en bits par octet (l'exactitude est au
hasard sous 500M paramètres), mesuré sur des instances tenues à l'écart d'une *famille*
consolidée, contexte effacé. σ ≈ 0 : une table de correspondance. σ ≈ 1 : la règle a été
apprise aussi bien que les instances.

---

## Mur D — Le calcul d'inférence ne se cumule pas

C'est le mur que personne ne nomme, et celui que notre architecture est le mieux placée
pour attaquer.

> Un modèle qui passe dix minutes de calcul à résoudre un problème difficile aujourd'hui
> n'en sait **rien de plus** demain. Chaque requête repart des mêmes poids gelés. Un humain
> devient meilleur sur une classe de problèmes en travaillant sur des instances ; un modèle
> non. La réponse de l'industrie aux problèmes difficiles est « dépenser plus de calcul à
> l'inférence » — et ce calcul est **jeté** à la fin de la réponse.

Prophet possède les deux pièces manquantes : la profondeur de récurrence est un cadran
réglable à l'exécution (donc on peut délibérément dépenser beaucoup sur un problème), et le
registre accepte une écriture en forme close sans rétropropagation (donc on peut garder le
résultat).

`consolidate_depth()` est le lien :

```
h_profond  = f_{k=16}(x)          # la passe chère
h_rapide   = f_{k=2}(x)           # la passe bon marché
cible      = λ (h_profond − h_rapide)
registre.write(h_rapide, cible)   # adressé par l'état de la passe rapide
```

Une passe bon marché ultérieure retrouve ce que la passe chère avait calculé.
Structurellement, c'est la même opération que la consolidation de contexte — même écriture
en forme close, autre axe.

**Deux choses décident si cela vaut quelque chose, et aucune n'est tranchée ici :**

1. **La vérification.** Consolider une réponse fausse est pire que ne rien consolider,
   parce que le modèle cesse de recalculer. `require_verified` refuse les épisodes non
   vérifiés par défaut ; fournir le vérificateur est le travail de l'appelant, et c'est la
   partie chère.
2. **La généralisation.** Mémoriser la réponse à un problème n'aide que sur ce problème.
   Le seul chiffre qui compte est le transfert vers des **instances voisines non
   consolidées**. `depth_transfer_error` le mesure explicitement sur un lot tenu à l'écart.
   Un registre qui ne fait que retrouver est un cache, pas une compétence — c'est le même
   critère qu'en §C.2, et ce n'est pas un hasard.

---

## Mur E — Trois échelles de temps, et celle qui manque

En rassemblant les quatre murs, la mémoire d'un modèle se décompose en échelles de temps,
et notre architecture en couvre trois sur quatre :

| Échelle | Mécanisme | Coût | État chez nous |
|---|---|---|---|
| **Dans un token** | Cœur récurrent bouclé *k* fois | Bande passante, pas de mémoire | **Implémenté** |
| **Dans le contexte** | CoT + cache KV | O(N) mémoire, goulot 2000:1 | Hérité du transformeur |
| **Entre sessions** | Registre + consolidation | Écriture creuse | **Implémenté** |
| **Dans le temps, sans émettre** | *manquant* | — | **Non implémenté** |

La case manquante est celle des **étapes de pause latentes** : consommer du calcul dans le
temps sans émettre de token, l'état récurrent portant le résultat d'une étape à la suivante.
Avec des mélangeurs à état borné, c'est presque gratuit — l'état ne grandit pas. C'est
l'élargissement du goulot du §A.2 appliqué à l'axe du temps plutôt qu'à celui de la
profondeur.

C'est le mécanisme que le track W1 doit spécifier ou rejeter.

---

## Ce que ce document ne résout pas

| Mur | Statut |
|---|---|
| Le goulot du CoT | **Mesurable** avec `prophet.analysis.bandwidth`. Le mécanisme d'élargissement reste à spécifier. |
| La profondeur fixe | **Attaqué** par la récurrence. Le pari sur les mélangeurs à état borné n'est pas prouvé. |
| L'oubli catastrophique | **Atténué** par la sparsité (11 % contre 89 %). Non résolu : le troisième étage manque. |
| Le calcul non cumulatif | **Mécanisme implémenté**, généralisation non démontrée. |
| La couverture factuelle | Borne physique. Contournée, jamais franchie. |
| La généralisation hors distribution | Aucune méthode connue. Hors périmètre, et nous le disons. |

Les quatre tracks W1–W4 sont chargés de contredire ce document là où il a tort. Leurs
rapports arriveront dans `docs/research/W*.md`.
