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
boucle n'obtient donc **aucun crédit asymptotique**. Une halte sous un plafond constant
adapte le coût moyen à l'entrée, mais ne change toujours pas la classe. Il faudrait que le
plafond lui-même croisse avec la taille de l'entrée, et entraîner le modèle dans ce régime.

**Conséquence pratique : la halte est requise pour l'efficacité adaptative, pas comme
preuve d'expressivité asymptotique.**
Une tête scalaire par position et par itération produit une distribution de temps d'arrêt ;
la perte de ponderation combine la perte de modélisation espérée sur ces temps d'arrêt et
une divergence vers un prior géométrique. Le prior n'est pas décoratif : sans lui, la tête
apprend à toujours réfléchir aussi longtemps qu'on l'y autorise, puisque calculer davantage
ne dégrade jamais la perte.

Trois détails que l'implémentation a fait apparaître :

- Sans sa propre perte, la tête de halte **ne reçoit aucun gradient** — la distribution
  d'arrêt n'entre pas dans les logits. Une tête non entraînée donne une profondeur qui
  *paraît* dépendre de l'entrée et n'est que du bruit. Le `Trainer` reprend donc
  automatiquement le poids depuis la configuration du modèle.
- Évaluer le coda à chaque itération pour scorer les points d'arrêt candidats écrivait
  *k* fois dans le **même** emplacement de cache. Le décodage incrémental produisait alors
  une sortie fluide, plausible et fausse, sans rien pour le signaler. Les passes de sonde
  sont désormais sans écriture et le vrai coda ne s'exécute qu'une fois. Mais une sonde
  sans cache ne voit pas non plus l'historique exact du coda ; le seuil adaptatif est donc
  refusé en décodage incrémental tant qu'un read-only cache ou un read-out équivalent
  n'existe pas.
- Une profondeur variable crée aussi des trous dans les caches récurrents : si le token
  *t* s'arrête à 1, l'état de profondeur 4 ne voit pas *t* lorsque *t+1* remonte à 4. La
  combinaison `halt_threshold + cache` échoue maintenant explicitement. Il faudra un
  backfill/recalcul démontré ou une profondeur monotone avant de la réactiver.

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

**Réserve, apportée par W1 (§A.3) :** avec un plafond *k_max* constant, ce gain est un
facteur constant, pas un changement de classe. Une halte dépendant de l'entrée réduit le
coût moyen mais ne suffit pas : pour acheter réellement de la profondeur au sens de la
complexité, il faut que *k_max* puisse croître avec la taille de l'entrée et entraîner le
modèle dans ce régime.

**Ce que nous avions surestimé — et un bug d'un caractère.** Notre pile est
majoritairement à état borné. W2 a trouvé que notre implémentation était *strictement plus
faible que la famille qu'elle prétend implémenter*, pour une raison d'une ligne.

La transition d'état d'une couche à règle delta est `α(I − β k kᵀ)`. Avec
**β ∈ (0,1)**, toutes ses valeurs propres restent strictement positives : aucun produit de
telles transitions ne peut changer de signe, et la parité est exactement un problème de
changement de signe. Avec **β ∈ (0,2)**, la transition peut réfléchir, et la parité
redevient atteignable.

Nous avions écrit `beta = sigmoid(...)`. Mesuré sur notre propre implémentation, parité
apprise à longueur 32 et évaluée à 128 :

| Plage de β | Couches | Exactitude @32 | Exactitude @128 |
|---|---:|---:|---:|
| (0, 1) | 1 | 0.531 | **0.508** |
| (0, 1) | 1 | 0.530 | **0.510** |
| (0, 1) | 2 | 0.521 | **0.504** |
| (0, 1) | 2 | 0.615 | **0.532** |
| **(0, 2)** | **1** | **1.000** | **0.996** |

Le hasard contre la résolution parfaite avec généralisation en longueur, pour une
multiplication. `linear_beta_max` est désormais un champ de configuration valant 2.0 par
défaut, et une vérification d'invariant refuse silencieusement 1.0.

Ce qu'il faut en retenir dépasse la parité : **une limite d'expressivité ne se voit pas
dans une courbe de perte.** Le modèle à β ∈ (0,1) s'entraînait normalement. Rien n'aurait
signalé qu'il lui manquait une classe entière de fonctions.

**Ce qui reste un pari.** Le rapport R02 signalait l'effondrement du rappel multi-clés
comme point de rupture le plus probable ; W2 ajoute que **le cadran de profondeur et le
budget de rappel sont le même cadran tirant en sens inverse** — le rapport effectif
linéaire/attention passe de 1:1 à *k*=1 à 8:1 à *k*=8. Augmenter *k* aggrave précisément la
faiblesse que les couches globales étaient censées compenser.

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
| **Sparse Memory Finetuning (SMF)** | **11 %** |

Dans cette expérience, la sparsité rend approximativement vraie la condition
d'orthogonalité : seules les rangées d'une couche mémoire préentraînée qui distinguent les
nouvelles données d'un corpus de fond sont ajustées. **Ce résultat ne valide pas encore le
registre Prophet** : ses clés sont aléatoires et gelées, et ses valeurs suivent une
écriture locale en forme close. Le 11 % est une motivation pour l'ablation, pas une mesure
du système présent.

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

> Un modèle qui passe dix minutes de calcul à résoudre un problème difficile aujourd'hui
> n'en sait **rien de plus** demain. Chaque requête repart des mêmes poids gelés. Un humain
> devient meilleur sur une classe de problèmes en travaillant sur des instances ; un modèle
> non. La réponse de l'industrie aux problèmes difficiles est « dépenser plus de calcul à
> l'inférence » — et ce calcul est **jeté** à la fin de la réponse.

### D.1 Correction : ce mur *est* nommé

J'avais présenté ceci comme « le mur que personne ne nomme ». C'est faux, et W4 le
documente : *sleep-time compute* énonce l'observation mot pour mot, et la littérature
adjacente la couvre sous d'autres noms — inférence amortie, distillation de contexte,
distillation Système-2 vers Système-1, itération d'expert. Sur environ 27 systèmes qui
stockent du raisonnement, **six seulement rendent l'inférence ultérieure réellement moins
chère à qualité égale**.

Ce qui reste inédit est étroit et vaut la peine d'être énoncé avec précision : une
**écriture sans rétropropagation** d'un delta de calcul dans un registre adressable. C'est
une contribution de mécanisme, pas d'observation.

### D.2 Correction plus sévère : notre mécanisme vise le mauvais axe

`consolidate_depth()` distille l'écart entre une passe profonde et une passe rapide. W4
remonte un chiffre de notre propre track R04 qui mine cette conception : **la profondeur
latente rapporte environ 1.8 points sur GSM8K, là où le chain-of-thought en rapporte
environ 33.** Si l'écart `h₁₆ − h₂` est petit, il n'y a presque rien à consolider.

La variante à construire d'abord est donc celle sur l'**axe du contexte** — consolider ce
qu'un long raisonnement verbalisé a apporté, en utilisant `consolidate()`, qui est déjà
implémenté et testé. C'est le même mécanisme sur l'entrée privilégiée qui porte
réellement le signal.

**Porte 0, avant toute dépense :** un balayage exactitude-contre-*k*. Si l'exactitude ne
monte pas avec la profondeur, la consolidation de profondeur n'a rien à stocker et le track
s'arrête là. Quelques minutes de calcul.

### D.3 L'adressage mémorise par construction

W4 a sondé l'adressage de notre registre : l'indice de Jaccard des emplacements atteints
par des instances de **même classe** contre des instances de **classe différente** vaut
0.530 contre 0.493 — c'est-à-dire **le hasard**. Le registre, tel qu'adressé aujourd'hui,
ne peut pas généraliser : il retrouve l'instance consolidée et rien d'autre.

C'est exactement l'échec que `depth_transfer_error` a été écrit pour détecter, et il le
détecte. La conception de l'adressage — W4 propose une clé à deux niveaux entraînée par
contraste — est le travail qui reste.

### D.4 L'économie, mesurée

| Grandeur | Valeur |
|---|---|
| Surcoût d'une passe *k*=16 contre *k*=2 | **4.67×**, pas 8× |
| Seuil de rentabilité, vérificateur gratuit | **≥ 4 requêtes** similaires |
| Seuil de rentabilité, auto-cohérence | **≥ 33 requêtes** — la vérification est 93 % du coût |
| Registre à 65 536 emplacements | 201 Mo, contre 158 Mo de poids |

La vérification domine le coût. C'est le vrai obstacle, pas l'écriture.

### D.5 Le risque, mesuré et sévère

Consolider depuis des solutions **correctes** dégrade quand même l'exactitude : l'étude
citée par W4 rapporte un modèle échouant sur 54 % de problèmes ARC-AGI qu'il avait
précédemment résolus, avec une utilité de la mémoire qui monte puis **redescend sous le
niveau sans mémoire**. C'est le même motif que W3 avait signalé sur la consolidation de
contexte.

Deux protections non négociables en découlent : une **quarantaine** avant admission dans
le registre, et un chemin `λ = 0` toujours atteignable — c'est-à-dire la possibilité de
désactiver la mémoire à l'exécution et de retrouver exactement le modèle sans mémoire.

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
| Le calcul non cumulatif | **Mécanisme implémenté, et mesuré comme mémorisant par construction.** L'observation est déjà nommée ailleurs ; l'axe choisi est probablement le mauvais. Porte 0 avant toute dépense. |
| Expressivité des mélangeurs à état borné | **Un bug d'un caractère corrigé** (β), vérifié : hasard → 0.996 sur la parité généralisée en longueur. Le rappel multi-clés reste ouvert, et empire avec *k*. |
| La couverture factuelle | Borne physique. Contournée, jamais franchie. |
| La généralisation hors distribution | Aucune méthode connue. Hors périmètre, et nous le disons. |

Les quatre tracks W1–W4 sont chargés de contredire ce document là où il a tort. Leurs
rapports arriveront dans `docs/research/W*.md`.
