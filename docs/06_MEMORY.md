# 06 — Mémoire persistante

> Track R03, prototypé dans `prophet/memory/`. C'est le pari le plus spéculatif du
> projet : une mémoire modifiable après déploiement.
>
> Statut : **mécanique locale mesurée à l'échelle jouet ; apprentissage continu non
> démontré.** L'ablation qui décide de son intégration est financée en priorité 2
> ([`05_ROADMAP.md`](05_ROADMAP.md)).

---

## 1. Le problème

Les modèles actuels ont les poids gelés. Leur seule mémoire est la fenêtre de contexte —
volatile, coûteuse en O(N), rejouée intégralement à chaque requête. Le fine-tuning, lui,
provoque un **oubli catastrophique**.

Le résultat sur lequel repose toute la conception, rapporté par R03 :

| Méthode | Connaissance antérieure perdue, à connaissance nouvelle égale |
|---|---:|
| Fine-tuning complet | **89 %** |
| LoRA | 71 % |
| **Sparse Memory Finetuning (SMF)** | **11 %** |

SMF sélectionne des rangées d'une couche mémoire préentraînée par contraste avec un corpus
de fond, puis les ajuste par gradient. Le registre Prophet emploie des clés aléatoires
gelées et une écriture fermée : le 11 % motive l'hypothèse de support creux mais **ne
mesure pas ce prototype**.

---

## 2. Deux étages, qui échouent différemment

| Étage | Où | Écrit par | Persistance | Taille (mini) |
|---|---|---|---|---|
| **1 — état de session** | Couches à delta gated | La passe avant elle-même | Une conversation, sérialisable | ~0.6 Mo |
| **2 — le registre** | Une couche à clés-produit | La passe de consolidation hors-ligne | Durable, partagé | ~33 Mo |

La séparation compte parce que les risques diffèrent. L'état de session est bon marché et
jetable : le perdre coûte une conversation. **Le registre est durable, donc dangereux** —
tout ce qui y entre persiste, y compris les erreurs. Il n'est donc jamais écrit
directement depuis une conversation en cours, seulement par une étape de consolidation
délibérée.

---

## 3. L'écriture en forme close

Le registre lit par somme pondérée de lignes de valeurs :

```
m(x) = Σᵢ aᵢ V[i]        avec aᵢ = wᵢ / n_heads
```

La dérivée de la sortie par rapport à une ligne de valeurs est donc simplement son poids.
Le système à résoudre pour déplacer la sortie de `−résidu` est sous-déterminé, et sa
solution de norme minimale s'écrit directement :

```
ΔV[i] = −lr · (aᵢ / ‖a‖²) · résidu
```

**Deux passes avant et un `scatter_add`.** Aucun gradient ne traverse le tronc, à aucun
moment. C'est ce qui rend l'opération exécutable sur un téléphone.

> **Le piège, rencontré en construisant ce module.** La division par `‖a‖²` n'est pas
> cosmétique : sans elle, le pas est celui d'un gradient à l'échelle près, et le registre
> sous-corrige d'un facteur `top_k`. Il apprend visiblement quelque chose — juste
> beaucoup trop lentement, sans que rien ne le signale.
>
> Vérifié par test dans le cas sans collision : un token isolé atteint sa cible à **1e-7
> près en une seule écriture**. Des tokens/têtes visant le même slot rendent le pas
> approximatif et doivent faire l'objet d'un test séparé.

### Clés gelées

Les clés ne bougent jamais après initialisation. Si elles dérivaient, chaque association
écrite auparavant pointerait silencieusement ailleurs, et la panne ne serait visible que
le jour où un fait ancien revient faux.

### Clés-produit

Adresser `n` emplacements directement coûte `n` produits scalaires. En scindant la requête
en deux moitiés confrontées à deux répertoires de `√n` sous-clés, on atteint les mêmes
emplacements pour `2√n` comparaisons. À 65 536 emplacements : **512 comparaisons au lieu
de 65 536**.

---

## 4. La consolidation, ou « sommeil »

Pendant une session, le modèle voit des choses qu'il ne peut pas garder. La passe de
consolidation tourne hors-ligne, après coup, et distille l'apport du contexte dans le
registre — pour qu'une session ultérieure en bénéficie **sans que le contexte soit
présent**.

```
h⁺ = f(contexte ‖ requête)      # le modèle a vu le contexte
h⁻ = f(requête)                 # le modèle ne l'a pas vu
cible = λ (h⁺ − h⁻)             # exactement ce que le contexte a apporté
registre.write(h⁻, cible)
```

> **Second piège, également rencontré.** R03 formule la cible comme `m(x) + λ(h⁺−h⁻)` —
> correct pour un petit pas de gradient, faux ici. Notre écriture étant la solution
> *exacte*, une cible incrémentale dépasse de tout ce que le registre contenait déjà. La
> première version de ce module faisait empirer le rappel à chaque passe : **1.00 → 1.44**.
> Avec une cible absolue : **1.00 → 0.003**.

---

## 5. Ce qui est mesuré

Toutes les valeurs viennent de `tests/test_memory.py`, sur un modèle jouet de 128
dimensions et un registre de 4 096 emplacements.

| Propriété | Mesure |
|---|---|
| Écriture exacte, token isolé | résidu → **1e-7** en une écriture |
| Rappel après consolidation, **contexte effacé** | erreur **1.00 → 0.003** |
| Oubli après apprentissage d'un second lot | erreur sur l'ancien : 0.000 → **0.229** |
| Idem, **avec replay à 25 %** | 0.000 → **0.145** (oubli réduit de **37 %**) |
| Taille de l'état de session | **indépendante de la longueur de conversation** |
| Poids du tronc modifiés pendant la consolidation | **aucun** (vérifié par test) |

La mesure qui compte est la deuxième : elle est prise **après effacement du contexte**.
C'est la seule qui distingue une mémoire d'un prompt plus long.

---

## 6. Honnêteté sur la portée

Ce qui est démontré à l'échelle jouet : une écriture isolée sans collision atteint sa
cible, un ledger externe peut reproduire l'effet résiduel du contexte sur les épisodes
écrits, le replay réduit une partie de l'interférence, et les poids du tronc ne bougent
pas. Ce chemin externe n'est pas encore aligné avec les points d'injection du modèle.

Ce qui **n'est pas** démontré : transfert vers des instances inédites, avantage face à un
retriever, reproduction du chiffre SMF de 11 %, amélioration d'un benchmark réel,
correction/suppression ciblée ou résistance à un usage prolongé. Ce sont les questions de
l'ablation E2, financée en priorité 2 :

> écrire → **effacer le contexte** → lire, comparé à une ligne de base de récupération
> d'information à budget de contexte égal.

Si elle échoue, l'étage 2 est abandonné et l'étage 1 — l'état de session sérialisable —
est conservé, parce qu'il est presque gratuit et déjà validé par les modèles hybrides
existants.

---

## 7. Risques

| Risque | État du prototype |
|---|---|
| Un exemple aberrant écrase un emplacement consensuel | Les contributions sont bornées avant agrégation ; la borne par slot après collision reste à implémenter |
| Les emplacements les plus utiles sont brassés à chaque session | **EWC allégé** : le pas décroît avec le nombre d'écritures |
| Dérive lente sur un déploiement long | La décroissance actuelle balaie toute la table ; TTL/tombstones ciblés sont requis |
| Effondrement sur quelques emplacements | `occupancy()` rapporte l'entropie d'écriture — un effondrement ne se voit **pas** dans la courbe de perte |
| Restauration d'un état produit par d'autres poids | **Empreinte du modèle** vérifiée au chargement ; refus par défaut |
| Empoisonnement de la mémoire | Aucun journal de provenance ni vérificateur attesté ; `verified=False` est seulement un défaut fail-closed minimal |
