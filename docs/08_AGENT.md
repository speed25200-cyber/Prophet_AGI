# 08 — Le pilier agentique

> Tracks A2 (compétence agentique), A3 (action typée), A4 (auto-vérification).
> Implémenté dans `prophet/agent/`. Ce document dit ce qui est construit, ce qui est
> mesuré, et ce qui ne l'est pas — dans cet ordre.

---

## 1. Ce que « agentique » veut dire une fois décomposé

Un agent n'échoue pas « parce qu'il est petit ». A2 a décomposé les échecs par cause,
sur les taxonomies publiées (MAST, trajectoires SWE-agent, HORIZON) :

| Cause | Part mesurée | Mécanisme retenu |
|---|---:|---|
| Terminer trop tôt / prétendre avoir fini | 18.6 % | `done` **refusé** tant qu'un vérificateur ou la tête de confiance ne l'accepte pas |
| Vérification absente ou fausse | 17.3 % | Une action irréversible sous le seuil déclenche `verify` d'abord |
| Décalage raisonnement / action | 13.2 % | Le span d'action est **contraint par grammaire**, le span de pensée ne l'est jamais |
| Répétition d'étapes (boucles) | 15.7 % | Détecteur de boucle dans le *harnais*, pas dans le modèle : il ne se voit pas tourner en rond |
| Ne pas demander | 6.8 % | `ask` = une question **et** l'action qui la résoudrait, jamais une abstention nue |
| Perte de contexte sur horizon long | 27.5 % des échecs de conception | Préfixe **épinglé, jamais compacté** ; fenêtre d'observations verbatim ; au-delà, **compaction portée par l'état** |

Le dernier point est celui que notre architecture attaque le mieux. Compacter un contexte
en le résumant fait passer les violations de contraintes de 0 % à 30–59 %. Nous ne
résumons rien : les observations trop anciennes sont évincées du cache d'attention et ne
survivent que par ce que l'état récurrent borné en a retenu. Rien n'est réécrit, donc rien
n'est réécrit faux.

---

## 2. La boucle, telle que construite

```
   instantané du cache ─► penser (texte libre, halte apprise, budget) ─► agir (grammaire,
   k profond, glouton) ─► PORTES ─► détecteur de boucle ─► exécuter ─► ingérer à k=1
   comme modalité distincte ─► évincer la plus vieille observation hors fenêtre
```

Une seule place où se prennent les décisions « agentiques » : les portes.

| Porte | Question | En dessous du seuil |
|---|---|---|
| `τ_act` | L'action irréversible est-elle correcte ? | `verify` d'abord |
| `τ_done` | Le but est-il atteint ? | Refus ; le vérificateur exécutable tranche s'il existe |
| `τ_ask` | Le doute porte-t-il sur l'*intention* de l'utilisateur ? | `ask` avec l'action proposée |

Ce qui est délibérément **absent** : planificateur séparé, critique, sous-agents,
résumeur. Chacun est l'endroit où la littérature trouve une nouvelle classe d'échec.

Le retour arrière est en **O(1)** : un instantané complet du cache par pas (état récurrent
+ fenêtres d'attention bornées), coût constant en longueur d'épisode. `rollback(step)`
restaure ; il ne rejoue pas.

### 2.1 Trois profondeurs sur un seul cache — ce que cela exige du modèle

Le schéma ci-dessus lit l'observation à *k*=1, pense sous halte apprise et agit à
`k_decide`. Sur un cache, ce n'est **pas gratuit** : l'état récurrent de l'itération *i*
n'existe que pour les tokens qui ont exécuté l'itération *i*. Un modèle entraîné à une
profondeur par séquence n'a *aucun comportement défini* pour un token profond qui suit un
token peu profond — il lirait un état qui n'a jamais vu une partie du contexte. La
première version de la boucle faisait exactement cela, et le modèle réel la refusait
(`Depth is fixed for a cache's lifetime`).

Deux régimes, tous deux exacts, choisis par `AgentConfig.depth_policy` :

| Régime | Modèle requis | Ce qui se passe |
|---|---|---|
| `fixed` | n'importe lequel | Tout l'épisode à `k_decide` ; la halte apprise ne peut que **baisser** la profondeur (les créneaux plus profonds se retirent, jamais lus à nouveau). Exact par construction, testé. |
| `token` | `recurrent.token_depth=True` | Chaque token porte un **plafond** de profondeur. À l'itération *i*, le cœur tourne sur la sous-séquence **compactée** des tokens encore actifs ; les autres gardent l'état de sortie qu'ils avaient. L'entraînement fait la même chose (spans `<|tool|>` à `ingest_depth`, plus un span aléatoire par séquence avec probabilité 0.25), donc « lire à 1, penser à 8 » est *dans la distribution*. |

L'équivalence est vérifiée numériquement, pas argumentée : la passe complète avec un
vecteur de plafonds égale, à 1e-4, un décodage incrémental qui nourrit chaque token à sa
propre profondeur (`tests/test_token_depth.py`). `auto` lit la config du modèle et choisit.

**Ce qui n'est pas validé :** que l'entraînement à plafonds par token ne coûte rien en
qualité par rapport à une profondeur par séquence. C'est une ablation obligatoire de la
recette A2 (§6) avant que `token_depth` n'entre dans une config livrée ; les configs
générées le laissent à `False`.

### 2.2 Les ids de contrôle ne se devinent pas depuis le texte

La boucle compose ses prompts en épissant les ids spéciaux explicitement (`<|system|>`,
`<|assistant|>`, `<|tool|>`) et encode le but, les schémas d'outils et **surtout les
observations** comme du texte brut. `ProphetTokenizer.encode` n'interprète les chaînes
`<|…|>` que sur demande (`parse_special=True`). Une sortie d'outil qui contient
`"<|assistant|>"` reste donc treize caractères : c'est la seule défense mécanique contre
une injection qui franchirait la frontière outil → modèle.

---

## 3. Actions typées : ce qu'A3 a corrigé dans l'hypothèse

L'hypothèse était « l'appel d'outil échoue par formatage, donc contraindre la sortie ».
Le mode d'échec dominant à 1–4B est **l'omission** (~68 % des échecs) et les **mauvaises
valeurs** d'arguments (~79 % des erreurs restantes) — pas la syntaxe. La contrainte
grammaticale règle la classe minoritaire à coût négligeable, et **doit se limiter au span
d'appel** : contraindre le raisonnement coûte 28–36 points à un modèle limité en capacité.

Construit : `ToolSchema`, `Action` (avec hachage canonique pour la détection de boucle),
`ToolRegistry` (validation des arguments), `ActionGrammar` (validateur de **préfixe**
JSON, conscient du schéma : nom d'outil, clés, types) et `ConstrainedDecoder` (décodeur
de référence qui masque la tête LM aux tokens gardant le préfixe viable). Six identifiants
de contrôle réservés dans le tokenizer : `<|tool_def|>`, `<|/tool_def|>`, `<|call|>`,
`<|/call|>`, `<|copy|>`, `<|nocall|>`.

**Construit — les têtes d'A3** (`heads.action_head`, `prophet/modeling/action.py`) :

| Tête | Lit | Décide | Paramètres (d=1536) |
|---|---|---|---|
| Sélection | l'état à `<|call|>` contre les états aux ancres `<|/tool_def|>` du prompt, plus une clé nulle apprise | *quel* outil, ou « aucun » (les actions réservées) — un seul argmax sur *n*+1 options, pas une chaîne de tokens | 2·d·d_k + d_k = 0.39M |
| Copie | l'état au début d'une valeur contre les **clés existantes** de la couche d'attention globale NoPE du coda | *où* dans le contexte commence et finit la valeur — zéro octet de cache en plus | 2·d·head_dim = 0.39M |
| Porte | l'état au début d'une valeur | copier ou générer | d + 1 |

Les cibles se lisent **dans le flux de tokens** lui-même (`build_action_targets`) : un
exemple SFT rendu avec les ids de contrôle dit déjà quel outil a été appelé (index de son
schéma parmi les ancres, 0 si aucun ou réservé) et quelles valeurs d'arguments étaient
verbatim dans le contexte (dernière occurrence alignée sur les tokens ; sinon la porte
apprend « non copiable »). Aucun second format de données à laisser dériver. La perte
ajoute sélection, pointeurs et porte, et abaisse à 0.1 le poids LM des tokens qu'un
runtime typé émet à la place du modèle (syntaxe, nom d'outil, noms de paramètres).
L'apprenabilité est vérifiée sur un exemple : sélection exacte, pointeurs exacts.

Au décodage, la boucle lit la sélection à `<|call|>` et **restreint la grammaire** à l'outil
choisi (ou aux seules actions réservées) ; la marge top-1 − top-2 est enregistrée par pas
comme signal d'ambiguïté. Le trainer refuse `action_head` sans tokenizer, puisque les
cibles en dépendent.

**Copie au décodage — construite.** La grammaire rend désormais l'état « début de
valeur » avec la clé et le type attendus. Dans un span d'action, chaque token nourri
score aussi le pointeur de copie à sa propre position ; au début d'une valeur, si la porte
dit *copier*, la boucle lit le span choisi (début, fin ≥ début) dans tout ce qui a été
nourri, le rend dans le JSON que le schéma attend (chaîne, entier, nombre, booléen) et ne
l'accepte que si la grammaire l'accepte — sinon elle génère. Une valeur copiée entre dans
le cache comme si elle avait été générée et coûte un pas, pas douze. Vérifié : une chaîne
copiée depuis le but remplit l'argument ; un span qui ne tient pas dans le type (un mot
pour un entier) est refusé et la génération reprend.

**Non construit :** le chemin par slots fermés (`<|slot_i|>`, contexte −95 %). A3-4
décide s'il survit.

---

## 4. Vérification : le mur derrière le mur

Le résultat d'A4 qui fixe la forme de tout le module : **sous ~7B, un modèle qui vérifie
son propre raisonnement ne fait pas mieux que le hasard** — les erreurs du vérificateur
sont couplées à celles du générateur. La vérification n'est plus facile que la génération
que dans trois cas : exécution ou consultation, comparaison de candidats, vérificateur
entraîné séparément.

D'où la règle, dérivée de l'arithmétique et non d'une préférence :

> Réfléchir et réessayer en proportion de ce que coûte la vérification : tentatives
> illimitées derrière un test exécutable, une seule derrière un vérificateur appris,
> aucune derrière rien — puis demander. Ne consolider que ce qu'un programme ou trois
> exécutions indépendantes ont confirmé ; **une tête de confiance peut gouverner l'action,
> jamais la mémoire.**

Un vérificateur appris à AUROC 0.80 admet ~30 % de réponses fausses en mémoire, ce qui
plafonne la classe *sous* ce qu'une passe profonde obtient déjà. C'est pourquoi les tiers :

| Tier | Qui a vérifié | Peut agir | Peut être mémorisé |
|---|---|---|---|
| 0 | Un programme | oui | **oui, immédiatement** |
| 1 | ≥ 3 exécutions indépendantes | oui | en quarantaine, promu par accord ultérieur |
| 2 | Une tête apprise | oui | **jamais** |
| 3 | Rien | — | refusé à l'entrée |

Signaux lus (tous gratuits sauf l'exécution) : tête de confiance, entropie et marge des
tokens, **désaccord entre profondeurs** (itération 2 contre finale), profondeur attendue
de la halte, **désaccord entre la tête MTP et la tête principale**, résultat d'exécution.
Les deux en gras sont propres à cette architecture et **n'ont aucune AUROC publiée** :
c'est l'expérience A4-0, une passe à k=8 sur ~1 500 exemples, gratuite, `auroc()` en une
ligne.

**Non construit :** l'entraînement du scoreur sur des étiquettes produites par un
programme (3–8 h-A100 estimées) ; il tourne sur un prior déclaré comme tel (`fitted=False`).

---

## 4 bis. La boucle fermée : de la quarantaine au corpus

Le mur C ([`07_WALLS.md`](07_WALLS.md)) dit que l'apprentissage continu a besoin d'un
troisième étage — la distillation de l'expérience vers les poids — et que le gradient ne
le fournit pas seul. Pour un agent, cet étage a maintenant un chemin mécanique :

```
épisode ─► quarantaine (provenance, tiers) ─► promotion (vérité terrain immédiate,
consensus après accord, appris jamais) ─► rendu en flux à ids de contrôle ─► source du
chargeur (parse_special) ─► cibles des têtes d'action lues dans ce flux ─► entraînement
```

`prophet/agent/render.py` rend un épisode promu exactement comme la boucle l'aurait
produit — but, schémas, pensée, appel ou `<|nocall|>`, observation — et l'expose comme
source de documents ; les pas malformés sont omis (ils ne sont pas une décision à
apprendre). Les trajectoires conservent la pensée et l'observation complète pour cela :
un résumé ne se dé-résume pas. Rien ici n'élargit ce que la promotion a permis : une
trajectoire rendue ne porte aucune provenance propre.

Ce qui n'est **pas** résolu par ce chemin : la mesure qui distingue une compétence d'une
mémoire (W3), et le coût en oubli d'un tel flux à 250M — les deux sont dans la recette A2,
non financée.

## 5. Où un petit agent peut réellement gagner

Pas sur SWE-bench ou GAIA en ensemble ouvert. Sur deux terrains :

1. **Familles de tâches répétées dans un environnement fixe** — même dépôt, mêmes outils,
   même utilisateur. Un modèle gelé a une courbe d'apprentissage plate ; le nôtre
   consolide des procédures vérifiées par famille, et `pass^k` passe du pile-ou-face au
   quasi-certain. C'est le benchmark « courbe d'apprentissage » à construire contre une
   ligne de base gelée 10× plus grosse.
2. **Sessions longues sur appareil** — 128k+ tokens, des centaines de pas. À ~2 KB/token
   de KV + état contre 32–131 KB/token pour un dense de 1–8B, c'est la différence entre une
   session qui tient sur le téléphone et une qui ne tient pas.

---

## 6. Ce qui est mesuré, et ce qui ne l'est pas

| Propriété | Statut |
|---|---|
| Grammaire de préfixe : jamais « viable » pour une chaîne incomplétable | testé |
| Décodeur contraint : fin d'appel autorisée seulement si complet | testé |
| Portes `done` / irréversible / `ask` | testées avec un modèle scripté |
| Détecteur de boucle → réflexion | testé |
| Retour arrière O(1) rejoue un pas à l'identique | testé (1e-6) |
| État récurrent d'un instantané constant en longueur d'épisode | testé |
| Quarantaine : tiers, promotion, persistance, **révocation** par version de vérificateur | testé |
| Règle de décision (exécution ≫ appris ≫ rien) | testée |
| **Compétence** de l'agent | **non mesurée** — le modèle n'est pas entraîné |
| AUROC du désaccord de profondeur comme prédicteur d'erreur | **non mesurée** — A4-0 |
| Têtes d'action typées : sélection, copie, porte, cibles depuis le flux, sélection et copie au décodage | **construites, entraînées une fois à 7M paramètres** |
| Chemin quarantaine → corpus (rendu, source, cibles) | **construit, exercé : les 600 épisodes du premier run agentique y sont passés** |
| État de session porté d'un épisode à l'autre (R03 appliqué à l'agent) | **construit, effet non mesuré** |
| Benchmark à vérificateurs et courbe d'apprentissage (`prophet/eval/agent_bench.py`) | **construit ; premier chiffre à 7M paramètres : 0 % → 55 % / 32.5 % de succès sur tâches inédites, copie d'arguments = +40 points ([`09_FIRST_RUN.md`](09_FIRST_RUN.md))** |
| Scoreur entraîné, recette d'entraînement agentique (~67 h-A100, A2 §8), chemin par slots fermés | **non construits** |
| Plafonds de profondeur par token (`recurrent.token_depth`) : mécanique exacte, entraînement câblé | **construit, non ablaté** |
