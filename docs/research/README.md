# Synthèse des douze tracks de recherche

Douze agents ont travaillé en parallèle, un par verrou de
[`../00_PROBLEM_LANDSCAPE.md`](../00_PROBLEM_LANDSCAPE.md). Chacun devait produire un
rapport chiffré, sourcé, et se terminant par **une** recommandation concrète assortie
d'un plan d'ablation.

Volume produit : **~7 500 lignes**, 12 rapports, plusieurs centaines de références.

> ### Avertissement de provenance — à lire avant les chiffres
>
> L'accès web des agents a été **dégradé pendant l'exécution** : le proxy sortant a bloqué
> arxiv.org, HuggingFace et ACL Anthology, et le quota de recherche partagé s'est épuisé en
> cours de route. Plusieurs tracks (R03, R09, R11 notamment) ont donc écrit une partie de
> leurs chiffres **de mémoire**, et l'ont signalé eux-mêmes.
>
> Les rapports marquent ces éléments (`†`, `[P]`, `VERIFY`). **Aucun chiffre non vérifié ne
> doit servir de base à une dépense de compute avant reprise de vérification.** C'est
> particulièrement vrai du tableau de bord de R11, qui est un artefact de fixation
> d'objectifs, pas une preuve.

---

## La convergence

Le résultat le plus important n'a été demandé à aucun agent. **Cinq tracks indépendants,
plus notre propre calculateur écrit séparément, sont arrivés à la même conclusion :**

| Source | Verdict |
|---|---|
| `prophet.scaling` (interne) | 73× à 7 326× sous le compute des concurrents |
| R02 | « 300 heures-A100 ≈ 5.6B tokens à 1.2B actifs — nous ne pouvons pas surpasser Qwen3 en pré-entraînement. » |
| R04 | « ~2 500× sous le compute de Qwen3-1.7B. La récurrence achète de la profondeur, pas de la connaissance. » |
| R05 | « 466× — le multiple de compute nécessaire pour égaler Qwen3-1.7B depuis une init aléatoire. » |
| R07 | « ~3 700× moins de compute que Qwen3-1.7B. Aucun optimiseur ne comble cet écart. » |
| R01 | Budget insuffisant pour un frontend octet entraîné de zéro ; retrofit d'un checkpoint. |

Quatre d'entre eux recommandent explicitement la **conversion de donneur** ou la
distillation comme chemin réaliste. C'est la décision D10, laissée ouverte dans
[`../01_ARCHITECTURE.md`](../01_ARCHITECTURE.md) §7.

---

## Seconde vague : les murs (W1–W4)

Une seconde série de tracks attaque non plus les problèmes *attaquables sous notre budget*
mais le **mécanisme** de verrous plus profonds. L'analyse de départ, avec son arithmétique,
est dans [`../07_WALLS.md`](../07_WALLS.md) ; les quatre agents sont chargés de la
contredire là où elle a tort.

| Track | Question |
|---|---|
| [W1 — Chain-of-thought](W1_chain_of_thought_wall.md) | Le CoT est-il un goulot d'information de ~2000:1, et la discrétisation est-elle simultanément ce qui le stabilise ? Et la distinction profondeur/bloc-notes tient-elle ? |
| [W2 — Expressivité des transformeurs](W2_transformer_expressivity_wall.md) | Que la boucle achète-t-elle *formellement* ? Et que perdons-nous en remplaçant l'attention par des mélangeurs à état borné ? |
| [W3 — Apprentissage continu](W3_continual_learning_wall.md) | Le troisième étage de mémoire — la distillation vers les poids — que nous n'avons pas construit. Et la mesure qui distingue une mémoire d'une compétence. |
| [W4 — Le calcul ne se cumule pas](W4_compute_does_not_compound.md) | Le mur que personne ne nomme. Est-il déjà résolu ailleurs, et notre mécanisme généralise-t-il ou mémorise-t-il ? |

### Ce que les tracks W ont trouvé dans le code, pas dans le plan

Les quatre rapports ont corrigé l'analyse de départ sur quatre points de fond, et deux
d'entre eux étaient des **défauts du code** :

| # | Trouvaille | Conséquence |
|---|---|---|
| 1 | `beta = sigmoid(...)` bornait la force d'écriture à (0,1), donc toutes les valeurs propres de la transition d'état restaient positives — **la parité était hors d'atteinte**. | Vérifié sur notre implémentation : 0.51 (hasard) contre **0.996** à 4× la longueur d'entraînement. Un caractère. `linear_beta_max` est désormais un interrupteur, et un invariant refuse 1.0. |
| 2 | `prophet_500m_probe.json` n'implémentait pas la pile documentée : la seule couche d'attention était **dans la boucle**, prélude et coda purement récurrents, `nope_layers` vide. | `design_warnings()` attrape cette classe d'erreur ; les configurations sont désormais générées et refusées si elles la déclenchent. |
| 3 | « 99.95 % de l'état est jeté » est **faux** — le cache KV conserve tout. Le vrai mécanisme : le token émis est le seul chemin qui **rentre à la couche 0**. | Cadrage corrigé ; le rapport passe de 2 180:1 à 250–750:1. |
| 4 | Notre décision D1 (« pas d'attention dans la boucle ») **est** la décision qui a supprimé le bloc-notes. Et un *k* constant n'achète **aucune classe de complexité**. | D1 documenté comme arbitrage et non comme élégance. La halte passe d'option à exigence, et elle est implémentée. |
| 5 | L'adressage de `consolidate_depth` **mémorise par construction** : recouvrement de Jaccard 0.530 contre 0.493 entre instances de même et de différente classe — le hasard. | Avertissement inscrit dans le code. Porte 0 (exactitude contre profondeur) obligatoire avant toute dépense. |
| 6 | Le « mur que personne ne nomme » **est nommé** (*sleep-time compute* et d'autres). | Revendication corrigée ; ce qui reste inédit est l'écriture sans rétropropagation, pas l'observation. |

---

## Troisième vague : le pilier agentique (A1–A4)

Une revue complète du dépôt, puis trois tracks sur ce qui sépare un modèle qui répond
d'un agent qui accomplit. La synthèse et ce qui en est construit sont dans
[`../08_AGENT.md`](../08_AGENT.md).

| Track | Verdict en une ligne | Effet sur le code |
|---|---|---|
| [A1 — Revue du code](A1_codebase_review.md) | 23 défauts, dont 7 silencieux : biais d'oubli remis à zéro par l'init, gradient de halte identique à toutes les positions, amortissement résiduel ×0.11 à l'exécution détruisant toute copie de donneur, `nope_layers` jamais lu, estimateur de paramètres 10 % trop bas. | Tous corrigés, chacun avec un test comportemental (`tests/test_review_fixes.py`). L'estimateur colle au modèle réel à 1e-4. Le chemin de données réel (B17) est construit : fichiers ou Hub, décontamination dans le flux, plafond d'époques au tirage, phases reprenables. |
| [A2 — Compétence agentique](A2_agentic_competence.md) | Les échecs d'un agent à 1–4B sont l'**omission** et la **perte de contexte long**, pas la syntaxe ; compacter par résumé fait passer les violations de 0 % à 30–59 %. | Boucle à fil unique : préfixe épinglé, fenêtre d'observations verbatim, éviction sans réécriture, retour arrière O(1). |
| [A3 — Action structurée](A3_structured_action.md) | La grammaire supprime le formatage (déjà rare) ; les têtes d'action (pointeur sur les ancres de schéma, copie de span) visent les vraies erreurs. | Grammaire préfixe et décodeur contraint construits ; têtes non construites, derrière `heads.action_head`. |
| [A4 — Auto-vérification](A4_self_verification.md) | Une tête de confiance peut porter le *agir*, jamais le *retenir* ; un désaccord entre profondeurs est un signal gratuit à mesurer. | Tiers de vérification, quarantaine à provenance, promotion/révocation. AUROC du désaccord de profondeur : expérience A4-0 non lancée. |

### Ce que le branchement a trouvé, que les rapports n'avaient pas vu

| # | Trouvaille | Conséquence |
|---|---|---|
| 7 | Lire une observation à *k*=1 puis penser à *k*=8 **sur le même cache** n'est pas défini pour un modèle entraîné à une profondeur par séquence : l'état de l'itération 8 n'a jamais vu l'observation. | `recurrent.token_depth` : plafonds de profondeur par token, cœur exécuté sur la sous-séquence compactée à chaque itération, en entraînement comme en inférence. Équivalence passe complète / décodage incrémental testée à 1e-4. Non ablaté ; hors des configs livrées. |
| 8 | Le biais de routage MoE était mis à jour **dans** le forward. Sous checkpointing d'activations — activé par défaut — le recalcul du backward route autrement et **plante** (`CheckpointError`) : aucune config MoE ne pouvait s'entraîner. | Le pas de rééquilibrage est enregistré dans les statistiques et appliqué **après** le backward, une fois par appel. Test de régression. |

---

## Verdicts par track

| Track | Verdict en une ligne | Effet sur la conception |
|---|---|---|
| [R01 — Tokenisation](R01_tokenization.md) | **Ne pas** construire un Prophet sans tokenizer de zéro. Le frontend octet tourne à **0.44× le débit** à l'échelle mini. La table de hash n-grammes de BLT fait **3.07B paramètres** pour un modèle « 1B ». | BPE 32 768 à repli octet, chiffres isolés. Frontend octet = retrofit ultérieur, porte de décision à 2 h. |
| [R02 — Attention](R02_attention_long_context.md) | Hybride 3:1 : gated DeltaNet majoritaire, fenêtre glissante + attention globale **sans encodage positionnel**. **4 Kio/token** de cache contre 32 pour un dense. | Pile retenue. Risque n°1 : effondrement du rappel multi-clés (89.8 % aiguille unique → **37.8 %** multi-aiguilles). |
| [R03 — Mémoire](R03_memory_continual_learning.md) | Mémoire à deux étages, une seule primitive (la règle delta) utilisée deux fois. L'oubli sparse perd **11 %** contre **89 %** en fine-tuning complet. | Ancré, activable. Ablation non financée au budget actuel. |
| [R04 — Raisonnement](R04_reasoning_test_time_compute.md) | Cœur récurrent partagé : **7.6× de compression paramétrique** à k=8. Piège signalé : Huginn met en cache le KV **par étape de récurrence**. | Pari central. Le piège KV est évité par construction (attention hors boucle). **Avertissement : la récursion sous-performe à 135M et ne gagne qu'à partir de ~360M** — l'ablation doit être à ≥ 350M. |
| [R05 — Sparsité](R05_sparsity_moe.md) | MoE oui, mais **5.1B total / 1.07B actifs**, pas 8–12B. Chemin : dense d'abord, puis upcycling. | Taille revue à la baisse. **L'iPhone n'aura pas de MoE** : l'ANE ne sait pas faire de gather d'experts par token. |
| [R06 — Données](R06_data_efficiency.md) | Trois phases 70/20/10 avec un recuit anormalement gras. Maths + code sur-pondérés à **26.7 %**. Multilingue plafonné à 2.5 %. | Encodé et validé dans `prophet/data/recipes.py`. |
| [R07 — Optimisation](R07_optimization_training.md) | Muon + AdamW, WSD. **Le gain réel de Muon est 1.1–1.4×, pas 2×.** Un MoE de 10B ne tient pas : **97 Gio contre 77 utilisables**. | Recette implémentée. La contrainte mémoire a tranché le conflit de taille avec R04. |
| [R08 — Quantification](R08_quantization_on_device.md) | **Ne pas** pré-entraîner en basse précision : le A100 est sm_80, un QAT serait simulé — tout le coût, zéro accélération. | Architecture consciente de la quantification dès le pas 0. Correction : les têtes MTP donnent **1.07–1.46×**, pas 2×, sur un MoE. |
| [R09 — Hallucination](R09_hallucination_calibration.md) | À ~2 bits/paramètre, Prophet ne peut contenir que **1–3 % de Wikidata**. L'abstention calibrée seule fait passer SimpleQA de ~3.0 à ~4.6. | Tête de confiance, porte de décision AUROC à 3 h. **Gain gratuit signalé : des tokens de provenance en pré-entraînement font passer la pénalité des données bruitées de 20× à 2×.** |
| [R10 — Post-entraînement](R10_post_training.md) | La distillation on-policy a battu le RL **sur toutes les métriques, à 1 800 contre 17 920 heures-GPU**. GRPO est faisable sur un A100 en LoRA. | La distillation prend le compute, le RL est du polissage. **Risque de licence critique** (ci-dessous). |
| [R11 — Évaluation](R11_evaluation.md) | Sous 500M paramètres, la plupart des benchmarks sont au hasard. **Décider sur le BPB, pas sur l'exactitude.** | Système à trois niveaux. Métrique phare proposée : qualité par gigaoctet. |
| [R12 — Multimodalité](R12_multimodal.md) | **Plus tard, mais les ancrages maintenant.** Coût des ancrages : ~0.5M paramètres, zéro FLOP. Coût de leur omission : un ré-entraînement complet. | Ancrages H1–H10 en v1. Audio : non. |

---

## Trois trouvailles transverses

**1. Un risque de licence qui pouvait rendre le projet impubliable.** R10 a lu les
conditions Gemma plutôt que leur résumé : §1.1(e) définit un modèle entraîné sur des
*sorties synthétiques de Gemma* comme un « Model Derivative », qui hérite des conditions
Gemma. **Une seule ligne générée par Gemma suffirait à rendre Prophet impubliable sous
Apache-2.0.** Vérifié aussi : `AM-DeepSeek-R1-Distilled-1.4M` — cité dans le cahier des
charges initial — est en **CC-BY-NC-4.0** ; `OpenHermes-2.5` n'a **aucune licence**.

→ Transformé en garde-fou exécutable : `prophet.data.mixture` **refuse** ces licences à la
validation, avant tout téléchargement. C'est une erreur et non un avertissement, parce que
le coût est asymétrique : découvrir le problème après l'entraînement ne se répare que par
un ré-entraînement.

**2. Un piège de déploiement évité par construction.** R04 signale que l'implémentation de
référence de la profondeur récurrente met en cache le KV *par étape de récurrence*,
multipliant la mémoire par *k*. Notre conception confine l'attention au prélude et au coda ;
mesuré : le cache d'attention est **identique à l'octet près** entre k=1 et k=8.

**3. Un conflit de conception arbitré par l'arithmétique.** R04 spécifiait 9.48B
paramètres ; R07 a calculé que cela demande 97 Gio contre 77 disponibles. Plutôt que de
faire la moyenne, `scripts/design_search.py` a énuméré l'espace de conception sous
*toutes* les contraintes simultanément et trouvé ce qui les satisfait réellement.

---

## Ce que la recherche n'a pas résolu

- **La couverture factuelle.** Borne physique. Contournée, pas résolue.
- **La vérification des chiffres.** Voir l'avertissement de provenance. C'est la première
  tâche de la semaine 1.
- **La décision D10.** Zéro ou conversion de donneur. Ouverte, et elle appartient au
  porteur du projet.
