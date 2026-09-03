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
