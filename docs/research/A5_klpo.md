# A5 — KLPO : l'amélioration de politique régularisée par la KL vers l'échantillonneur, sans critique

**Track :** A5 · **Statut :** thèse importée, implémentée, ablation en cours · **Date :** 2026-09-21
**Source :** Zhang et al., *KL-Regularized Policy Optimization for Critic-Free Agentic
Reinforcement Learning*, rapport technique du 18 septembre 2026, révisé le 20 ;
dépôt [yifanzhang-pro/KLPO](https://github.com/yifanzhang-pro/KLPO) (Apache-2.0).
Lignée : RPG (Zhang et al., 2026), SPPO et GPO (Wu et al., 2024 ; Zhang et al., 2025),
BPO, *Score Centering* (Marek et Ryabinin, 2026), FlashREINFORCE (même auteur).

Convention de confiance : `[V]` vérifié sur le code du dépôt ; `[S]` résumé de recherche,
texte intégral du rapport non lu ; `[F]` chiffre de FlashREINFORCE, pas de KLPO.

## 1. La thèse en neuf pas

Soit π_sam la politique qui a produit l'épisode (le modèle d'il y a *k* mises à jour),
π_θ la politique entraînée, A_u l'avantage au token *u*, β la force de régularisation.

| Pas | Énoncé | Ce qu'il apporte |
|---|---|---|
| 1. Objectif local | J_u(π) = E_{y_u∼π} A_u − β·UKL(π‖π_sam) | l'amélioration se fait **autour de l'échantillonneur**, pas d'une politique de référence fixe |
| 2. Optimum de Gibbs | π⁺(y_u) = π_sam(y_u)·exp((A_u − c*)/β) | forme close ; c* est la log-partition |
| 3. Normaliseur exact | c* = β·log E_{π_sam} e^{A/β} = β·KL(π_sam‖π⁺) | la quantité cible est un KL, pas un critique |
| 4. Condition d'optimalité | β·log(π⁺/π_sam) = A_u − c* | l'optimum devient une **cible de régression** sur le log-ratio |
| 5. Régression à normaliseur fixe | L = (1/2β)·E Σ_u (β·ℓ_θ,u − A_u + c*)², ℓ_θ,u = log(π_θ/π_sam) | moindres carrés sur la condition, comme SPPO et GPO |
| 6. KLPO canonique | c* remplacé par sa valeur profilée c_θ = β·KL(π_sam‖π_θ) | plus de normaliseur à apprendre |
| 7. Régression par token | h_u = R − β·ℓ_θ,u, centré par sa moyenne conditionnelle sous π_sam ; même gradient | récompense terminale seule, pas d'estimation d'avantage |
| 8. Substitut exact | L^bp = −sg(h_u)·(log p_{y_u} − Σ_v q_v log p_v) ; ∇^AD = ∇L_tok | le gradient exact se réalise avec un score **centré sous l'échantillonneur** |
| 9. MC-KL | Σ_v q_v log p_v ≈ (1/M) Σ_j log p_{v_j}, v_j ∼ q iid | sans biais dès M = 1 ; des tirages de **tokens** aux préfixes visités, pas de déploiements |

Ce que la méthode supprime : le groupe de réponses par prompt (GRPO), le clipping du
ratio (PPO), la passe de politique de référence, le critique, le normaliseur appris.
Ce qu'elle exige : les log-probabilités par token de l'échantillonneur **telles qu'utilisées
à la génération**, et M tirages auxiliaires par préfixe avec leurs log-probabilités.

Implémentation de référence `[V]` : `klpo/loss.py`, fonction `klpo_token_loss` ; h calculé
sans gradient ; perte = −Σ_tokens h·(log p_a − moyenne_j log p_{v_j}) ; somme sur les
tokens sans normalisation par la longueur ; masque sur le prompt, le remplissage et les
sorties d'outils. Quatre estimateurs du terme de correction (MC, Top-K avec queue
agrégée, binaire, complet) ; seuls MC indépendant et complet ont l'identité exacte en
population. Le README : « GPU training and paper-scale benchmark reproduction have not
been validated ».

## 2. État de la preuve

| Affirmation | Preuve | Statut |
|---|---|---|
| Équivalence des gradients (régression, token, substitut) | dérivation du rapport, tests CPU du dépôt sur un arbre énuméré | `[V]` tests, `[S]` preuves |
| Stabilité à retard 8 contre GRPO à retard 1, Qwen3-30B-A3B avec outil Python, moyenne de trois bancs 66,4 | figure 3 du rapport **FlashREINFORCE** | `[F]` une graine, avg@4 |
| Résultats de KLPO à l'échelle | aucun publié | absent |

Les courbes de FlashREINFORCE disent : GRPO s'effondre deux fois (moyenne ≈ 15 % vers
500 mises à jour, ≈ 35 % vers 700) puis récupère ; les trois variantes de FlashREINFORCE
(porte δ, rejeu, sans porte) se superposent. La stabilité vient du cœur de la méthode,
pas des réglages. Ce cœur (échantillonnage d'importance par token, région de confiance
par séquence) n'est pas KLPO ; KLPO en est la reformulation par régression.

## 3. Pourquoi c'est pertinent pour Prophet

La boucle fermée (docs/31) entraîne aujourd'hui par rejet : seuls les épisodes vérifiés,
en entropie croisée. KLPO utilise **tous** les épisodes, avec R ∈ {0, 1} venant du
vérificateur exécutable (tier 0), un seul déploiement par tâche (notre budget), et
l'échantillonneur périmé qu'est le modèle du tour précédent comme référence de la KL.
Cette KL borne mécaniquement la dérive à chaque tour : c'est exactement ce que H3
(l'oubli) mesure. Et KLPO est asynchrone par construction, ce qui correspond au
fonctionnement par sessions de notre entraînement.

## 4. Ce que l'intégration approxime, et pourquoi

| Point | Choix | Raison |
|---|---|---|
| Tokens de politique | ceux que la boucle a **tirés** (span de réflexion, span d'action quand `sample_actions` est actif) ; exclus : prompt, observations, ids de contrôle, valeurs copiées par le pointeur, tokens décodés en glouton | pour un échantillonneur en masse de Dirac, la correction est nulle : rien à apprendre |
| Distribution de l'échantillonneur q | la distribution **réellement tirée** : logits masqués par la grammaire, divisés par la température, enregistrés à la génération avec M tirages auxiliaires | contrat du rapport : l'échantillonneur réel, pas une reconstruction |
| Distribution entraînée p | le modèle courant, sans masque, à température 1 | le masque n'est pas disponible à l'entraînement ; la KL tire alors la masse de p vers ce que la grammaire autorise, ce qui est le comportement voulu ; c'est une approximation déclarée |
| Génération du bras KLPO | température 1,0 pour les deux spans | rapproche q de p hors masque |
| Têtes d'action typées | inchangées par la perte KLPO ; entraînées par le fine-tuning sur épisodes vérifiés du même tour | KLPO ne couvre que les tokens de langage |
| Bras `closed-klpo` | fine-tuning par rejet **puis** K pas KLPO sur tous les épisodes du tour | isole l'apport de KLPO à budget de tâches égal |

Le flux d'entraînement de KLPO est le flux exact que la boucle a nourri au modèle
(`EpisodeResult.ids`), pas un rendu : la leçon de docs/09 (deux décalages
train/décodage à 55 % → 0 %) s'applique ici à la lettre.

## 5. Ce qui décidera

Un bras de plus dans le pilote de docs/31, avec la même comptabilité. Passe si, à budget
de tâches égal, `closed-klpo` gagne au moins autant que `closed` en succès tenu à l'écart
et dérive moins en bits par octet ; échoue sinon. Le critère est écrit avant que le bras
ne tourne (docs/31 §1, H5). Rien ici n'est adopté dans une configuration livrée avant
cette ablation.

## 6. Pièces

| Pièce | Fichier |
|---|---|
| Perte, tenseurs d'épisodes, boucle de mise à jour | `prophet/train/klpo.py` |
| Enregistrement de l'échantillonneur à la génération | `prophet/agent/loop.py` (`record_sampling`, `sample_actions`, `mc_draws`) |
| Bras `closed-klpo` | `scripts/closed_loop.py` |
| Tests | `tests/test_klpo.py`, `tests/test_closed_loop.py` |
