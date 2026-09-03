# 00 — Cartographie des limitations de l'IA actuelle

> Document fondateur du projet Prophet. Objectif : recenser **exhaustivement** les
> verrous qui bloquent la génération actuelle de modèles de langage, les classer par
> levier réel, et n'en retenir qu'un sous-ensemble attaquable avec **un seul A100 80GB**.
>
> Règle du projet : on ne construit rien qui ne réponde à un problème listé ici.

---

## 0. Méthode de priorisation

Chaque problème est noté sur trois axes :

| Axe | Signification |
|---|---|
| **Gravité** | À quel point ce verrou plafonne les capacités réelles (1–5) |
| **Levier@notre échelle** | Gain atteignable avec ~300 A100-heures et ≤1.5B params actifs (1–5) |
| **Risque** | Probabilité que la solution échoue ou coûte plus qu'elle ne rapporte (1–5) |

Le **score de priorité** = `Gravité × Levier / Risque`. Un problème très grave mais
insoluble à notre échelle (ex. : couverture factuelle du monde) est *délibérément*
contourné, pas attaqué de front.

### L'asymétrie fondamentale, chiffrée

Ces nombres sont produits par `python -m prophet.scaling`, pas estimés à la main.
Budget de référence : **300 heures-A100 à 35 % de MFU = 1.18e20 FLOPs**.

| Modèle | Paramètres | Tokens | FLOPs d'entraînement | Heures-A100 équivalentes | Écart |
|---|---:|---:|---:|---:|---:|
| Gemma-3-1B | 1.0B | 2.0T | 1.2e22 | 30 500 | **102×** |
| SmolLM2-360M | 360M | 4.0T | 8.6e21 | 22 000 | **73×** |
| Llama-3.2-1B | 1.24B | 9.0T | 6.7e22 | 170 000 | **568×** |
| SmolLM2-1.7B | 1.7B | 11.0T | 1.1e23 | 285 000 | **951×** |
| Qwen3-1.7B | 1.7B | 36.0T | 3.7e23 | 934 000 | **3 114×** |
| Qwen3-4B | 4.0B | 36.0T | 8.6e23 | 2 198 000 | **7 326×** |
| **Prophet** | **~500M actifs** | **~40B** | **1.2e20** | **300** | — |

Il faut nommer ce que cela implique, sans euphémisme :

> **Nous sommes 73× sous le plus modeste des concurrents.** Même SmolLM2-360M — le
> modèle le plus frugal de la liste — a coûté 22 000 heures-A100.

Deux conséquences structurent tout le projet :

1. **Les benchmarks de connaissance ne sont pas gagnables par pré-entraînement.**
   La couverture factuelle s'achète en tokens, et il nous en manque deux à trois ordres
   de grandeur. Attaquer MMLU de front consomme le budget sans résultat.
2. **La capacité par paramètre et par gigaoctet est gagnable.** La distillation depuis
   des professeurs ouverts, la profondeur de raisonnement achetée à l'inférence, et un
   mélange de données optimisé pour un petit budget sont des leviers qui **ne dépendent
   pas** de la taille du cluster adverse.

Corollaire opérationnel majeur, qui n'était pas dans le plan initial :
**la distillation n'est pas une option de post-entraînement, c'est la stratégie centrale
de pré-entraînement.** Apprendre depuis les logits d'un professeur fort transfère
l'information bien plus efficacement par token que la prédiction du token suivant sur du
texte brut. C'est le seul mécanisme connu qui permette d'importer le résultat de
36 000 milliards de tokens sans les payer.

### Point de fonctionnement retenu

À 300 heures-A100, la frontière compute-optimale se situe autour de 991M paramètres pour
19.8B tokens. Mais la mémoire d'inférence est notre contrainte dure, donc nous acceptons
un léger surcoût de perte pour un modèle plus petit et plus rapide :

| Paramètres actifs | Tokens | Tokens/param | Perte prédite | Taille int4 | Rôle |
|---:|---:|---:|---:|---:|---|
| 700M | 28B | 40 | 2.577 | 0.35 GB | compute-optimal |
| **500M** | **39B** | **79** | **2.582** | **0.25 GB** | **cible principale** |
| 350M | 56B | 160 | 2.597 | 0.17 GB | variante iPhone |

Le sur-entraînement (79 tokens/param contre 20 pour Chinchilla) est un choix délibéré :
il coûte 0.005 de perte prédite et rend le modèle 30 % plus petit et plus rapide à
l'inférence — exactement le bon échange quand la mémoire est rare.

On ne gagne **pas** en faisant la même chose en plus petit. On gagne uniquement en
changeant les termes du problème : distillation, architecture, allocation du budget de
tokens, calcul au moment de l'inférence, et mémoire persistante.

---

## 1. Le verrou du tokenizer

**Problème.** BPE/SentencePiece est une couche de compression heuristique gelée avant
l'entraînement, qui :

- rend le modèle aveugle aux caractères (compter les lettres, épeler, manipuler des chaînes) ;
- fragmente les nombres de façon incohérente, ce qui sabote l'arithmétique ;
- inflige une « fertilité » 2 à 5× plus élevée aux langues non-anglaises et au code
  (donc un coût d'inférence proportionnellement plus élevé pour ces usages) ;
- coûte, à notre échelle, une fraction énorme des paramètres. Mesuré par
  `prophet.budget` sur une configuration réaliste (`d_model=1024`, 16 couches,
  vocabulaire de 49 152, embeddings liés) : **50.3M paramètres sur 230.7M, soit 21.8 %
  du modèle** dépensés en table de correspondance qui n'effectue aucun calcul. Avec un
  vocabulaire de 128k, la part dépasse 40 % ;
- crée des « glitch tokens » et des angles morts de sécurité ;
- fixe définitivement le compromis compression/granularité avant d'avoir vu la moindre donnée.

**Pourquoi c'est un levier pour nous.** Chaque paramètre récupéré sur l'embedding
peut être réinvesti en profondeur ou en experts. Et un frontend adaptatif alloue le
calcul là où l'entropie est haute — c'est du calcul dynamique gratuit.

**Piste.** Patching dynamique par entropie sur des octets (lignée BLT / H-Net),
ou schéma hybride « patches au-dessus d'un petit vocabulaire ».

→ Track **R01**. Gravité 4 · Levier 4 · Risque 3.

---

## 2. Le coût quadratique de l'attention et le mur du cache KV

**Problème.** Deux coûts distincts, souvent confondus :

- **Prefill** : O(N²) en calcul — borne la longueur de contexte utilisable en pratique.
- **Décodage** : borné par la **bande passante mémoire**, pas par le calcul. Chaque token
  généré exige de relire *tous* les poids actifs + *tout* le cache KV.

Arithmétique sur nos cibles :

| Cible | Bande passante | Plafond théorique tok/s à 1.3B params actifs en 4 bits (~0.65 GB) |
|---|---|---|
| RTX 5090 | ~1.79 TB/s | ~2 750 tok/s |
| Mac Studio Ultra | ~0.8 TB/s | ~1 230 tok/s |
| iPhone 17 Pro | ~0.06–0.12 TB/s | ~90–180 tok/s |

Le cache KV s'ajoute à ce budget et le domine en contexte long : un transformeur dense
classique en GQA peut consommer plusieurs Go à 128k tokens — impossible sur iPhone.

**Piste.** Pile hybride : majorité de couches à état récurrent borné (coût mémoire
**constant** par token), minorité de couches d'attention complète pour le rappel exact,
plus compression latente du KV.

→ Track **R02**. Gravité 5 · Levier 5 · Risque 2.

---

## 3. Le cerveau gelé : aucune mémoire persistante, aucun apprentissage continu

**Problème.** C'est, à notre avis, **la** limitation qui sépare l'état de l'art de
quelque chose qui mérite le mot « intelligence » :

- les poids sont figés après l'entraînement ; le modèle ne peut rien apprendre d'une interaction ;
- la seule « mémoire » est la fenêtre de contexte — volatile, coûteuse en O(N), et rejouée intégralement à chaque requête ;
- le fine-tuning provoque un **oubli catastrophique** et n'est pas exécutable sur l'appareil ;
- les systèmes de mémoire externes (RAG, MemGPT…) sont des rustines applicatives, pas une propriété du modèle ;
- conséquence : aucune accumulation d'expertise, aucune personnalisation réelle, aucune tenue de tâche sur plusieurs jours.

**Pourquoi c'est notre meilleure asymétrie.** Un petit modèle qui *accumule* bat un
gros modèle figé sur toute tâche longue ou personnalisée. C'est un axe où la taille
du concurrent ne l'aide pas.

**Piste.** Mémoire à deux étages : (i) poids rapides intra-couche mis à jour en ligne
pendant la session, (ii) phase de « consolidation » hors-ligne qui distille la mémoire
de session en un delta de poids creux — l'analogue du sommeil.

→ Track **R03**. Gravité 5 · Levier 4 · Risque 4.

---

## 4. Le raisonnement : profondeur fixe et pensée verbeuse

**Problème.** Un transformeur standard dépense **exactement le même calcul** pour
« 2+2 » et pour une intégrale. La profondeur est fixe, donc la complexité du circuit
calculable par token est bornée. L'industrie contourne ça par le chain-of-thought,
qui a trois défauts pour nous :

- il verbalise la pensée en tokens → latence et cache KV proportionnels à la difficulté ;
- il est très coûteux sur appareil (des milliers de tokens de « réflexion » à 100 tok/s) ;
- la trace verbalisée n'est pas nécessairement le calcul réellement effectué.

**Pourquoi c'est un levier majeur.** Nous n'avons pas les paramètres ; nous pouvons
en revanche acheter de la capacité avec du **calcul au moment de l'inférence**.
Un bloc récurrent partagé bouclé *k* fois multiplie la profondeur effective sans
multiplier les paramètres — exactement le compromis qu'il nous faut quand la mémoire
est la ressource rare (32 Go, 8 Go). Bonus : *k* devient un cadran réglé à l'exécution —
petit sur iPhone, grand sur 5090.

→ Track **R04**. Gravité 5 · Levier 5 · Risque 3.

---

## 5. Densité : payer pour tous les paramètres à chaque token

**Problème.** Un modèle dense couple *capacité de connaissance* et *coût par token*.
Sur du matériel grand public où le décodage est borné par la bande passante, c'est
le pire compromis possible : on relit 100 % des poids pour produire un token qui n'en
concerne qu'une fraction.

**Piste.** Activation creuse (MoE à grain fin + expert partagé) pour découpler les deux :
~10B de paramètres de connaissance, ~1.3B lus par token.

**Contrainte dure.** Un seul GPU, pas de parallélisme d'experts, et les états
d'optimiseur d'un modèle de 10B ne tiennent pas naïvement dans 80 Go. Le chemin
« dense d'abord, puis *upcycling* creux » est probablement le seul viable.

→ Track **R05**. Gravité 4 · Levier 5 · Risque 4.

---

## 6. L'inefficacité en données (le verrou décisif pour nous)

**Problème.** Les modèles actuels ont besoin de 10T–36T tokens là où un humain
apprend sa langue avec ~10⁸ mots. L'efficacité par token est misérable, et personne
n'a de solution générale. Pour nous, ce n'est pas un problème académique : c'est
**le** facteur limitant.

Sous-problèmes :

- qualité des corpus (le web brut est majoritairement du bruit) ;
- allocation du budget de tokens entre domaines (web / code / maths / multilingue / synthétique) ;
- combien d'époques peut-on répéter avant que la répétition cesse de payer — question
  critique pour nous, car 40B tokens de données réellement excellentes n'existent
  peut-être pas en accès libre ;
- la phase de *recuit* (décroissance du LR sur des données de très haute qualité) est
  souvent le plus gros gain unitaire de tout le pipeline, et elle est bon marché ;
- données synthétiques : gains réels sur MMLU, mais risque d'effondrement et de licence.

→ Track **R06**. Gravité 5 · Levier 5 · Risque 2. **Priorité maximale.**

**Reformulation après calibration du budget.** Avec ~40B tokens et non 300B, le problème
change de nature : il ne s'agit plus de sélectionner le meilleur sous-ensemble d'un
corpus abondant, mais de maximiser l'information transférée par token. Cela promeut la
**distillation depuis des professeurs ouverts** (Qwen3 en Apache-2.0, DeepSeek-R1 en MIT)
du rang de technique de post-entraînement à celui de mécanisme principal
d'apprentissage.

---

## 7. L'optimisation : chaque heure-A100 gaspillée est de la capacité perdue

**Problème.** AdamW + cosine est la référence, et elle est loin d'être optimale.
Sous-problèmes : transfert des hyperparamètres entre échelles (on ne peut pas se
permettre de chercher le LR au bon format), instabilités (pics de perte, divergence),
MFU médiocre sur un seul GPU, et — spécifique à Colab — **des sessions interrompues**
qui exigent une reprise déterministe et un planning de LR compatible avec les coupures.

→ Track **R07**. Gravité 4 · Levier 4 · Risque 2.

---

## 8. La précision : faire tenir le modèle dans la mémoire réelle

**Problème.** Un modèle de 4B en BF16 = 8 Go : hors budget pour un iPhone, à l'étroit
avec un contexte long sur 32 Go. La quantification post-entraînement est donc
obligatoire — mais elle dégrade **beaucoup plus** les petits modèles, et
particulièrement les modèles **sur-entraînés** (beaucoup de tokens par paramètre),
ce qui est précisément notre régime.

Conséquence : la basse précision ne peut pas être un post-traitement. Elle doit
contraindre l'architecture dès le premier jour (normalisation QK, dimensions
compatibles Hadamard, absence d'activations à valeurs aberrantes, formes d'opérateurs
compatibles ANE).

→ Track **R08**. Gravité 4 · Levier 4 · Risque 3.

---

## 9. L'hallucination et l'absence de calibration

**Problème.** Le modèle ne sait pas ce qu'il ne sait pas, et l'entraînement récompense
la devinette : à choix binaire entre « je ne sais pas » (0 point) et une supposition
(espérance > 0), l'optimum statistique est de deviner. À cela s'ajoute une contrainte
physique implacable pour nous :

> La capacité de connaissance mesurée d'un transformeur est de l'ordre de **2 bits par
> paramètre**. À ~1.3B paramètres actifs, Prophet ne peut simplement **pas** contenir
> une fraction significative des faits du monde.

Il est donc *irrationnel* de viser la couverture factuelle. La stratégie correcte est
inverse : faire de l'abstention calibrée et de la récupération d'information une
**capacité native**, et transformer une faiblesse structurelle en argument produit.
Corollaire opérationnel majeur : ne jamais faire de SFT sur des faits que le modèle
de base ignore — cela lui *enseigne* à halluciner.

→ Track **R09**. Gravité 5 · Levier 3 · Risque 3.

---

## 10. Le post-entraînement : là où les scores se gagnent réellement

**Problème.** Le modèle de base n'est pas le produit. L'écart base → instruct est
souvent plus grand que l'écart entre deux architectures. Sous-problèmes : qualité et
licence des jeux SFT, distillation depuis des professeurs ouverts, RL à récompense
vérifiable (GRPO & co.) dont le coût mémoire sur un seul GPU est problématique,
la destruction de la calibration par le RLHF, et le piratage de récompense.

→ Track **R10**. Gravité 5 · Levier 5 · Risque 3.

---

## 11. L'évaluation : le tableau de bord et l'intégrité scientifique

**Problème.** Sans boucle d'évaluation rapide et fiable, tous les autres tracks sont
aveugles. Sous-problèmes :

- **contamination** massive des corpus web par les jeux de test ;
- sensibilité énorme des scores au format de prompt et à la normalisation ;
- la plupart des benchmarks sont au niveau du hasard sous 500M params → inutilisables
  pour des ablations rapides ; il faut un sous-ensemble « signal précoce » validé ;
- absence de métrique standard pour ce qui est *notre* argument : qualité par
  paramètre actif et qualité par gigaoctet.

→ Track **R11**. Gravité 4 · Levier 4 · Risque 2.

---

## 12. La multimodalité sur appareil

**Problème.** Les concurrents directs (Gemma-3-4B, Qwen2.5-VL-3B, Phi-4-multimodal)
sont multimodaux. Mais un encodeur visuel coûte des tokens (729 tokens pour une image
en 384px) et du budget d'entraînement que nous n'avons pas.

**Position par défaut, à confirmer par la recherche :** *later*, mais avec les
**points d'ancrage architecturaux posés dès v1** (embeddings typés par modalité,
identifiants de position 2D-compatibles, option de masque bidirectionnel sur les
segments d'image, points de montage d'adaptateurs par modalité) pour que la vision
soit ajoutable sans ré-entraîner le tronc.

→ Track **R12**. Gravité 3 · Levier 2 · Risque 4.

---

## 12bis. Analyse plus profonde : voir `07_WALLS.md`

Ce document recense les problèmes **attaquables sous notre budget**. Une analyse
séparée, [`07_WALLS.md`](07_WALLS.md), cherche le **mécanisme** de quatre verrous plus
profonds — dont un que la littérature ne nomme pas :

- **A** — le chain-of-thought est un goulot d'information *et* un correcteur d'erreur ;
  et il rend deux services distincts (profondeur sérielle, bloc-notes relisible) dont un
  seul est remplaçable par la récurrence ;
- **B** — la profondeur fixe borne la classe de circuits calculables ;
- **C** — pourquoi la descente de gradient détruit ce qu'elle ne met pas à jour, et le
  troisième étage de mémoire que nous n'avons pas construit ;
- **D** — **le calcul d'inférence ne se cumule pas.** Un modèle qui passe dix minutes sur
  un problème n'en sait rien le lendemain.

## 13. Verrous reconnus mais **hors périmètre v1**

Listés pour l'honnêteté intellectuelle — nous ne les attaquons pas, et nous disons pourquoi.

| Verrou | Pourquoi hors périmètre |
|---|---|
| **Couverture factuelle du monde** | Borne physique (~2 bits/paramètre). Contourné par récupération + abstention (§9), pas résolu. |
| **Généralisation hors distribution / abstraction (ARC-AGI)** | Aucune méthode connue ne le résout ; le tenter consommerait tout le budget sans garantie. |
| **Ancrage sensorimoteur / incarnation** | Nécessite robotique et données d'interaction ; hors budget. |
| **Interprétabilité mécaniste complète** | Champ de recherche entier. On se limite à l'instrumentation (routage, halte, confiance) exploitable. |
| **Alignement de valeurs à grande échelle** | On applique les bonnes pratiques ouvertes ; on ne prétend pas résoudre le problème. |
| **Robustesse adversariale / jailbreaks** | Aucune défense connue ne tient ; on documente la surface d'attaque au lieu de promettre une garantie. |
| **Efficacité énergétique de l'entraînement** | Notre budget est déjà minuscule ; le gain marginal est nul. |

---

## 14. Synthèse : matrice de priorité

| # | Verrou | Track | Grav. | Levier | Risque | Score | Décision v1 |
|---|---|---|---|---|---|---|---|
| 6 | Efficacité en données | R06 | 5 | 5 | 2 | **12.5** | **Cœur** |
| 2 | Attention / cache KV | R02 | 5 | 5 | 2 | **12.5** | **Cœur** |
| 4 | Profondeur de raisonnement | R04 | 5 | 5 | 3 | **8.3** | **Cœur** |
| 10 | Post-entraînement | R10 | 5 | 5 | 3 | **8.3** | **Cœur** |
| 7 | Optimisation | R07 | 4 | 4 | 2 | 8.0 | Cœur |
| 11 | Évaluation | R11 | 4 | 4 | 2 | 8.0 | Infrastructure |
| 1 | Tokenizer | R01 | 4 | 4 | 3 | 5.3 | Conditionnel |
| 8 | Quantification | R08 | 4 | 4 | 3 | 5.3 | Contrainte transverse |
| 3 | Mémoire persistante | R03 | 5 | 4 | 4 | 5.0 | **Pari différenciant** |
| 5 | Sparsité / MoE | R05 | 4 | 5 | 4 | 5.0 | Phase 2 (upcycling) |
| 9 | Hallucination | R09 | 5 | 3 | 3 | 5.0 | Différenciant produit |
| 12 | Multimodalité | R12 | 3 | 2 | 4 | 1.5 | Ancrages seulement |

---

## 15. La thèse du projet, en un paragraphe

> Nous ne pouvons pas gagner en paramètres ni en tokens. Nous pouvons gagner en
> **allocation**. Prophet dépense ses paramètres là où ils comptent (pas dans une table
> d'embedding de 128k entrées), remplace la mémoire O(N) par une mémoire à état borné,
> achète de la profondeur de raisonnement avec du calcul récurrent réglable plutôt
> qu'avec des poids, remplace la couverture factuelle — physiquement hors d'atteinte —
> par de l'abstention calibrée et de la récupération, et accumule à l'usage ce qu'il ne
> pouvait pas apprendre à l'entraînement. Chaque choix ci-dessus est un choix que les
> laboratoires disposant de 200 000 GPU n'ont **aucune raison** de faire — ce qui est
> précisément pourquoi il nous reste de l'espace.
