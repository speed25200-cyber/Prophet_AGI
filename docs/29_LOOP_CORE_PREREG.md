# 29 — Pré-enregistrement : ce qu'on boucle sur un téléphone

**Statut : v0, rédigé avant tout entraînement. Aucun run lancé. À figer (hachage des
configs et des scripts) avant le premier pas, après accord sur la question.**
**Date :** 2026-09-20
**Base de code :** branche `claude/codex-results-analysis-5jz95y`, issue de la reprise
Codex `0b36ba2` (902 tests CPU passent, 22 sautés sans CUDA).

## 0. Ce qui existe déjà, et ce qui n'existe pas

Revue faite le 2026-09-20 par recherche web (arXiv est inaccessible en direct depuis
l'environnement de rédaction ; chaque ligne est marquée `[S]` : résumé de recherche,
à confirmer sur le texte intégral avant citation dans un article).

| Travail | Échelle | Ce qui est dans la boucle | Profondeur à l'entraînement | Composition | Quantification | Appareils |
|---|---|---|---|---|---|---|
| Huginn, 2502.05171 `[S]` | 3,5B, 800B tokens | attention | log-normale-Poisson, gradient tronqué | tâches de raisonnement | non | non |
| MoR, 2507.10524 `[S]` | 135M–1,7B | attention | routage par token | non | non | non |
| Ouro, 2510.25741 `[S]` | 1,4B / 2,6B, 7,7T tokens | attention | allocation apprise, régularisée par l'entropie | benchmarks | non | non |
| LoopFormer, 2602.11451 `[S]` | à vérifier | attention | trajectoires de longueur variable, cohérence par raccourci | benchmarks | non | non |
| Hyperloop, 2604.21254 `[S]` | à vérifier | attention + hyper-connexions | à vérifier | non | INT4 GPTQ, une profondeur | motivé par l'embarqué, mesures à vérifier |
| LoopQ, 2605.16343 `[S]` | Ouro, LoopFormer, Parcae | attention | modèles existants | non | PTQ W4A4 ; l'erreur s'accumule à chaque boucle | non |
| Fixed-Point Reasoners, 2606.18206 `[S]` | petits modèles | attention | halte par point fixe | Sudoku, labyrinthe, ARC-AGI | non | non |
| Readout blind spot, 2606.24898 `[S]` | à vérifier | attention | par boucle | non | non | non |
| DeepLoop, 2607.13491 ; Adaptive Depth, 2607.20519 `[S]` | Ouro | attention | mise à l'échelle de la profondeur ; lecture des trajectoires | synthétiques | non | non |
| recurrent-depth-mamba (dépôt GitHub, sans article) `[S]` | jouet | SSM Mamba | fixe | non | non | non |
| **Ce programme** | 375M, 300M tokens | **GDN à état borné contre attention**, sous contrôle | uniforme 2..6, gradient complet | tables à 1–4 sauts, dans le mélange | int8 / int4 par cœur et par profondeur | RTX 5090, Mac Studio, iPhone |

Conséquences pour la nouveauté :

- Le « cadran de profondeur » entraîné (un jeu de poids, plusieurs budgets) est publié :
  LoopFormer. Il ne peut pas être la contribution.
- Le bouclage comme architecture économe en paramètres pour l'embarqué est publié :
  Hyperloop, avec un test INT4.
- L'accumulation d'erreur de quantification à travers les boucles est mesurée : LoopQ,
  y compris sur un modèle à profondeur élastique.
- **Aucun de ces travaux ne met autre chose que de l'attention dans la boucle.** Or c'est
  la décision D1 de Prophet : un cœur à état borné (gated delta), dont le cache
  n'augmente pas avec la profondeur ni avec le contexte, contre un cœur à attention dont le
  cache est multiplié par *k*. C'est la seule question où les contraintes du projet
  (téléphone, mémoire) sont un avantage plutôt qu'un handicap, et elle n'a pas de
  comparaison contrôlée dans la littérature recensée.
- Le dépôt GitHub « recurrent-depth-mamba » est le voisin le plus proche : échelle jouet,
  sans bras attention, sans quantification, sans article.

Si, en lisant les textes intégraux, l'un de ces travaux contient déjà la comparaison
cœur borné contre cœur attention sous contrôle, ce programme s'arrête et le rapporte.

## 1. La question

À blocs exécutés égaux, paramètres égaux, données et graines égales, **que coûte et que
rapporte un cœur bouclé à état borné par rapport à un cœur bouclé à attention**, sur
cinq axes : mémoire d'inférence, perte de langage, courbe de profondeur, composition,
quantification. Le témoin non partagé (pile de 20 blocs) sert de référence de qualité.

Ce que ce programme ne prétend pas : un modèle compétitif, un raisonnement, un assistant,
une mémoire persistante, un seuil d'échelle universel.

## 2. Hypothèses et critères, fixés avant le premier pas

Toutes les comparaisons de perte utilisent un bootstrap apparié par document
(10 000 tirages, PCG64 graine 0, quantiles linéaires), et rapportent la dispersion entre
les trois graines. Un critère se juge sur la moyenne des graines, avec l'écart entre
graines affiché à côté.

| Hypothèse | Énoncé mesurable | Critère de passage |
|---|---|---|
| **H1 mémoire** | Le cache d'inférence du cœur GDN est constant en *k* ; celui du cœur attention croît en *k* × contexte. | Mesure des octets réels de cache à contexte 2 048 / 8 192 / 32 768 et *k* = 2, 4, 6. Pas de critère : c'est une quantification du prix. |
| **H2 qualité à FLOPs égaux** | Le cœur borné ne coûte pas plus de 2 % de bits par octet face au cœur attention, à *k* = 4. | BPB_gdn(4) / BPB_attn(4) ≤ 1,02 en moyenne sur 3 graines. |
| **H3 courbe de profondeur** | Sous entraînement élastique, le cœur attention gagne davantage aux boucles supplémentaires que le cœur borné. | Interaction : [BPB(6)/BPB(4)]_attn < [BPB(6)/BPB(4)]_gdn sur 3 graines sur 3. |
| **H4 composition** | À 3 et 4 sauts, la précision croît avec *k* pour le cœur attention et non pour le cœur borné. | Précision(6) − précision(2) > 0 avec intervalle au-dessus de zéro sur 3 graines pour l'attention ; comparaison de cet écart entre cœurs. |
| **H5 quantification** | L'erreur int4 par boucle s'accumule moins vite dans le cœur borné. | Δ(k) = BPB_int4(k) − BPB_fp32(k) ; la pente de Δ en *k* est plus faible pour GDN sur 3 graines sur 3. |

Un échec de H2, H3, H4 ou H5 est un résultat, publié tel quel. Le programme ne se
« répare » pas en changeant un critère après coup.

## 3. Bras

| Bras | Paramètres résidents | Cœur bouclé | Profondeur à l'entraînement | Graines |
|---|---:|---|---|---|
| `lc_gdn` | 374,7M | 4 blocs GDN, sans attention | uniforme 2..6, gradient sur toutes les passes | 0, 1, 2 |
| `lc_attn` | 376,2M (FFN élargi à 4,93× pour égaliser, écart +0,41 %) | 4 blocs attention (l'ablation A-KV, avertissement D1 accepté explicitement) | uniforme 2..6, gradient sur toutes les passes | 0, 1, 2 |
| `lc_plain` | 920,7M | pile de 16 blocs, une passe | — | 0 |

Commun aux trois : prélude 2 blocs et coda 2 blocs à attention, largeur 1 792, vocabulaire
32 768 (tokenizer du pilote, inchangé), 20 blocs exécutés par token à *k* = 4, halte,
têtes auxiliaires et adaptateur d'entrée désactivés, réinjection additive de l'entrée,
état initial aléatoire à l'entraînement et nul à l'inférence, pas de normalisation
inter-boucle (comme les runs R04 existants, pour rester comparable ; voir §7).

Les configurations sont générées par `scripts/build_configs.py`, passent
`ProphetConfig.validate()`, et leur seul avertissement de conception attendu est celui de
l'attention dans le cœur du bras `lc_attn`. `python -m prophet.budget` et
`scripts/gpu_check.py` (mémoire de pointe à *k* = 6, lot 8, séquence 2 048) sont exécutés
avant le premier run ; leurs sorties sont jointes au pré-enregistrement figé.

## 4. Données

- FineWeb-Edu `sample-10BT`, révision épinglée `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`,
  lue en flux depuis le Hub dans Colab, mêmes filtres que le pilote (score ≥ 3, 400 à
  50 000 octets, dédoublonnage normalisé, séparation par URL).
- Objectif : **au moins 320 millions de tokens d'entraînement**, soit une seule passe pour
  chaque run ; validation tenue à l'écart : 2 000 documents complets.
- Exclusions : les documents des pilotes précédents (train et validation), tout span de
  13 mots partagé avec la validation, avec ARC-Easy / ARC-Challenge / GSM8K / MMLU /
  HumanEval / LAMBADA, et avec les tâches synthétiques de test.
- **Tâches de composition, 3 % des tokens d'entraînement**, générées par un script
  déterministe : consultation de tables à 1, 2, 3 et 4 sauts (`a → b`, `b → c`, …,
  question « départ a, sauts 3, réponse : »), et addition à 2 à 6 chiffres. Réponse en un
  seul token. Tables de test jamais vues à l'entraînement, 2 000 exemples par nombre de
  sauts.

Le corpus reste hors Git ; son manifeste (révision, filtres, comptes, empreintes SHA256)
est commité.

## 5. Budget

| Poste | Valeur |
|---|---:|
| Tokens par run | 300 M (18 310 pas de 8 × 2 048) |
| Coût mesuré par pas (R04, A100 40 Go) | ≈ 1,97 s boucle, ≈ 2,16 s pile |
| Heures A100 par run | ≈ 10 (boucle), ≈ 11 (pile), hors évaluation et sauvegardes |
| Total, 7 runs | **≈ 70 heures A100** |
| Évaluations, quantification, diagnostics | ≈ 6 heures |

Planning WSD, Muon 0,01 / AdamW 3e-4, décroissance de poids 0,1, chauffe 100 pas,
décroissance 18 %, écrêtage 1,0, perte par blocs de 512 tokens, checkpoint tous les 512
pas sur Drive, reprise exacte vérifiée sur huit pas continus contre un plus sept avant le
premier segment long. Ces valeurs sont celles des runs R04 existants ; elles ne sont pas
réglées sur ce programme.

Si le budget ne peut pas être réuni : réduire les tokens par run à 200 M, jamais le
nombre de graines.

## 6. Mesures sur chaque checkpoint final

1. Bits par octet sur la validation à *k* = 1, 2, 3, 4, 5, 6, 8 (fp32, TF32 désactivé).
2. Précision de composition par nombre de sauts et par *k*.
3. Octets de cache après préremplissage à contexte 2 048 / 8 192 / 32 768, *k* = 2, 4, 6,
   lot 1, comptés sur les tenseurs réels du cache.
4. Quantification poids seuls, arrondi au plus proche, par canal, int8 et int4, sans
   dépendance externe ; BPB à chaque *k*.
5. Appareils : tokens/s et mémoire de pointe à *k* = 2, 4, 6, lot 1, amorce 512 tokens,
   128 tokens générés, sur RTX 5090, Mac Studio (MPS ou MLX) et iPhone via export CoreML
   du bras `lc_gdn` si l'export aboutit ; sinon, documenter ce qui bloque.
6. ARC-Easy à *k* = 4 comme contrôle de cohérence, jamais comme critère.

Courbes intermédiaires à 100 M et 200 M tokens pour les mesures 1 et 2.

## 7. Risques connus, et ce qu'on en fait

- **Norme des états qui croît avec les boucles** (docs/24 : RMS de 1 000 à 7 000 sur
  huit boucles). Le mécanisme est décrit dans 2606.24898 `[S]` : sans normalisation entre
  boucles, la perte par boucle ne contrôle pas l'échelle radiale. Ce programme garde
  l'architecture R04 telle quelle pour rester comparable ; une normalisation inter-boucle
  est la première ablation de suite, pas une modification en cours de route.
- **Mémoire du bras attention à *k* = 6** : 24 applications d'attention par token à
  séquence 2 048. Le préflight `gpu_check` à *k* = 6 décide ; si ça ne tient pas à lot 8,
  les trois bras passent à lot 4 avec accumulation de gradient sur 2, avant tout run.
- **Une seule passe sur le corpus** : aucun run ne dépasse 1,0 époque.
- **Contamination** : la décontamination lexicale ne couvre ni les paraphrases ni les
  connaissances déjà présentes dans le web. Aucun score de benchmark n'est un critère.

## 8. Interdits

- Aucun écran court sur poids réchauffés ; tous les runs partent de zéro.
- Aucune conversion de donneur, aucune mémoire persistante, aucun nouveau composant.
- Aucun changement de protocole après le premier pas ; un changement crée un nouveau
  contrat de run, l'ancien est conservé.
- Aucune sélection de recette sur un score intermédiaire.
- Vérification : les portes existantes (`gpu_check`, `audit_r04_checkpoint`,
  `audit_r04_restart`) et un seul script de vérification locale par run. Pas de nouvelle
  infrastructure de hachage. Au plus un cinquième du temps de session.

## 9. Livrables

- Ce document, figé avec les empreintes des configs, du générateur de tâches et du
  lanceur, avant le premier pas.
- `docs/30_LOOP_CORE_RESULTS.md` : tous les runs, toutes les graines, tous les échecs.
- Preuves JSON sous `docs/experiments/2026-*-loop-core-*/`.
- `docs/paper/loop_core.md` : brouillon d'article de 6 à 8 pages avec le tableau de
  positionnement ci-dessus, complété sur textes intégraux.
- Un notebook Colab épinglé qui exécute une file de commandes et renvoie les preuves
  vers GitHub (voir `notebooks/loop_core_runner.ipynb`).

## 10. Exécution : ce qui est construit, et la procédure

Tout ce qui ne demande pas de GPU est écrit et testé sur CPU (branche
`claude/codex-results-analysis-5jz95y`) :

| Pièce | Fichier | Test |
|---|---|---|
| Les trois bras, budgétés, avertissement D1 accepté explicitement | `scripts/build_loop_core_configs.py`, `configs/loop_core/` | `tests/test_loop_core_configs.py` |
| Tâches de composition et leur notation | `prophet/data/composition.py`, `prophet/eval/composition.py` | `tests/test_composition.py` |
| Corpus en une passe, exclusions, manifeste | `scripts/prepare_loop_core_corpus.py`, `scripts/stage_corpus.py` | `tests/test_loop_core_corpus.py`, `tests/test_loop_core_queue.py` |
| Lanceur de session : protocole figé, jalons exacts, historique des profondeurs, instantané Drive, préflight mémoire | `scripts/run_loop_core.py` | `tests/test_run_loop_core.py` |
| Mesures finales : BPB par profondeur, composition, octets de cache, int8/int4 | `scripts/eval_loop_core.py`, `prophet/quant/rtn.py` | `tests/test_eval_loop_core.py` |
| Transport Colab → GitHub : file de commandes reprenable, preuves sans poids | `scripts/colab_queue.py`, `queue/loop_core/programme.json`, `notebooks/loop_core_runner.ipynb` | `tests/test_colab_queue.py` |

La file `programme.json` enchaîne, dans l'ordre du §11 : porte GPU, cache des benchmarks,
construction du corpus et publication sur Drive, préflights mémoire à *k* = 6 des trois
bras, puis pour chaque run l'entraînement par sessions bornées jusqu'au marqueur
`RUN_COMPLETE` et ses mesures finales. Une commande qui échoue arrête la file ; une
session Colab qui expire reprend là où elle s'est arrêtée.

### Côté utilisateur

1. Créer un jeton GitHub à grain fin, portée *Contents : read and write* sur
   `speed25200-cyber/Prophet_AGI`, et l'enregistrer dans les secrets Colab sous
   `GITHUB_TOKEN` (accès notebook activé). Sans jeton, les preuves restent sur Drive
   sous `loop-core/queue-state/export/`.
2. Vérifier que le corpus pilote audité (docs/13) est bien sous
   `MyDrive/Prophet_AGI/R04/corpus-v1/` avec son `tokenizer.json` ; sinon corriger
   `PILOT` dans la première cellule.
3. Ouvrir `notebooks/loop_core_runner.ipynb` sur un A100, exécuter les cellules 1 et 2,
   surveiller avec la cellule 3. À chaque nouvelle session, réexécuter 1 et 2.
4. Relever la branche `results/loop-core` : chaque commande terminée y pousse ses
   rapports sous `results/loop-core-programme/`.

### Côté analyse

Les rapports de `results/loop-core` sont relus à chaque étape ; les critères du §2 sont
calculés par un script d'analyse à écrire une fois les trois graines mesurées, jamais
avant, pour que le calcul ne s'ajuste pas aux premiers résultats.

## 11. Ordre d'exécution

0. Ce document ; accord sur la question.
1. Configs des trois bras, budget, avertissements de conception ; générateur de tâches
   de composition ; script de corpus ; lanceur et file de commandes ; tests CPU.
2. Sur A100 : portes GPU, préflight mémoire à *k* = 6, reprise exacte à taille réelle.
3. Graine 0 de `lc_gdn` et `lc_attn`, évaluation complète.
4. Graines 1 et 2.
5. `lc_plain`.
6. Quantification, cache, appareils.
7. Rédaction.
