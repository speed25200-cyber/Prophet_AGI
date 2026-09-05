# Prophet — notes pour les agents travaillant sur ce dépôt

## Contexte

Prophet est un projet de recherche multi-semaines : concevoir et entraîner une
architecture de LLM de nouvelle génération sous des contraintes de compute extrêmes.
Lire `docs/00_PROBLEM_LANDSCAPE.md` avant toute contribution.

## Contraintes non négociables

- **Entraînement** : un seul A100 80GB (Google Colab, sessions interruptibles).
  Toute proposition exigeant du multi-GPU ou du multi-nœud est hors périmètre.
  A100 = Ampere sm_80 : **pas de FP8 matériel**, pas de FlashAttention-3.
- **Inférence** : doit tourner sur 1× RTX 5090 (32 Go), un Mac Studio, et — en
  variante réduite — un iPhone 17 Pro (~8 Go, budget applicatif réel 3–5 Go).
- **Reprise** : tout entraînement doit être interruptible et reprenable de façon
  déterministe (dataloader inclus).

## Règles d'ingénierie

1. **Rien sans chiffres.** Toute affirmation de performance s'accompagne de son
   arithmétique (FLOPs, octets, tok/s) ou d'une citation.
2. **Ablation avant adoption.** Aucun composant architectural n'entre dans la version
   principale sans une ablation à 50–500M paramètres qui le valide.
3. **Réversibilité.** Chaque module est activable/désactivable par configuration.
   Une idée non validée ne doit jamais être un couplage dur.
4. **Pas de données dans git.** Corpus, checkpoints et poids restent hors dépôt.
5. **Hygiène de contamination.** Toute source de données passe par la décontamination
   avant d'entrer dans un mélange d'entraînement.
6. **Honnêteté des résultats.** On rapporte les échecs. Un score non reproduit est
   marqué comme tel.

## Structure

```
docs/           Spécifications, registre de décisions, rapports de recherche
                (R01–R12 verrous, W1–W4 murs, A1–A4 revue et agentique)
prophet/
  config.py     Schéma de configuration — tout pari est un interrupteur explicite,
                validate() refuse l'impossible, design_warnings() refuse l'incohérent
  budget.py     Paramètres, mémoire, débit par appareil, alertes d'allocation
  scaling.py    Points de fonctionnement sous budget d'heures-A100
  plan.py       Allocation du compute entre les tracks (ordre strict, sans remplissage)
  modeling/     Couches (attention GQA/SWA/NoPE, delta gated), MoE, modèle à
                profondeur récurrente avec halte apprise
  data/         Tokenizer Prophet-Tok v1, mélanges, décontamination, streaming reprenable
  train/        Muon + AdamW, planning WSD, checkpointing atomique, boucle, pertes
  eval/         Métriques (BPB) et harnais à trois niveaux
  memory/       Registre à clés-produit (écriture en forme close), état de session,
                consolidation de contexte et de profondeur
  convert/      Conversion d'un donneur ouvert vers l'architecture Prophet
  analysis/     Mesure de la bande passante des canaux de raisonnement
  kernels/      Réservé aux noyaux Triton/CUDA — vide tant qu'aucun GPU n'a servi
configs/        Configurations générées par scripts/build_configs.py (jamais à la main)
scripts/        Scripts exécutables : entraînement, conversion, vérification, sondes
tests/          ~300 tests ; les plus importants sont des tests d'équivalence
```

## Ce que ce dépôt a appris à ses dépens

Six défauts **silencieux** ont été trouvés en construisant — chacun s'entraînait
normalement et aurait produit un modèle fluide et faux :

| Défaut | Comment il a été trouvé |
|---|---|
| Init aléatoire de l'état récurrent fuyant à l'inférence | test d'équivalence préremplissage/décodage |
| Embeddings liés écrasés au chargement après copie du state dict | test sur dictionnaire copié en profondeur |
| `β = sigmoid` bornant l'écriture delta à (0,1) : parité hors d'atteinte | track W2, vérifié par sonde : hasard → 0.996 |
| Config livrée avec l'attention **dans** la boucle | track W2 ; désormais `design_warnings()` |
| Sondes de halte écrivant *k* fois dans le même cache | test d'équivalence avec halte |
| `nope_layers` réglé, vérifié, documenté — et ignoré par le modèle | revue A1 : grep des champs jamais lus |

**Règle qui en découle :** un champ de configuration que rien ne lit est un bug, pas une
réserve. Toute nouvelle option doit être lue par le code qui l'honore *et* couverte par
un test comportemental, dans le même commit.

## Avant de proposer un changement d'architecture

Passer par les outils, pas par l'intuition :

```bash
python -m prophet.budget <config.json>   # est-ce que ça tient en mémoire ?
python scripts/design_search.py          # qu'est-ce qui satisfait toutes les contraintes ?
python -m prophet.plan                   # d'où vient le compute ?
```

Ces outils ont déjà corrigé des erreurs de conception avant qu'elles ne coûtent quoi que
ce soit. Une proposition sans passage par eux n'est pas recevable. Un changement de
modèle passe aussi par la suite de tests d'équivalence (`tests/test_modeling.py`) : le
décodage incrémental doit rester identique à une passe complète, avec cache, halte et
mémoire activés.

## Conventions

- Python ≥ 3.11, PyTorch ≥ 2.5, typage explicite sur les interfaces publiques.
- Formatage `ruff format`, lint `ruff check`.
- Les configurations sont des dataclasses sérialisables, pas des dictionnaires libres.
- Les noms de branches de travail : `claude/<sujet>`.
