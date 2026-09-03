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
docs/           Spécifications et rapports de recherche (R01–R12)
prophet/
  modeling/     Modules d'architecture
  data/         Pipeline de données et mélanges
  train/        Boucles d'entraînement, optimiseurs, schedules
  eval/         Harnais d'évaluation
  kernels/      Noyaux Triton/CUDA
configs/        Configurations d'expériences (YAML)
scripts/        Scripts exécutables (Colab compris)
tests/          Tests unitaires et de forme
```

## Conventions

- Python ≥ 3.11, PyTorch ≥ 2.5, typage explicite sur les interfaces publiques.
- Formatage `ruff format`, lint `ruff check`.
- Les configurations sont des dataclasses sérialisables, pas des dictionnaires libres.
- Les noms de branches de travail : `claude/<sujet>`.
