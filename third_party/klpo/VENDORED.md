# KLPO, copie de référence

Source : https://github.com/yifanzhang-pro/KLPO, commit `ecd92da48fd6ee7ddaf51cad7f6c9c6999343dc9`
(tête de `main` le 2026-09-21), fichiers lus sur raw.githubusercontent.com à ce commit et
comparés octet à octet. Licence Apache-2.0 (`LICENSE`, `NOTICE` conservés). Auteurs :
Yifan Zhang et al., *KL-Regularized Policy Optimization for Critic-Free Agentic
Reinforcement Learning*, rapport technique, septembre 2026.

Ce répertoire est une copie **de référence**, non modifiée : la perte que Prophet utilise
est réécrite dans `prophet/train/klpo.py` et confrontée numériquement à `klpo/loss.py`
par `tests/test_klpo_reference.py`. Le backend d'entraînement GPU du dépôt d'origine
(labs-molt, dépôt séparé) n'est pas copié ; `scripts/check_molt.py`, `scripts/train_molt.py`
et `klpo/molt.py` sont conservés pour référence et `tests/test_molt.py` est sauté sans
`MOLT_SOURCE_PATH`. Le paquet n'est pas installé : il se charge par le chemin.

Exécuter la référence (état au 2026-09-21 : 91 tests passés, 1 sauté) :

```
cd third_party/klpo
PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python examples/train_toy.py --route token --kl-estimator mc --mc-samples 4 --updates 40
```

`ruff` exclut ce répertoire (`pyproject.toml`), `pytest` ne le collecte pas (`testpaths`).

| Fichier | Octets | SHA256 |
|---|---:|---|
| `KLPO.pdf` | 899179 | `b4029a0eb42d7a05fbd9e4297f022a16d5d286dcd5cc4e0a5e4b1ce9d0db5c51` |
| `LICENSE` | 11357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |
| `NOTICE` | 660 | `feb09f271f2a2c3cf1a8aeeb121e3d10d06d8a0d8c86f6dd9061e80b39600781` |
| `README.md` | 8166 | `183d87aa1161700573dac16759f58530b9647517547c7546e544726219a47d65` |
| `assets/token-regression-mc.png` | 281144 | `73bbcddbac3b31018a088e5295eb853aa6fbeacb188c49f5ee99a80f0b940bcf` |
| `citation.bib` | 252 | `349bc5ae50fe8a886e937250f8ca4d5f71dfa27edf6cea7570c38efd2f977d43` |
| `docs/algorithms.md` | 16324 | `eee743ac1b903255253de30383c21fa705a99474953f806f3ab66919fd2d98c7` |
| `docs/training.md` | 14528 | `2ae194d2b10b376589bd42f2cb5428cc23d79772fa1a93078b98f380aab46c97` |
| `examples/train_toy.py` | 8379 | `b56e22b0c059817c1f7e647ddbd554a44ded585415d7f2a0b0476fe8a683623c` |
| `klpo/__init__.py` | 379 | `155e26bbe1ef79901763b5cbe5232393d21ca8382687512d4c6b30b4464d4ead` |
| `klpo/_validation.py` | 4854 | `9d82aa525bb348d5b3d26e705f219a777b18e29a06a6e06d0f0c73d35b056d3e` |
| `klpo/budget.py` | 2819 | `2cf10f976d2cfc95718a8ce1dc057f08abdd41833fb801dd015f6ad1e47f3484` |
| `klpo/loss.py` | 17209 | `7ff24795adec318be56271a80f20bef55dbc8ccce0125be62b6de607a65e9355` |
| `klpo/molt.py` | 4536 | `bf58effa0f94b9190b0123cf3dd7e1f5de1d1312c7cd46f9dcf278db71250f65` |
| `pyproject.toml` | 776 | `604f64b22a5643b0906a8f7125d2068c136aa294e5b7bc940b63a2bfa8dc5c7b` |
| `scripts/check_molt.py` | 1491 | `6716dd73b554b9a69793cb88de09472b459f6654aeb6bc0c589fdf4f30785e80` |
| `scripts/train_molt.py` | 6793 | `61a067ca531c890c57ce72777eb365061d824f04a1a2fb814ed21cd1cab0d0e7` |
| `tests/test_loss.py` | 11503 | `e997d4f04edf6c03a0e41bedeb7302b8bfb35a6e4299a086c1fc43ebc107014f` |
| `tests/test_mc.py` | 7555 | `739a7b5838f1f863e028234893909a500659360a9c823434659744aa6304444c` |
| `tests/test_molt.py` | 7930 | `af55df852519a3bd9c5b55aca36de0e11585847457ed73487f118903c1f57cb0` |
