# 32 — La boucle fermée à 7 M sur CPU : premiers chiffres

**Statut : v1 terminée (trois bras, trois graines retenues), bras KLPO v1 en cours, pilote v2
(grammaire compacte) à venir. Tout chiffre ici vient de `rounds.jsonl` ; rien n'a été
retouché après coup.** Pré-enregistrement et amendements : docs/31. **Date :** 2026-09-21.
Base de code : `claude/codex-results-analysis-5jz95y`. Poids de départ : le 7 M de docs/09
(`prophet-cpu-first-run`, pas 1163), CPU 4 cœurs.

## 0. Ce qu'on sait maintenant, en cinq lignes

1. **La boucle tourne de bout en bout** : tâches inédites → boucle d'agent → vérificateur
   exécutable → promotion → rendu → entraînement avec rejeu → banc, cinq tours par bras,
   trois graines, comptabilité du compute, reprise déterministe (le tour 0 de chaque bras
   redonne le chiffre de la calibration à la décimale près).
2. **Le modèle apprend de ses propres épisodes vérifiés, mais pas de façon fiable** :
   +0,18 et +0,15 de succès sur deux graines, 0,00 sur la troisième ; l'oracle au même
   budget fait +0,27, +0,40, +0,28. H1 échoue, H2 vaut 0,35.
3. **L'oubli ne dépend pas de la source des épisodes** : +0,44 bit/octet en cinq tours pour
   `closed` comme pour `oracle`. H3 passe, mais la recette elle-même oublie beaucoup.
4. **Deux modes d'échec, tous deux vus jeton par jeton** : (a) le span d'appel s'ouvre par
   des espaces que la grammaire tolère et meurt deux jetons plus loin — un désaccord
   entraînement / décodage, corrigé pour v2 ; (b) après un tour, le modèle appelle `done`
   là où il faut `note`, bien formé et faux — une dérive de la recette d'entraînement, non
   corrigée.
5. **`calc` et deux graines sur cinq sont saturées à 100 % dès l'amorce** : la marge de
   progression dépend de la famille et des 100 tâches tirées pour l'amorce.

## 1. Protocole tel qu'exécuté

| Paramètre | Valeur |
|---|---|
| Famille | `lookup` (amendement 1 ; `calc` sature à 100 % dès l'amorce) |
| Amorce | 100 trajectoires parfaites, 200 pas, commune aux bras d'une graine |
| Graines | 0, 2, 3 retenues ; 1 exclue (départ 1,0), amendement 2 |
| Tours | 5 × 30 tâches inédites × 2 tentatives ; 60 pas d'entraînement, rejeu 0,5 |
| Entraînement par tour | Muon 0,01 / AdamW 2e-3, planning WSD neuf à chaque tour, séquences 512 × 8 |
| Génération | température 0,7 ; banc en décodage glouton, 2 × 30 tâches (graines 7 et 11) |
| Langue tenue à l'écart | 200 documents, bits par octet |
| Grammaire | tolérante (v1) : les blancs hors chaîne sont sautés |

Coût : 0,26–0,28 h de CPU par bras `oracle`, 0,31–0,34 h par bras `closed` (génération
comprise : 41–52 k tokens générés par bras), 0,18 h par amorce.

## 2. Résultats v1, trois bras

Succès sur les 60 tâches jamais vues, par tour :

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 0,733 | 0,883 | 0,600 | 0,683 | 0,967 | 0,917 | **+0,183** [+0,067, +0,300] |
| closed | 2 | 0,567 | 0,500 | 0,233 | 0,583 | 0,617 | 0,567 | **+0,000** [−0,083, +0,100] |
| closed | 3 | 0,700 | 0,767 | 0,900 | 0,983 | 0,700 | 0,850 | **+0,150** [+0,033, +0,267] |
| oracle | 0 | 0,733 | 0,917 | 0,817 | 1,000 | 1,000 | 1,000 | +0,267 [+0,167, +0,383] |
| oracle | 2 | 0,567 | 0,617 | 0,717 | 1,000 | 0,983 | 0,967 | +0,400 [+0,267, +0,533] |
| oracle | 3 | 0,700 | 0,967 | 0,967 | 0,983 | 1,000 | 0,983 | +0,283 [+0,167, +0,400] |
| frozen | 0, 2, 3 | = t0 | = t0 | | | | | 0 (contrôle exact) |

Part de sorties malformées au banc (moyenne des deux bancs) :

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 13 % | 4 % | 0 % | 12 % | 1 % | 1 % |
| closed | 2 | 0 % | 2 % | **45 %** | 6 % | 0 % | 0 % |
| closed | 3 | 4 % | 4 % | 7 % | 0 % | 4 % | 0 % |
| oracle | 0 | 13 % | 1 % | 0 % | 0 % | 0 % | 0 % |
| oracle | 2 | 0 % | 6 % | 3 % | 1 % | 8 % | 4 % |
| oracle | 3 | 4 % | 3 % | 1 % | 1 % | 0 % | 1 % |

Bits par octet tenus à l'écart :

| Bras | Graine | t0 | t5 | Δ |
|---|---:|---:|---:|---:|
| closed | 0 | 2,104 | 2,527 | +0,422 |
| closed | 2 | 2,090 | 2,551 | +0,461 |
| closed | 3 | 2,124 | 2,572 | +0,447 |
| oracle | 0 | 2,104 | 2,546 | +0,442 |
| oracle | 2 | 2,090 | 2,514 | +0,423 |
| oracle | 3 | 2,124 | 2,519 | +0,395 |

Épisodes du bras `closed` : résolus par tour (sur 30) 22/26/22/16/30 (graine 0),
14/19/12/18/17 (graine 2), 20/25/30/29/24 (graine 3) ; promus au total 116, 80, 128.
Trajectoires promues de forme parfaite (`read_file > note > done`) : 97 %, 79 %, 95 %.
Les autres contiennent un pas malformé (retiré au rendu) ou une note en double.

## 3. Verdicts, critères de docs/31 §1

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H1** auto-amélioration | **échoue** | gain > 0 sur 2 graines sur 3 ; sur la graine 2, 0,000 [−0,083, +0,100]. Banc cumulé : 180 tâches, 28 gagnées, 8 perdues |
| **H2** rendement | rapporté | gain(closed) / gain(oracle) = **0,35** sur la moyenne des graines (0,111 / 0,317) |
| **H3** oubli | **passe** | Δ BPB closed +0,443 contre oracle +0,420, écart +0,023 ≤ 0,05 |
| **H4** compute | **échoue** | gain par heure du dernier tour : −0,85, −0,77, +2,25 |
| **H5** KLPO | en cours | bras `closed-klpo` v1 lancé sur les graines 0, 2, 3 |

H3 passe pour une mauvaise raison : les deux bras oublient autant parce que la recette
(60 pas à taux plein, planning neuf à chaque tour, rejeu 0,5) domine. +0,44 bit/octet en
cinq tours, c'est 21 % de perplexité en plus sur la langue tenue à l'écart. À l'échelle
A100 la recette par tour doit changer avant le rejeu (§7).

## 4. Les deux modes d'échec, jeton par jeton

Le banc est glouton et déterministe (`frozen` le prouve : deux tours, même chiffre). Les
chutes viennent donc de l'entraînement du tour. Diagnostic sur les checkpoints, mêmes
tâches que le banc.

**(a) La forme.** Graine 2, tour 2 (0,500 → 0,233, 45 % malformées). Le modèle amorce ouvre
chaque appel par `{"`. Le modèle du tour 2 émet `<|call|>`, ` `, ` `, puis plus rien de
viable parmi ses 64 premiers candidats : le span meurt, le pas est « malformé », et il
recommence au pas suivant jusqu'à épuiser les quatre pas. Le rendu écrit les appels en JSON
compact, sans blanc ; la grammaire de décodage, elle, sautait les blancs comme un lecteur
JSON. Un modèle qui dérive vers l'indentation (rejeu de code, peu de lignes) trouvait donc
une porte ouverte que l'entraînement n'avait jamais montrée. Correctif : grammaire compacte
(`ActionGrammar(compact=True)`, amendement 3, tests d'accord rendu / grammaire). Décodage
seulement ; testé en v2.

**(b) Le choix d'action.** Graine 3, tour 4 (0,983 → 0,700, 4 % malformées seulement). Sur
les tâches ratées, le modèle lit le fichier puis appelle **`done`** avec la bonne valeur en
`summary`, au lieu de `note` ; le vérificateur refuse (`verification failed`), et le modèle
répète `done` jusqu'au bout du budget. Les trajectoires promues n'enseignent pas ce
raccourci (95 % sont `read_file > note > done`, aucune ne contient un `done` refusé) ; c'est
la tête de sélection d'action qui bascule après 60 pas à taux plein sur 24–30 lignes.
L'oracle connaît le même creux (graine 0, tour 2 : 0,917 → 0,817, bien formé) et en sort
parce que ses 30 lignes par tour sont toutes parfaites. Ce mode n'est pas corrigé par la
grammaire ; il relève de la recette (taux, pas par tour, rejeu) et, par construction, de la
régularisation KL vers l'échantillonneur du bras KLPO (H5).

**(c) Le corps de `done`, dès l'amorce.** Les 13–17 % de sorties malformées du modèle
amorce (graine 0, tour 0, identiques sous les deux grammaires) ont une seule forme :
`{"name":"done","args":{"` puis `{` — après `"args":{`, le modèle suit le motif majoritaire
des lignes d'entraînement (`{"path"`, `{"text"`) au lieu du `}` que `done` demande, et le
span meurt sur le jeton suivant. La bonne valeur est déjà dans les notes ; il manque un pas
de budget (quatre pas par épisode) pour réémettre `done`. Ce mode ne dépend pas de la
grammaire ; il relève de la taille du modèle et du budget de pas, et il coûte à tous les
bras de la même façon (le banc du tour 0 est commun).

## 5. Ce que l'amorce décide

| Famille | Amorce | Départ (banc 7 / 11) | Suite |
|---|---|---|---|
| `calc` | 100 / 200 | 1,00 / 1,00 | saturé, H1 impossible : run arrêté au tour 0 |
| `calc` | 40 / 80 | 0,00 / 0,00 (0 % malformées : bien formé, faux) | seuil entre 40/80 et 100/200 |
| `lookup` | 100 / 200, graine 1 | 1,00 / 1,00 | exclue |
| `lookup` | 100 / 200, graines 0, 2, 3 | 0,67/0,80 · 0,50/0,63 · 0,63/0,77 | retenues |

Le mur d'amorçage de docs/31 §2 est réel et étroit : `calc` passe de 0 à 100 % sans
intermédiaire, `lookup` dépend des 100 tâches tirées. La marge de la boucle (ce qu'elle a
à apprendre) est une variable de l'expérience, pas une constante.

## 6. Bras KLPO v1 : effondrement en deux tours, arrêté

Graine 0, mêmes amorce, tâches et bancs que `closed` ; à chaque tour, fine-tuning par
rejet identique à `closed` **puis** 60 pas KLPO (β = 0,1, lr 5e-4, M = 8, génération à
température 1,0) sur tous les épisodes du tour avec leur récompense 0/1 :

| Tour | Succès | Malformées | BPB | Épisodes récompensés | Perte KLPO (premier → dernier pas) | log p − log q moyen |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,733 | 17 % / 10 % | 2,104 | — | — | — |
| 1 | 0,517 | 26 % / 37 % | 2,176 | 24 / 41 | −1,4 → −13,1 | −0,92 nat/token |
| 2 | **0,000** | 53 % / 51 % | 2,283 | 11 / 51 | +5,6 → −0,2 | −1,40 nat/token |

Le run a été arrêté après ce tour. **H5 échoue** par arithmétique : le gain de la graine 0
est au plus −0,733 (le succès ne peut que remonter de 0 à 0,733 en trois tours au mieux) ;
même avec les gains maximaux possibles sur les graines 2 et 3 (+0,433 et +0,300), la
moyenne serait 0,000 < +0,111, le gain moyen de `closed`. Les graines 2 et 3 n'ont pas été
lancées en v1 ; elles le sont en v2 (docs/31, amendement 4).

Ce que les chiffres disent du mécanisme : en 60 pas sur les **mêmes** enregistrements, la
politique s'éloigne de l'échantillonneur de 0,9 puis 1,4 nat par token en moyenne. Le
rapport KLPO nomme cette réutilisation « substitut empirique » et demande des tirages
frais de l'échantillonneur historique à chaque pas ; nous ne les avons pas (A5 §4). À
β = 0,1, le terme de KL ne retient rien de cette dérive, et la part malformée double à
chaque tour : le mode d'échec (a) de §4, amplifié par l'échantillonnage à température 1,0.
La comparaison à budget égal n'est donc pas « KLPO contre rejet » mais « rejet + 60 pas de
gradient de politique mal régularisé contre rejet seul ». Le bras v2 garde ces
hyperparamètres (pré-enregistrés) et la grammaire compacte ; un bras à β plus grand et à
moins de pas est la suite naturelle si v2 confirme l'effondrement.

## 7. Pilote v2 (grammaire compacte) et suite

*En cours ; les lignes ci-dessous sont écrites au fil des tours.* Pré-enregistré (docs/31
amendement 3, H6) : quatre bras, graines 0, 2, 3, grammaire compacte, rien d'autre de changé.

**Réplication.** Tant qu'aucun blanc n'apparaît dans un span d'appel, v2 rejoue v1 au bit
près : bras `oracle` des graines 0 et 2, bras `closed` de la graine 0 (mêmes tokens générés
par tour, 8 815 / 7 705 / 8 708 / 10 219 / 6 557, mêmes BPB à la quatrième décimale, mêmes
succès). Le pipeline est déterministe de bout en bout.

**Le point de contrôle de H6, graine 2, tour 2 : même modèle, deux grammaires.** Les tours
0–2 de v2 reprennent exactement l'entraînement de v1 (mêmes 9 754 tokens générés au tour 2,
même BPB 2,2283), donc les poids évalués au tour 2 sont les mêmes ; seule la grammaire du
banc diffère :

| Grammaire du banc | Succès | Malformées (banc 7 / 11) |
|---|---:|---:|
| tolérante (v1) | 0,233 | 49 % / 42 % |
| compacte (v2) | **0,550** | 8 % / 8 % |

19 tâches sur 60 récupérées sans toucher aux poids : le mode (a) était bien un défaut de
décodage. Les 8 % restants sont le mode (c) (corps de `done`). À partir du tour 3 les deux
pilotes divergent (la génération elle-même passe par la grammaire compacte).

Pour l'échelle A100 (docs/31 §3), trois choses sont acquises dès maintenant : la marge de
départ se calibre avant de lancer (tour 0 seul, par graine et par famille) ; la recette par
tour doit être mesurée sur l'oubli avant tout (un planning neuf à taux plein par tour est
la cause du +0,44) ; le banc doit rapporter la part malformée à chaque tour, sans quoi les
deux modes d'échec se confondent.
