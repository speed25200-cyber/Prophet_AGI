# 32 — La boucle fermée à 7 M sur CPU : premiers chiffres

**Statut : quatre pilotes terminés (v1 tolérant, v2 grammaire compacte, v3 taux ÷ 4, v4
promotion canonique), trois graines retenues, bras KLPO à chaque fois. Tout chiffre ici vient de `rounds.jsonl` ; rien n'a été
retouché après coup.** Pré-enregistrement et amendements : docs/31. **Date :** 2026-09-21.
Base de code : `claude/codex-results-analysis-5jz95y`. Poids de départ : le 7 M de docs/09
(`prophet-cpu-first-run`, pas 1163), CPU 4 cœurs.

## 0. Ce qu'on sait maintenant, en sept lignes

1. **La boucle tourne de bout en bout** et se rejoue au bit près : tâches inédites →
   boucle d'agent → vérificateur exécutable → promotion → rendu → entraînement avec rejeu →
   banc, cinq tours par bras, trois graines, comptabilité du compute, reprise déterministe.
2. **Le modèle apprend de ses propres épisodes vérifiés**, et de mieux en mieux à mesure
   que les défauts tombent : gain moyen du bras fermé +0,111 (v1), +0,122 (v2), +0,206
   (v3), **+0,256** (v4, intervalles excluant zéro sur les trois graines) ; l'oracle au
   même budget fait +0,278 : rendement 0,35 → 0,92.
3. **L'oubli venait de la recette, pas de la source des épisodes** : +0,44 bit/octet en
   cinq tours pour tous les bras à taux plein, +0,07 au taux divisé par quatre, sans perte
   de gain.
4. **Trois défauts, tous vus jeton par jeton** : (a) la grammaire de décodage admettait des
   blancs que le rendu n'écrit jamais (0,233 → 0,550 à poids égaux) ; (b) un planning
   neuf à taux plein par tour faisait basculer le choix d'action ; (d) deux trajectoires
   vérifiées mais bâclées sur quatorze suffisaient à apprendre une boucle sans fin sur
   `note` (0,017 → 0,550). Reste (c), le corps de `done` du modèle amorce.
5. **`calc` et deux graines sur cinq saturent à 100 % dès l'amorce** ; la marge de
   progression dépend de la famille et des 100 tâches tirées. **KLPO** n'a tenu cinq tours
   dans aucune de ses deux configurations (dérive à β = 0,1, oscillation à β = 1,0).
6. **Le vérificateur filtre, il ne contredit pas** (v9, §14) : sur la graine dure de
   `lookup`, le modèle amorce copie la valeur du *mauvais* champ sur 9 tâches de banc sur
   30 ; ses propres succès ne contredisent jamais cette règle, et la boucle fermée reste à
   +0,000 [−0,100, +0,100] là où l'oracle, qui reçoit aussi les tâches ratées, fait +0,300.
   Le rendement d'une famille est borné par la part des tâches que le modèle ne réussit
   jamais — 18 sur 60 ici. **La reprise qui explore ce qu'elle a lu** (v11b, §16 : à la
   deuxième et troisième tentative, le départ du span copié depuis une observation est
   tiré parmi les 3 meilleurs) casse ce piège : 13 épisodes contradictoires sur 114 promus,
   0,583 → **0,817** (+0,233 [+0,100, +0,367]), 6 tâches jamais réussies au lieu de 18, à
   compute égal (+20 % de jetons générés). Dose-réponse sur trois pilotes : 0, 3, 13
   épisodes contradictoires → +0,000, +0,133, +0,233.
7. **Le proposeur du programme 3 ne démarre pas à 7 M** (§22) : sur les quatre barreaux de
   l'échelle de calibration, au mieux 9 propositions valides sur 30 (0,30 < 0,5), aucune
   nouvelle, quand le solveur atteint 0,950 ; la forme d'un appel s'apprend, pas deux
   listes alignées par position. La sonde a trouvé un défaut de la grammaire de décodage
   (des caractères de contrôle bruts admis dans une chaîne), corrigé pour tout décodage.

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

**Résultats v2, trois bras** (graine 3 : voir l'incident de reprise ci-dessous ; le chiffre
retenu est celui de la reprise propre) :

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 0,733 | 0,883 | 0,600 | 0,683 | 0,967 | 0,917 | **+0,183** [+0,067, +0,300] |
| closed | 2 | 0,567 | 0,500 | 0,550 | 0,400 | 0,517 | 0,600 | **+0,033** [-0,083, +0,150] |
| closed | 3 | 0,700 | 0,767 | 0,900 | 0,983 | 0,700 | 0,850 | **+0,150** [+0,033, +0,267] |
| oracle | 0 | 0,733 | 0,917 | 0,817 | 1,000 | 1,000 | 1,000 | +0,267 [+0,167, +0,383] |
| oracle | 2 | 0,567 | 0,617 | 0,717 | 1,000 | 0,983 | 0,967 | +0,400 [+0,267, +0,533] |
| oracle | 3 | 0,700 | 0,967 | 0,967 | 0,983 | 1,000 | 1,000 | +0,300 [+0,183, +0,417] |

Part de sorties malformées au banc, bras `closed` :

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 13 % | 4 % | 0 % | 12 % | 1 % | 1 % |
| closed | 2 | 0 % | 2 % | 8 % | 19 % | 2 % | 2 % |
| closed | 3 | 4 % | 0 % | 7 % | 0 % | 0 % | 0 % |

Bits par octet tenus à l'écart :

| Bras | Graine | BPB t0 | BPB t5 | Δ |
|---|---:|---:|---:|---:|
| closed | 0 | 2,104 | 2,527 | +0,422 |
| closed | 2 | 2,090 | 2,580 | +0,490 |
| closed | 3 | 2,124 | 2,573 | +0,448 |
| oracle | 0 | 2,104 | 2,546 | +0,442 |
| oracle | 2 | 2,090 | 2,514 | +0,423 |
| oracle | 3 | 2,124 | 2,519 | +0,395 |

Sur les graines 0 et 3, v2 égale v1 tour par tour (mêmes tokens générés, mêmes promus :
116 et 128) : aucun blanc n'y a jamais été émis. La grammaire compacte n'a agi que sur la
graine 2, et seulement au tour 2 (0,233 → 0,550) ; dès le tour 3, la boucle y retombe par
le mode (b) (0,400, copies du mauvais champ) et finit à 0,600, soit +0,033 au lieu de 0,000.

**Verdicts v2.**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H1** | **échoue** | gain > 0 sur 3 graines sur 3, mais l'intervalle de la graine 2 contient zéro ([−0,083, +0,150]) ; banc cumulé 180 tâches, 31 gagnées, 9 perdues |
| **H2** | rapporté | gain(closed) / gain(oracle) = **0,38** (0,122 / 0,322), contre 0,35 en v1 |
| **H3** | **passe** | Δ BPB closed +0,454 contre oracle +0,420, écart +0,033 ≤ 0,05 |
| **H4** | **échoue** | gain par heure du dernier tour : −0,82, +1,15, +2,76 |
| **H5** | **échoue** | `closed-klpo` 0,733 → **0,000 dès le tour 1** (malformées 58 %), arrêté ; amendement 6 |
| **H6** | **échoue** | malformées ≤ 5 % à chaque tour : non, graine 2 tours 2 et 3 à 8 % puis 20 % (modes c et b) ; le mode (a), lui, a disparu |

**Bras KLPO v2.** Même effondrement qu'en v1, plus rapide : 25 épisodes récompensés sur 42,
60 pas KLPO, perte +1,0 → −10,3, log p − log q = −0,83 nat/token, 0,000 au banc. À β = 0,1
sur des enregistrements réutilisés 60 fois, la KL ne retient rien ; v3 le rejoue à β = 1,0
et 20 pas (amendement 6), la seule comparaison propre étant alors `closed-klpo` v3 contre
`closed` v3.

**Incident de reprise, graine 3.** Le conteneur a redémarré pendant le tour 5 du bras
`closed` : le tour avait généré et mis en quarantaine ses 24 épisodes promus, sans écrire
son enregistrement. La reprise a régénéré le tour (mêmes tâches, même graine : 24 épisodes
identiques) et entraîné sur les deux copies : **0,750 avec 152 promus**. Une copie du run
remise à l'état du tour 4 (entrées d'avant le tour 5 retirées, checkpoint du tour 4) et
rejouée seule donne **0,850 avec 128 promus**, exactement v1. C'est ce chiffre qui figure
ci-dessus. Le défaut est corrigé pour v3 (entrées étiquetées par tour, orphelins écartés à
la reprise, test), et compté comme vingtième défaut silencieux dans CLAUDE.md.

**Note d'exploitation.** Deux processus d'entraînement en parallèle sur les quatre cœurs
(sur-souscription OpenMP) ont ralenti chacun d'un facteur dix ; un banc de trois minutes
n'aboutissait pas en cinquante. Un seul run à la fois sur cette machine.

**Suite : pilote v3** (docs/31 amendement 5, H7) : taux de pointe des tours divisés par
quatre, rien d'autre pour `oracle`, `closed`, `frozen` ; `closed-klpo` à β = 1,0 et 20 pas
(amendement 6). Graines 0, 2, 3, amorces réutilisées.

## 8. Pilote v3 : le taux de pointe divisé par quatre

Amendement 5 : `--lr-scale 0.25` pour les tours (Muon 0,0025, AdamW 5e-4), grammaire
compacte, rien d'autre ; amorces réutilisées ; bras KLPO à β = 1,0 et 20 pas (amendement 6).

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 0,733 | 0,633 | 0,917 | 0,917 | 0,950 | 0,967 | **+0,233** [+0,117, +0,367] | +0,183 |
| closed | 2 | 0,567 | 0,017 | 0,533 | 0,533 | 0,650 | 0,683 | **+0,117** [-0,017, +0,250] | +0,033 |
| closed | 3 | 0,700 | 0,600 | 0,933 | 0,950 | 0,933 | 0,967 | **+0,267** [+0,167, +0,383] | +0,150 |
| oracle | 0 | 0,733 | 0,750 | 0,900 | 0,950 | 1,000 | 0,983 | +0,250 [+0,150, +0,367] | +0,267 |
| oracle | 2 | 0,567 | 0,667 | 0,833 | 0,867 | 0,850 | 0,867 | +0,300 [+0,167, +0,433] | +0,400 |
| oracle | 3 | 0,700 | 0,783 | 0,950 | 1,000 | 1,000 | 0,983 | +0,283 [+0,167, +0,400] | +0,300 |

Part de sorties malformées au banc, bras `closed` :

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| closed | 0 | 13 % | 19 % | 3 % | 1 % | 2 % | 0 % |
| closed | 2 | 0 % | 11 % | 5 % | 2 % | 1 % | 0 % |
| closed | 3 | 4 % | 11 % | 0 % | 1 % | 1 % | 0 % |

Bits par octet tenus à l'écart :

| Bras | Graine | BPB t0 | BPB t5 | Δ v3 | Δ v2 |
|---|---:|---:|---:|---:|---:|
| closed | 0 | 2,104 | 2,165 | +0,060 | +0,422 |
| closed | 2 | 2,090 | 2,178 | +0,088 | +0,490 |
| closed | 3 | 2,124 | 2,180 | +0,056 | +0,448 |
| oracle | 0 | 2,104 | 2,164 | +0,060 | +0,442 |
| oracle | 2 | 2,090 | 2,172 | +0,082 | +0,423 |
| oracle | 3 | 2,124 | 2,180 | +0,056 | +0,395 |

**Ce que v3 change.** L'oubli est divisé par sept (Δ BPB moyen du bras `closed` +0,068
contre +0,454 en v2 ; l'oracle +0,066 contre +0,420), le gain du bras `closed` monte
(+0,206 en moyenne contre +0,122) et son rendement contre l'oracle passe de 0,38 à
**0,74**. Sur le banc cumulé, 44 tâches gagnées pour 7 perdues (v2 : 31 pour 9). Les creux
des tours 2 à 4 ont disparu sur les trois graines, et la part malformée finit à 0 %
partout. Le prix : un premier tour qui **recule** sur les trois graines (0,633, 0,017, 0,600
contre 0,733, 0,567, 0,700), le plus fort sur la graine 2, où 14 trajectoires vérifiées
mais bâclées (notes répétées) suffisent à faire boucler le modèle sur `note` sans jamais
appeler `done` ; il s'en relève seul dès le tour 2 (0,533) et finit à 0,683.

**Verdicts v3, trois bras.**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H1** | **échoue**, de peu | gain > 0 sur 3 graines sur 3 ; l'intervalle de la graine 2 contient encore zéro ([−0,017, +0,250]) |
| **H2** | rapporté | gain(closed) / gain(oracle) = **0,74** (0,206 / 0,278) |
| **H3** | **passe** | Δ BPB closed +0,068 contre oracle +0,066 |
| **H4** | **passe** | gain par heure du dernier tour : +0,32, +0,64, +0,69 |
| **H7** | **échoue** sur (i), passe (ii) et (iii) | (i) tour 1 sous succès(t0) − 0,10 sur les graines 2 et 3 ; (ii) +0,206 ≥ +0,122 ; (iii) +0,068 ≤ +0,227 |
| **H5** | **échoue** | gain moyen `closed-klpo` −0,033 (−0,350 / −0,033 / +0,283) contre +0,206 pour `closed` ; Δ BPB +0,069 contre +0,068 |

**Bras KLPO v3 (β = 1,0, 20 pas).** Il ne s'effondre plus d'un coup, il oscille :

| Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain | Malformées (t1 … t5) | log p − log q (nat/token) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0,733 | **0,950** | 0,850 | **0,200** | 0,867 | 0,383 | −0,350 | 1 %, 9 %, 29 %, 9 %, 51 % | −0,25 → −0,65 |
| 2 | 0,567 | 0,533 | 0,483 | 0,483 | 0,567 | 0,533 | −0,033 | ≤ 2 % | −0,23 → −0,50 |
| 3 | 0,700 | 0,517 | 0,467 | 0,433 | 0,833 | **0,983** | +0,283 | 17 %, 17 %, 18 %, 12 %, 1 % | −0,13 → −0,61 |

Le premier tour de la graine 0 est le meilleur de tous les pilotes (0,733 → 0,950 en un
tour, contre 0,633 pour `closed` v3 et 0,750 pour l'oracle) : à β = 1,0, la politique reste à
0,25 nat/token de son échantillonneur et apprend aussi des 17 épisodes non récompensés.
Mais deux tours plus loin elle tombe à 0,200 avec 29 % de malformées, remonte, retombe.
L'oubli, lui, reste faible (+0,07 bit/octet). La variance d'un tour à l'autre est le
problème, pas la dérive de la langue ; la KL vers l'échantillonneur borne chaque tour, pas
la trajectoire des tours. H5 échoue en v3 comme en v1 et v2, mais pour une raison
différente à chaque fois (dérive à β = 0,1, oscillation à β = 1,0), et la piste n'est pas
close : un β entre les deux, moins de pas encore, ou des tirages frais de l'échantillonneur
à chaque pas (la variante sans biais du rapport) sont les trois candidats, à une variable
par run.

H7 échoue par sa condition de non-effondrement, et c'est la bonne lecture : la recette
n'est pas la seule cause des creux. Le taux réduit supprime ceux qui venaient de
l'entraînement lui-même (tours 2 à 4, oracle compris) mais pas celui du premier tour, qui
vient des **données** : des trajectoires vérifiées par le résultat et bâclées dans le
processus. D'où le bras `closed-clean` (amendement 7, H8), lancé ensuite sur les mêmes
amorces et comparé à `closed` v3.

## 9. Pilote v4 : ne promouvoir que la forme canonique

Amendement 7 : bras `closed-clean`, identique à `closed` v3 (taux ÷ 4, grammaire compacte,
mêmes amorces, mêmes tours) sauf la promotion, réservée aux trajectoires sans pas malformé,
sans `done` refusé, sans pas répété, et finissant par `done`. Comparé au bras `closed` v3
du même répertoire.

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Écartées / promues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-clean | 0 | 0,733 | 0,600 | 0,983 | 0,983 | 0,983 | 0,983 | **+0,250** [+0,133, +0,367] | +0,063 | 12 / 126 |
| closed-clean | 2 | 0,567 | 0,550 | 0,567 | 0,633 | 0,633 | 0,833 | **+0,267** [+0,150, +0,383] | +0,083 | 3 / 96 |
| closed-clean | 3 | 0,700 | 0,800 | 0,933 | 0,883 | 0,917 | 0,950 | **+0,250** [+0,133, +0,367] | +0,061 | 3 / 129 |
| closed | 0 | 0,733 | 0,633 | 0,917 | 0,917 | 0,950 | 0,967 | +0,233 [+0,117, +0,367] | +0,060 | 0 / 125 |
| closed | 2 | 0,567 | 0,017 | 0,533 | 0,533 | 0,650 | 0,683 | +0,117 [-0,017, +0,250] | +0,088 | 0 / 79 |
| closed | 3 | 0,700 | 0,600 | 0,933 | 0,950 | 0,933 | 0,967 | +0,267 [+0,167, +0,383] | +0,056 | 0 / 123 |

**Ce que le filtre change.** Il écarte peu : 12, 3 et 3 trajectoires sur cinq tours. Mais
sur la graine 2, les **2 trajectoires sur 14** écartées au tour 1 (deux notes en double)
sont exactement celles qui faisaient boucler le modèle sur `note` : 0,550 au lieu de
0,017, puis une montée régulière jusqu'à 0,833 (`closed` v3 : 0,683 ; v2 : 0,600 ; v1 :
0,567). Sur les trois graines, le gain est **+0,256** en moyenne contre +0,206, avec les
trois intervalles qui excluent zéro ([+0,133, +0,367], [+0,150, +0,383], [+0,133,
+0,367]) ; banc cumulé : 49 tâches gagnées pour 3 perdues sur 180. L'oubli est
inchangé (+0,069 contre +0,068). Le rendement contre l'oracle v3 monte à **0,92**
(0,256 / 0,278) : à budget de tâches égal, la boucle qui n'apprend que de ses propres
succès de forme canonique fait presque aussi bien que celle qui reçoit la trajectoire
parfaite de chaque tâche.

**Verdicts v4.**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H8** | **échoue** sur (i), passe (ii) et (iii) | (i) graine 0, tour 1 : 0,600 < 0,633 (mode c, corps de `done` : 23 % de malformées, 2 trajectoires écartées seulement) ; (ii) +0,256 ≥ +0,206 ; (iii) +0,069 ≤ +0,088 |
| **H1**, appliquée à `closed-clean` | **passe** | gain > 0 sur 3 graines sur 3, intervalle excluant zéro sur chacune |
| **H2**, `closed-clean` | rapporté | 0,92 |

H8 échoue par la même condition que H7, et sur la même graine 0 au même tour : ce recul
n'est ni la recette (v3) ni les données bâclées (v4), c'est le mode (c), le corps de
`done` que le modèle amorce sait déjà mal écrire (13–17 % au tour 0) et que le premier
tour d'entraînement dégrade avant que les tours suivants ne le réparent. Sa correction est
ailleurs : dans l'amorce (plus d'exemples de `done` à arguments vides) ou dans le budget de
pas. Le critère de non-effondrement à −0,10 était trop strict d'une tâche sur soixante ;
il est gardé tel quel, et l'échec rapporté.

**Ce que quatre pilotes ont établi, en trois lignes.** (1) Un désaccord entraînement /
décodage (grammaire tolérante) et deux défauts de recette (taux plein par tour, promotion
du processus bâclé) suffisaient à faire d'une boucle qui marche une boucle qui échoue ; les
trois sont trouvés par le diagnostic jeton par jeton, jamais par le score seul. (2) Corrigés
un par un, avec une variable par pilote, la boucle fermée passe de +0,111 à +0,256 de succès
tenu à l'écart, de 0,35 à 0,92 de l'oracle, et de +0,44 à +0,07 bit/octet d'oubli.
(3) KLPO, dans sa forme actuelle (enregistrements réutilisés, β fixe), n'a jamais tenu
cinq tours ; la piste reste ouverte, à une variable par run.

## 10. Pilote v5 : `done` sans argument

Amendement 8 : le schéma de `done` perd sa clé facultative `summary` que rien ne lisait,
et le scanner comme le validateur traitent un schéma vide comme « aucun paramètre » (ils
le traitaient comme « tout est permis », ce qui laissait le mode (c) intact au premier
essai, abandonné et relancé). Même recette que v4 pour le reste ; les mêmes amorces.

**À poids égaux, le mode (c) disparaît** : le modèle amorce de la graine 0 passe de 0,733
(malformées 17 % / 10 %) à **0,850 (0 % / 0 %)** au tour 0, sans qu'un poids ait changé.

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Malformées max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-clean | 0 | 0,850 | 0,817 | 0,950 | 0,950 | 0,967 | 1,000 | **+0,150** [+0,067, +0,250] | +0,062 | 1 % |
| closed-clean | 2 | 0,567 | 0,550 | 0,567 | 0,583 | 0,667 | 0,667 | **+0,100** [-0,033, +0,233] | +0,082 | 2 % |
| closed-clean | 3 | 0,700 | 0,933 | 0,967 | 1,000 | 0,983 | 1,000 | **+0,300** [+0,183, +0,417] | +0,058 | 1 % |
| oracle | 0 | 0,850 | 0,900 | 0,967 | 1,000 | 1,000 | 0,983 | +0,133 [+0,050, +0,233] | +0,060 | 0 % |
| oracle | 2 | 0,567 | 0,667 | 0,833 | 0,867 | 0,850 | 0,867 | +0,300 [+0,167, +0,433] | +0,082 | 0 % |
| oracle | 3 | 0,700 | 0,950 | 0,950 | 1,000 | 1,000 | 0,983 | +0,283 [+0,167, +0,400] | +0,056 | 0 % |

**Verdicts v5.**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H9 (i)** aucun tour sous t0 − 0,10 | **passe** | pires tours : 0,817 (t0 0,850), 0,550 (0,567), 0,700 (0,700) |
| **H9 (ii)** malformées ≤ 5 % à chaque tour | **passe** | maximum 2 % |
| **H9 (iii)** rendement ≥ 0,8 | **échoue** | 0,767 (+0,183 / +0,239) |
| **H9** | **échoue** sur (iii) | |
| **H1**, `closed-clean` | échoue, de peu | intervalle de la graine 2 : [−0,033, +0,233] ; banc cumulé 39 gagnées, 6 perdues |

Deux graines sur trois finissent à **60/60** sur les tâches jamais vues (graines 0 et 3,
comme l'oracle), l'oubli reste à +0,06–0,08 bit/octet, et plus aucun tour ne recule de
plus d'une tâche sur soixante : les quatre modes d'échec identifiés sont fermés. Le
rendement chute pourtant à 0,77, pour deux raisons arithmétiques : sur la graine 0 la
marge est réduite (départ 0,850, oracle +0,133) ; sur la graine 2 le bras fermé fait
+0,100 quand l'oracle fait +0,300 — les 12 à 21 épisodes vérifiés par tour n'y suffisent
pas là où 30 trajectoires parfaites suffisent. La graine 2 est le vrai reste : un départ à
0,567 où chaque tour ne promeut qu'une douzaine d'épisodes et où v4 (+0,267) et v5 (+0,100)
divergent par le seul tirage des générations. La variance d'échantillonnage à
30 tâches × 2 tentatives est la limite de ce pilote, pas un défaut de plus.

## 11. Pilote v6 : une seconde famille, `files`

Amendements 9 et 10. À l'amorce de 100 / 200, `files` est résolu d'emblée (1,000 / 0,983 /
1,000 sur les graines 0, 1, 2) : les 7 % de docs/09 tenaient aux défauts de décodage
depuis corrigés. L'échelle d'amorce pré-enregistrée retient la première marche, 50 / 100 :
départs 0,700 / 0,783 / 0,917, graines 0 et 1 dans la fenêtre [0,10, 0,90]. Recette v5
(taux ÷ 4, grammaire compacte, `done` sans argument, promotion canonique).

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Résolues par tour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| closed-clean | 0 | 0,700 | 0,817 | 0,850 | 0,883 | 0,817 | 0,800 | **+0,100** [-0,017, +0,217] | +0,068 | 23/22/25/26/25 |
| closed-clean | 1 | 0,783 | 0,750 | 0,900 | 0,867 | 0,850 | 0,867 | **+0,083** [+0,000, +0,167] | +0,078 | 22/23/26/26/25 |
| oracle | 0 | 0,700 | 0,933 | 0,983 | 1,000 | 0,983 | 0,967 | +0,267 [+0,150, +0,383] | +0,066 | 30/30/30/30/30 |
| oracle | 1 | 0,783 | 0,933 | 0,950 | 1,000 | 0,917 | 0,950 | +0,167 [+0,050, +0,283] | +0,065 | 30/30/30/30/30 |

**Verdicts v6.**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H10 (i)** gain > 0, intervalle excluant zéro | **échoue** | [−0,017, +0,217] et [+0,000, +0,167] |
| **H10 (ii)** rendement ≥ 0,6 | **échoue** | 0,423 (+0,092 / +0,217) |
| **H10 (iii)** aucun tour sous t0 − 0,10 | **passe** | pires tours 0,800 (t0 0,700) et 0,750 (0,783) |
| **H10** | **échoue** sur (i) et (ii) | |

La boucle **démarre** sur `files` (gains positifs, aucun effondrement, malformées 0 %,
aucune trajectoire écartée, oubli +0,07) mais plafonne vers 0,85 quand l'oracle atteint
0,95–0,97. Le diagnostic sur les trente tâches du banc 7 (graine 0, tour 5) est sans
ambiguïté : **5 des 6 échecs sont la boucle sur `note`** — le bon nom de fichier est noté,
puis `note` est répété jusqu'au bout des quatre pas sans jamais appeler `done` ; le sixième
est un argument de `grep` tronqué (`ir` pour `iris`). Aucune donnée bâclée n'est en cause
(0 trajectoire écartée) : c'est la tête de sélection qui, après une note, ne bascule pas
vers `done`, ce que le budget de quatre pas transforme en échec. Les trajectoires promues
ne contiennent jamais deux pas identiques ; le décodeur, lui, l'autorise. Même désaccord
entraînement / décodage que les blancs (a) et le corps de `done` (c) : d'où l'amendement 11.

## 12. Pilote v7 : pas de pas répété au décodage

Amendement 11 : `AgentConfig.no_repeat_action`, la grammaire du pas *i* exclut le nom de
l'action du pas *i − 1*. Mêmes amorces `files` 50 / 100 que v6, même recette pour le reste.

**À poids égaux**, le modèle amorce de la graine 0 passe de 0,700 à **0,850** au tour 0 :
neuf de ses dix-huit échecs étaient la boucle sur `note`. Celui de la graine 1 ne bouge pas
(0,783) : ses échecs sont d'une autre nature (ci-dessous).

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | v6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-clean | 0 | 0,850 | 0,800 | 0,850 | 0,817 | 0,883 | 0,933 | **+0,083** [+0,017, +0,150] | +0,070 | 0,700 → 0,800 |
| closed-clean | 1 | 0,783 | 0,833 | 0,900 | 0,883 | 0,900 | 0,900 | **+0,117** [+0,033, +0,217] | +0,070 | 0,783 → 0,867 |
| oracle | 0 | 0,850 | 0,933 | 0,983 | 1,000 | 0,983 | 0,967 | +0,117 [+0,033, +0,217] | +0,066 | 0,700 → 0,967 |
| oracle | 1 | 0,783 | 0,950 | 0,950 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,065 | 0,783 → 0,950 |

**Verdicts v7 (H11 = H10 sous v7).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H11 (i)** intervalles excluant zéro | **passe** | [+0,017, +0,150] et [+0,033, +0,217] |
| **H11 (ii)** rendement ≥ 0,6 | **passe**, à la limite | 0,600 (+0,100 / +0,167) |
| **H11 (iii)** aucun tour sous t0 − 0,10 | **passe** | pires tours 0,800 (t0 0,850) et 0,783 (0,783) |
| **H11** | **passe** | banc cumulé : 13 tâches gagnées, 1 perdues sur 120 |

C'est la première hypothèse de généralité tenue : sur une seconde famille, la boucle
fermée gagne sur chaque graine avec un intervalle qui exclut zéro, sans effondrement, à 60 %
du rendement de l'oracle. Le diagnostic (banc 7, tour 5, la règle active) ne laisse que
deux modes, tous deux hors de la boucle elle-même :

- graine 0, 1 échec sur 30 : un argument de `grep` tronqué (`ir` pour `iris`) ;
- graine 1, 4 échecs sur 30, **tous** un pointeur de copie en retard d'un jeton : `chor_0.txt`
  pour `anchor_0.txt`, `antern_2.txt` pour `lantern_2.txt`. Le bon fichier est trouvé, sa
  copie perd son premier jeton, `done` est refusé. C'est le mode **(e)** : une cible de copie
  mal alignée à l'entraînement quand la valeur ouvre l'observation sans espace devant elle
  propre à l'amorce de la graine 1. Vérification faite : les cibles de copie des
  trajectoires parfaites rendues sont justes (0 sur 60 mal alignées), c'est le pointeur
  **appris** qui est décalé. Et la boucle ne peut pas le corriger : à la génération, le
  pointeur prend l'argmax quelle que soit la température, donc les tâches qu'il rate ne
  donnent jamais d'épisode vérifié. D'où l'amendement 12 (le pointeur explore).

## 13. Pilote v8 : le pointeur de copie explore

Amendement 12 : `AgentConfig.sample_copy`, début et fin du span copié tirés de leur softmax
à la température de génération (0,7) ; banc glouton inchangé, mêmes amorces `files`
50 / 100, recette v7 pour le reste.

| Bras | Graine | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | v7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-clean | 0 | 0,850 | 0,800 | 0,833 | 0,867 | 0,867 | 0,867 | **+0,017** [+0,000, +0,050] | +0,069 | +0,083 |
| closed-clean | 1 | 0,783 | 0,767 | 0,867 | 0,883 | 0,900 | 0,917 | **+0,133** [+0,050, +0,217] | +0,065 | +0,117 |
| oracle | 0 | 0,850 | 0,933 | 0,983 | 1,000 | 0,983 | 0,967 | +0,117 [+0,033, +0,217] | +0,066 | +0,117 |
| oracle | 1 | 0,783 | 0,950 | 0,950 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,065 | +0,217 |

**Verdicts v8 (H12).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H12 (i)** intervalles excluant zéro | **échoue** | graine 0 : [+0,000, +0,050] |
| **H12 (ii)** rendement ≥ 0,6 | **échoue** | 0,450 (+0,075 / +0,167) |
| **H12 (iii)** aucun tour sous t0 − 0,10 | passe | pires tours 0,800 et 0,767 |
| **H12 (iv)** graine 1 : gain ≥ +0,117 et ≤ 1 copie décalée sur 30 | **échoue** | gain +0,133 ≥ +0,117, mais **3** copies décalées sur 30 (`antern_x.txt` pour `lantern_x.txt`) |
| **H12** | **échoue** | |

L'exploration du pointeur ne corrige pas le mode (e) : sur la graine 1, les trois échecs du
banc 7 sont encore des copies en retard d'un jeton, et la graine 0, qui n'en avait pas en
v7, en montre deux (`con_0.txt`, `acon_2.txt` pour `beacon_x.txt`). Lecture : le pointeur
décalé est **confiant** ; à température 0,7 sa softmax ne tire presque jamais le bon
début, et quand la tâche est réussie par une autre voie, la ligne promue enseigne la même
cible alignée qu'avant, que le pointeur reçoit déjà depuis l'amorce sans la suivre. Ce
que la boucle ne réussit jamais, elle ne l'apprend jamais : le mur d'amorçage de docs/31
§2 au niveau du jeton, et une limite de **couverture**, pas de décodage. L'oracle, qui
reçoit trente trajectoires parfaites par tour sur ces mêmes noms, le corrige en deux tours.
Les écarts de gain entre v7 et v8 (±4 tâches sur 60) sont dans le bruit d'échantillonnage
de la génération. `sample_copy` reste disponible mais désactivé par défaut ; la recette de
référence est celle de v7.

Pour l'échelle A100 (docs/31 §3), trois choses sont acquises dès maintenant : la marge de
départ se calibre avant de lancer (tour 0 seul, par graine et par famille) ; la recette par
tour doit être mesurée sur l'oubli avant tout (un planning neuf à taux plein par tour est
la cause du +0,44) ; le banc doit rapporter la part malformée à chaque tour, sans quoi les
deux modes d'échec se confondent.

## 14. Pilote v9 : la graine dure de `lookup` sous la recette de référence

Amendement 13 : `lookup`, graine 2, amorce 100 / 200 réutilisée (celle de v3 à v5), bras
`closed-clean` seul avec `no_repeat_action`, taux ÷ 4, grammaire compacte, `done` sans
argument, promotion canonique ; comparé aux bras v5 de la même graine et de la même amorce.
Le tour 0 diffère d'une tâche entre v9 et v5 (0,583 contre 0,567) parce que le banc v9
décode sous `no_repeat_action`.

| Bras | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Malformées max | Heures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-clean v9 (référence) | 0,583 | 0,567 | 0,600 | 0,533 | 0,517 | 0,583 | **+0,000** [−0,100, +0,100] | +0,083 | 1 % | 0,28 |
| closed-clean v5 | 0,567 | 0,550 | 0,567 | 0,583 | 0,667 | 0,667 | +0,100 [−0,033, +0,233] | +0,082 | 2 % | 0,29 |
| oracle v5 | 0,567 | 0,667 | 0,833 | 0,867 | 0,850 | 0,867 | +0,300 [+0,167, +0,433] | +0,082 | 0 % | 0,25 |

Génération v9, par tour : 18, 21, 22, 16, 21 tâches résolues sur 30 (43, 39, 38, 44, 39
épisodes), sans tendance ; 98 épisodes promus en cinq tours, **tous canoniques** (aucune
rétrogradation, tous en trois pas : `read_file` → `note` → `done`). Sur le banc, 28 tâches
sur 60 réussissent à chaque tour, **18 n'y arrivent jamais**, 14 basculent ; entre le tour 0
et le tour 5, 5 gagnées, 5 perdues.

**Verdicts v9 (H13).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H13 (i)** intervalle excluant zéro | **échoue** | +0,000 [−0,100, +0,100] |
| **H13 (ii)** aucun tour sous 0,467 | passe | pire tour 0,517 |
| **H13 (iii)** rendement contre l'oracle v5 ≥ 0,6 | **échoue** | 0,000 (+0,000 / +0,300) |
| **H13** | **échoue** | |

**Diagnostic, jeton par jeton** (banc 7, décodage glouton sous `no_repeat_action`, mêmes
poids que le banc). Au tour 0, 14 échecs sur 30 : **9 notent la valeur de l'autre champ
textuel** — `ripple` pour `city`, `Kyoto` pour `code`, jamais l'année (`{"city":…,"year":…,
"code":…}`, ordre des clés constant) —, 5 sont la boucle sur `done` du mode (d) (`note`
d'un fragment `done","args":{}}ver`). Au tour 5, 12 échecs : **11 mauvais champ, 1 boucle
sur `done`**. La boucle a réduit le mode qu'elle réussit parfois ; elle n'a pas touché à
celui qu'elle ne réussit jamais.

Lecture — mode **(f)**, une erreur de *sélection* systématique. Le modèle amorcé applique
sur ces tâches une règle qui n'est pas « la valeur de la clé nommée » ; les épisodes qu'il
promeut sont exactement ceux où sa règle coïncide avec la clé nommée, et ils sont donc
compatibles avec elle : rien dans le bassin promu ne la contredit, et 60 pas par tour sur ce
bassin la confirment autant qu'ils enseignent la bonne. L'oracle reçoit à chaque tour les
trajectoires parfaites des tâches *ratées* aussi, qui la contredisent : +0,300 en cinq
tours à partir des mêmes poids. À température 0,7 et deux tentatives, le pointeur ne
bascule pas (v8 l'avait montré pour le mode (e)), et le nombre de tâches résolues par tour
ne monte pas.

Ce qui en découle, au-delà de cette graine : **un vérificateur filtre, il ne contredit
pas**. La boucle fermée corrige ce que le modèle réussit *parfois* (grammaire, boucle sur
`note` ou sur `done`, dérive de contenu — les modes (a) à (d)) et ne peut pas corriger ce
qu'il rate *toujours* sur un sous-ensemble (modes (e) et (f)). Le rendement contre l'oracle
sur une famille est borné par 1 − (part des tâches jamais réussies) : 0,70 ici, 0,87 à 1,0
pour l'oracle. Trois voies, aucune acquise : une exploration qui atteint l'autre champ
(le pointeur tiré de v8 n'y arrivait pas), un signal négatif sur les épisodes ratés (le bras
KLPO était cela et n'a pas tenu à 7 M), ou un curriculum où le sous-ensemble raté devient
atteignable. Pour l'échelle A100 : rapporter à chaque tour l'ensemble des tâches de banc
jamais réussies, en plus du gain — c'est lui qui dit si la boucle a encore quelque chose à
apprendre d'elle-même.

## 15. Pilote v10 : deux familles dans une boucle — interrompu après un tour

Amendement 14 : `lookup` + `files` dans une seule boucle, amorce **mixte** (50 trajectoires
parfaites de chaque famille, entrelacées, 100 pas), recette de référence, 30 tâches de
chaque famille par tour, 60 pas pour tous les bras, banc 2 × 30 par famille. La calibration
(graine 0, tour 0 seul) a donné 0,467 sur chaque famille — retenue par la règle de
l'amendement 14 (départ strictement entre 0 et 1 sur chaque famille).

| Tour | `lookup` | `files` | Malformées `files` | BPB | Résolues (l / f) | Rétrogradées | Lignes entraînées |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0,467 | 0,467 | 0,8 % / 2,5 % | 2,093 | — | — | — |
| 1 | 0,383 | **0,000** | **40 % / 38 %** | 2,061 | 12 / 14 | 18 (**14 `files`**, 4 `lookup`) | 8, toutes `lookup` |

Le bras mixte a résolu 26 tâches sur 60 au tour 1, en a rétrogradé 18 comme non
canoniques — les **quatorze** de `files` —, s'est entraîné 60 pas sur 8 lignes `lookup`,
et `files` est tombé de 0,467 à 0 avec 40 % de sorties malformées. Le pilote a été
interrompu là ; ses bras mono-famille et oracle à cette amorce n'ont pas été lancés (ils
répondraient à une question que le tour 1 a déjà tranchée), et v11 a pris le processeur.

**Diagnostic, sur le checkpoint d'amorce** (banc 7, même décodage que le banc) :

| Famille | Résolues | … dont canoniques | Forme dominante des succès | Échecs |
|---|---:|---:|---|---|
| `files` | 18 / 30 | **0** | `grep → done refusé → note → done` (18 / 18) | 11 avec deux `done` refusés, 1 malformé |
| `lookup` | 13 / 30 | 5 | `read_file → done refusé → note → done` (8 / 13) | 17 |

À demi-exposition par famille (50 lignes et 100 pas, contre 50 / 100 pour `files` seule en
v6–v8 et 100 / 200 pour `lookup` seule en v3–v9), l'amorce mixte a appris à appeler
`done` dès la première observation. Le vérificateur accepte ces épisodes (le `done` refusé
est suivi de la bonne note) ; la promotion canonique de l'amendement 7 les refuse tous, à
raison — v4 a montré ce que deux trajectoires bâclées suffisent à enseigner. Sur `files`,
la boucle n'a donc **rien** à apprendre d'elle-même, et la boucle « mixte » est de fait
une boucle `lookup` seule : l'oubli par omission que H14 (iii) devait mesurer sur les bras
mono-famille s'est produit sur le bras qui devait l'éviter, et en un tour (0,467 → 0,000).
En v7, sur `files` seule, aucun des 121 épisodes promus en cinq tours n'avait été
rétrogradé ; en v9, aucun des 98 de `lookup`.

**Verdict.** H14 n'est pas évaluée : sa règle de calibration était fausse. Le succès au
tour 0 ne dit pas ce que la boucle peut apprendre ; il faut le **succès canonique** —
ce que le banc mesure désormais (`canonical_rate`, `canonical_by_family`, amendement 16 :
retenue si chaque famille démarre strictement entre 0 et 0,95 **et** a un succès canonique
strictement positif). v10b relance le même protocole avec une amorce 100 / 200 par famille,
après v11.

Deux choses acquises pour l'échelle A100 : (1) une amorce partagée entre familles se
calibre famille par famille **sur la forme** de ses succès, pas sur leur nombre ; (2) une
famille sans épisode canonique dans le bassin n'est pas seulement stagnante, elle est
**détruite** par les 60 pas sur les autres — l'oubli par omission est rapide (un tour),
et une boucle multi-familles doit vérifier à chaque tour que chaque famille a apporté des
lignes, ou geler la mise à jour.

**Addendum v10b — l'échelle de calibration (amendement 16, note ; amendement 20).** Sur
le succès canonique, aucune graine ne laisse une marge aux deux familles à la fois :

| Graine | *N* / *S* par famille | `lookup` (canonique) | `files` (canonique) | Verdict |
|---:|---:|---:|---:|---|
| 0 | 50 / 100 | 0,467 (0,167) | 0,467 (**0,000**) | v10 : `files` sans épisode canonique |
| 0 | 100 / 200 | 0,900 (0,750) | **1,000** (1,000) | `files` saturé |
| 1 | 100 / 200 | **1,000** (0,150) | 0,783 (0,333) | `lookup` saturé |
| 2 | 100 / 200 | 0,100 (0,083) | **1,000** (1,000) | `files` saturé |
| 0 | 75 / 150 | 0,367 (0,367) | **1,000** (1,000) | `files` saturé ; retenue sous la règle assouplie |

Sous une amorce partagée, `files` passe de zéro succès canonique à la saturation entre 50 et
75 trajectoires, et `lookup` varie de 0,10 à 1,00 selon la graine à *N* / *S* égaux — la
marge d'une famille dépend autant du tirage de l'amorce que de sa taille. L'amendement 20
juge une famille saturée sur ce qu'elle garde (perte ≤ 0,05) et lance v10c sur la graine 0
à 75 / 150 : quatre bras, cinq tours, recette de référence sans exploration. Résultats en
§19.

## 16. Pilotes v11 et v11b : la reprise explore le pointeur

Amendements 15 et 17, même graine dure de `lookup` (2), même amorce et même recette que
v9 ; seule la génération change, le banc reste glouton. v11 : à la deuxième tentative, le
départ de **chaque** span copié est tiré uniformément parmi les 3 meilleurs du pointeur.
v11b : le tirage ne porte que sur les spans dont le départ préféré est dans une
**observation d'outil** (le nom de fichier lu dans le but garde l'argmax), et deux reprises
au lieu d'une. `explorés` compte les épisodes vérifiés issus des reprises qui explorent :
les seuls qui contredisent la règle du modèle.

| Bras | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Explorés | Jamais réussies | Jetons générés |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v9 (témoin, sans exploration) | 0,583 | 0,567 | 0,600 | 0,533 | 0,517 | 0,583 | +0,000 [−0,100, +0,100] | +0,083 | 0 | 18 | 47 k |
| v11 (tous les spans, 1 reprise) | 0,583 | 0,567 | 0,583 | 0,550 | 0,567 | 0,717 | +0,133 [+0,000, +0,267] | +0,087 | 3 | 10 | 49 k |
| v11b (observations, 2 reprises) | 0,583 | 0,600 | 0,583 | 0,483 | 0,650 | **0,817** | **+0,233** [+0,100, +0,367] | +0,085 | 13 | **6** | 57 k |
| oracle v5 (référence) | 0,567 | 0,667 | 0,833 | 0,867 | 0,850 | 0,867 | +0,300 [+0,167, +0,433] | +0,082 | — | — | 0 |

Explorés par tour, v11b : 3, 4, 3, 3, 0 (v11 : 1, 0, 2, 0, 0). Entre le tour 0 et le
tour 5 de v11b : 16 tâches gagnées, 2 perdues ; sur les 18 tâches que v9 n'a jamais
réussies, **11** sont réussies au tour 5 de v11b et 12 l'ont été au moins une fois.
Malformées : 0 % à tous les tours. Compute : 0,29 h contre 0,28 h (v9), +20 % de jetons
générés pour la troisième tentative.

**Verdicts (H15, critères de l'amendement 17 pour v11b).**

| Hypothèse | v11 | v11b |
|---|---|---|
| **H15 (i)** mécanisme : explorés ≥ 10 (v11) / ≥ 15 (v11b) | **échoue** (3) | **échoue** (13 : deux de moins que le seuil relevé ; le seuil initial de 10 est passé) |
| **H15 (ii)** gain > 0, intervalle excluant zéro | **échoue** ([+0,000, +0,267]) | **passe** ([+0,100, +0,367]) |
| **H15 (iii)** jamais réussies ≤ 12 sur 60 | passe (10) | **passe** (6) |
| **H15** | échoue | **échoue sur (i)**, de deux épisodes ; (ii) et (iii) passent |

**Diagnostic, banc 7, checkpoints finaux.** v11 : 8 échecs, dont 4 encore du mauvais champ
(`Perth` pour `code`, `ripple` pour `city`…). v11b : **4 échecs, aucun du mauvais champ**
(deux notes vides, un span malformé, un fragment) ; sur ces 4, le pointeur ne place plus
la bonne valeur au second rang que dans un cas. Le mode (f) a disparu du banc 7 en cinq
tours, avec 13 épisodes contradictoires sur 114 promus (11 %).

Lecture. Ce que §14 concluait — un vérificateur filtre, il ne contredit pas — tient, et la
réponse est de fabriquer la contradiction **là où elle est bon marché** : à la reprise,
sur les tâches que la politique vient de rater, et seulement sur le choix que le modèle
peut rater systématiquement (ce qu'il copie depuis ce qu'il a lu), pas sur ce que le but
lui donne. v11, qui explorait aussi le nom de fichier, ne réussissait une reprise que si
les deux tirages sortaient bien (1 / 9) : 3 épisodes en cinq tours, et un gain à la limite
du bruit. La relation est monotone sur trois pilotes à poids, tâches et recette égaux : 0,
3, 13 épisodes contradictoires → +0,000, +0,133, +0,233 ; le rendement contre l'oracle v5
passe de 0 à 0,78. Le tour 3 de v11b (0,483) rappelle que la variance d'un banc de 60
tâches est de ±0,1 ; c'est le tour 5 et l'ensemble jamais réussi qui portent le verdict.

Ce que cela change pour l'échelle A100 : l'exploration du pointeur à la reprise entre
dans la recette de référence comme option **validée sur une graine et une famille** ; sa
généralité (`files`, mode (e) : le pointeur décalé d'un jeton) est la variable suivante
(amendement 18). Et la comptabilité doit rapporter `explorés` à chaque tour : c'est le
nombre d'épisodes qui apprennent quelque chose que le modèle ne savait pas déjà.

## 17. Pilote v12 : l'exploration ciblée sur `files`

Amendement 18 : `files`, graines 0 et 1, amorces 50 / 100 de v6–v8, recette de référence
plus l'exploration de v11b (`--copy-topk 3 --copy-explore observations --attempts 3`),
bras `closed-clean` seul ; témoin v7 (même amorce, deux tentatives, sans exploration).

| Graine | Bras | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Explorés | Jamais réussies | Promus | Jetons |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | v12 | 0,850 | 0,833 | 0,867 | 0,883 | 0,867 | 0,867 | +0,017 [+0,000, +0,050] | +0,073 | 8 | 6 | 137 | 51,6 k |
| 0 | v7 | 0,850 | 0,800 | 0,850 | 0,817 | 0,883 | 0,933 | +0,083 [+0,017, +0,150] | +0,070 | — | 4 | 122 | 49,9 k |
| 0 | oracle v7 | 0,850 | 0,933 | 0,983 | 1,000 | 0,983 | 0,967 | +0,117 [+0,033, +0,217] | +0,066 | — | 0 | 150 | 0 |
| 1 | v12 | 0,783 | 0,783 | 0,983 | **1,000** | 1,000 | 1,000 | **+0,217** [+0,117, +0,317] | +0,073 | **3** | 0 | 134 | 51,7 k |
| 1 | v7 | 0,783 | 0,833 | 0,900 | 0,883 | 0,900 | 0,900 | +0,117 [+0,033, +0,217] | +0,070 | — | 5 | 123 | 49,3 k |
| 1 | oracle v7 | 0,783 | 0,950 | 0,950 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,065 | — | 0 | 150 | 0 |

**Verdicts v12 (H16).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H16 (i)** explorés ≥ 10 par graine | **échoue** | 8 (graine 0), 3 (graine 1) |
| **H16 (ii)** gain ≥ v7 par graine, intervalle excluant zéro | **échoue** sur la graine 0 | +0,017 [+0,000, +0,050] contre +0,083 ; graine 1 : +0,217 [+0,117, +0,317] contre +0,117, passe |
| **H16 (iii)** rendement moyen ≥ 0,8 | **échoue** | 0,70 (+0,117 / +0,167) |
| **H16** | **échoue** | |

**Diagnostic, graine 0, checkpoint final, banc 7.** Trois échecs : deux sont le mode (e)
(`con_0.txt` et `acon_2.txt` notés pour `beacon_0.txt` et `beacon_2.txt`), un une note
vide. Sur les deux premiers, le bon départ (`be`, premier jeton après le saut de ligne :
`beacon` se découpe en `be`, `a`, `con`) est au **rang 11 et 15** du pointeur
(p = 0,016 et 0,002), derrière `con`, `a`, `":` et `"list_f` : la masse est *à
l'intérieur* du nom, et un tirage parmi les 3 meilleurs ne l'atteint jamais. C'est la
borne du mécanisme de v11b : il fabrique la contradiction quand la bonne valeur est au
second rang (mode (f), `lookup`), pas quand elle est au onzième (mode (e), `files`).
L'amendement 19 restreint le tirage aux débuts de mots — v13.

**La graine 1 n'est pas un effet de l'exploration.** Trois épisodes explorés sur 134
promus (2 %) ne portent pas un gain de +0,217 ; la troisième tentative ajoute 11 lignes
promues en cinq tours à jetons presque égaux (+5 %), et le reste est la variance d'un
run — deux bancs de 30 tâches, une génération stochastique. On rapporte l'écart avec v7
tel quel et sans cause établie ; il dit surtout que sur `files` graine 1 la boucle
fermée peut rejoindre l'oracle (1,000 dès le tour 3, aucune tâche jamais réussie), et
que la mesure du rendement à ±0,1 demande plus de graines que ce processeur n'en donne.

## 18. Pilote v13 : n'explorer que les débuts de mots

Amendement 19 : v12 plus `--copy-boundaries explore` (le tirage exploratoire ne considère
que les positions qui ouvrent un mot ou une observation) ; `files`, graines 0 et 1, mêmes
amorces, bras `closed-clean` seul.

| Graine | Bras | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Explorés | Jamais réussies | Jetons |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | v13 | 0,850 | 0,817 | 0,867 | 0,800 | 0,850 | 0,900 | +0,050 [−0,017, +0,117] | +0,069 | **12** | 5 | 55,9 k |
| 0 | v12 | 0,850 | 0,833 | 0,867 | 0,883 | 0,867 | 0,867 | +0,017 [+0,000, +0,050] | +0,073 | 8 | 6 | 51,6 k |
| 0 | v7 | 0,850 | 0,800 | 0,850 | 0,817 | 0,883 | 0,933 | +0,083 [+0,017, +0,150] | +0,070 | — | 4 | 49,9 k |
| 1 | v13 | 0,783 | 0,900 | 0,967 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,076 | 2 | 0 | 48,0 k |
| 1 | v12 | 0,783 | 0,783 | 0,983 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,073 | 3 | 0 | 51,7 k |
| 1 | v7 | 0,783 | 0,833 | 0,900 | 0,883 | 0,900 | 0,900 | +0,117 [+0,033, +0,217] | +0,070 | — | 5 | 49,3 k |

**Verdicts v13 (H16 et H17).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H16 (i)** explorés ≥ 10 par graine | **échoue** sur la graine 1 | 12 (graine 0, passe), 2 (graine 1) |
| **H16 (ii)** gain ≥ v7, intervalle excluant zéro | **échoue** sur la graine 0 | +0,050 [−0,017, +0,117] contre +0,083 ; graine 1 : +0,217 contre +0,117, passe |
| **H16 (iii)** rendement moyen ≥ 0,8 | passe, à la limite | 0,80 (+0,133 / +0,167) |
| **H17** `beacon_0` et `beacon_2` réussies au tour 5 (banc 7) | **échoue** | `beacon_0` oui, `beacon_2` non (`acon_2.txt`) |
| **H16**, **H17** | **échouent** | |

**Diagnostic, graine 0, checkpoint final, banc 7.** Trois échecs : `arnet_2.txt` pour
`garnet_2.txt`, `acon_2.txt` pour `beacon_2.txt`, une note vide. Sur les deux premiers, le
bon départ est maintenant au **rang 2 et 3** du pointeur (p = 0,008 et 0,025 ; rang 1 et 2
parmi les débuts de mots), contre 11 et 15 en v12 : les douze épisodes contradictoires
l'ont fait remonter sans encore le faire passer devant `arnet_` (0,992) et `acon_` (0,836).
Le mécanisme atteint le mode (e) ; cinq tours ne suffisent pas à le retourner sur cette
graine. Le premier candidat *hors* observation parmi les débuts de mots est le mot du but
(`saff`, 0,134) : restreindre l'argmax lui-même aux frontières (`always`) donnerait la
mauvaise réponse ici ; ce n'est pas un a priori à adopter sans la restriction aux
observations, et il n'est pas testé.

**La graine 1 se répète.** v12 et v13, deux runs à une variable près, atteignent tous deux
1,000 au tour 3 avec 2 et 3 épisodes explorés — v7, à deux tentatives, plafonnait à 0,900.
L'exploration n'explique pas cet écart ; ce qui distingue ces deux runs de v7 est la
**troisième tentative** : 30 épisodes promus par tour dès le tour 3 (tout le tour résolu)
contre 25–27. Une hypothèse à une variable, pré-enregistrée à l'amendement 21 : la
troisième tentative seule, sans exploration.

## 19. Pilote v10c : deux familles dans une boucle, quatre bras

Amendement 20 : graine 0, amorce mixte 75 / 150 par famille (`lookup` 0,367, tous canoniques ;
`files` 1,000, jugé sur ce qu'il garde), recette de référence sans exploration, 30 tâches
de chaque famille par tour, 60 pas pour tous les bras, banc 2 × 30 par famille. Quatre bras
à partir du même checkpoint.

| Bras | Tours sur | `lookup` t0 → t5 | `files` t0 → t5 (pire tour) | Gain union [IC 95 %] | Δ BPB | Promus | Heures |
|---|---|---:|---:|---:|---:|---:|---:|
| `closed-clean` mixte | les deux | 0,367 → 0,350 | 1,000 → 1,000 (0,983) | −0,008 [−0,025, +0,000] | +0,028 | 199 | 0,36 |
| `closed-clean` mono `lookup` | `lookup` | 0,367 → 0,350 | **1,000 → 0,000** dès t1 | −0,508 [−0,600, −0,417] | +0,024 | 47 | 0,36 |
| `closed-clean` mono `files` | `files` | **0,367 → 0,000** dès t1 | 1,000 → 1,000 | −0,183 [−0,258, −0,117] | +0,023 | 150 | 0,30 |
| `oracle` mixte | les deux | 0,367 → **1,000** | 1,000 → 0,967 (0,917) | +0,300 [+0,208, +0,392] | +0,023 | 300 | 0,26 |

Tâches `lookup` résolues par tour, bras mixte : 8, 8, 10, 9, 14 sur 30 (mono `lookup` : 8, 8,
10, 10, 15) ; bras mixte, `lookup` par tour : 0,333, 0,350, 0,350, 0,350, 0,350 ; oracle :
0,483, 0,917, 0,867, 0,917, 1,000.

**Verdicts v10c (H14, amendement 20).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H14 (i)** la boucle mixte apprend `lookup` et garde `files` | **échoue** | `lookup` −0,017 [−0,050, +0,000] ; `files` gardé (perte 0,000) |
| **H14 (ii)** pas d'interférence à compute égal : somme mixte ≥ max(mono) | passe | −0,017 contre −1,017 (mono `lookup`) et −0,367 (mono `files`) |
| **H14 (iii)** pas d'oubli par omission | **échoue** | `files` 1,000 → 0,000 (mono `lookup`) ; `lookup` 0,367 → 0,000 (mono `files`), dès le tour 1 |
| **H14 (iv)** rendement union ≥ 0,6 | **échoue** | −0,03 (−0,008 / +0,300) |
| **H14** | **échoue** | |

Trois choses, nettes.

1. **L'oubli par omission est total, symétrique et immédiat.** Soixante pas (taux ÷ 4,
   rejeu 0,5 du corpus de base) sur un bassin qui omet une famille apprise la font tomber
   à zéro au premier tour, et elle ne revient pas ; le rejeu du corpus de base ne la
   protège en rien, puisque ce corpus ne contient pas d'épisodes. Le bassin mixte, qui
   garde les épisodes promus de chaque famille, la protège entièrement (0,983 au pire).
   Pour une boucle à plusieurs familles, la règle est mécanique : **chaque tour s'entraîne
   sur le bassin cumulé de toutes les familles apprises**, jamais sur celui du tour.
2. **Le bassin mixte n'interfère pas, mais n'aide pas.** `lookup` fait la même chose dans le
   bassin mixte (150 lignes `files` pour 47 `lookup`) que seul : rien. Ce n'est pas la
   dominance de `files` qui bloque `lookup`, c'est le régime de §14 sur une graine plus
   basse — 8 tâches résolues par tour, toutes du même type que celles que la politique
   sait déjà faire, et rien qui la contredise. L'oracle, qui reçoit les trente trajectoires
   par tour, porte `lookup` de 0,367 à 0,917 en deux tours à partir des mêmes poids.
3. **Ce que le bras mixte ne peut pas mesurer**, c'est le gain de `files` : saturé à
   l'amorce. La question « le bassin mixte apprend-il les deux ? » reste ouverte sur cette
   famille de tâches à 7 M ; ce que ce pilote établit, c'est qu'il garde ce qu'il sait et
   qu'il n'apprend pas plus qu'une boucle seule.

Variable suivante (amendement 22, v10d) : le même bras mixte avec la reprise qui explore de
§16, la seule mécanique qui ait fait bouger `lookup` sur une graine bloquée. Si `lookup` y
progresse alors que `files` reste gardé, la boucle multi-familles a sa recette ; sinon la
graine 0 de `lookup` est hors de portée de la boucle fermée à 7 M, et on le dira.

## 20. Pilote v14 : la troisième tentative seule

Amendement 21 : v7 plus `--attempts 3`, sans exploration ; `files`, graines 1 et 0, amorces
50 / 100, bras `closed-clean` seul.

| Graine | Bras | t0 | t1 | t2 | t3 | t4 | t5 | Gain [IC 95 %] | Δ BPB | Explorés | Jamais réussies | Promus |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | v14 (3 tentatives, sans exploration) | 0,783 | 0,767 | 0,883 | 0,867 | 0,850 | 0,883 | +0,100 [+0,017, +0,183] | +0,072 | — | 6 | 124 |
| 1 | v12 / v13 (3 tentatives, exploration) | 0,783 | 0,783 / 0,900 | 0,983 / 0,967 | 1,000 | 1,000 | 1,000 | +0,217 [+0,117, +0,317] | +0,073 / +0,076 | 3 / 2 | 0 | 134 / 139 |
| 1 | v7 (2 tentatives) | 0,783 | 0,833 | 0,900 | 0,883 | 0,900 | 0,900 | +0,117 [+0,033, +0,217] | +0,070 | — | 5 | 123 |
| 0 | v14 | 0,850 | 0,817 | 0,833 | 0,817 | 0,867 | 0,850 | +0,000 [−0,050, +0,050] | +0,075 | — | 8 | 123 |

**Verdict (H18).** **Échoue** : graine 1 +0,100, jamais 1,000 (critère : ≥ +0,200 et 1,000 au
plus tard au tour 4). La troisième tentative seule fait ce que faisaient deux tentatives ;
les deux runs qui ont atteint 1,000 au tour 3 (v12, v13) sont donc les deux runs avec
exploration, à 2 et 3 épisodes explorés près. Trois runs contre deux, une mesure par
condition : c'est compatible avec « ces deux ou trois épisodes contradictoires ont suffi »
et avec la variance d'un banc de 60 tâches ; on ne tranche pas ici. Ce que ce pilote
ferme, c'est l'explication la plus économe (le budget de reprise), et la ligne des pilotes
CPU sur `files` : à ±0,1 par mesure, la question suivante coûte plus de graines que ce
processeur n'en donne.

## 21. Pilote v10d : le bassin mixte avec la reprise qui explore

Amendement 22 : le bras mixte de v10c (graine 0, amorce 75 / 150, `lookup` 0,367 et `files`
1,000) avec l'exploration de §16 (`--copy-topk 3 --copy-explore observations --attempts 3`).

| Bras | `lookup` t0 → t5 | `files` t0 → t5 | Gain `lookup` [IC 95 %] | Explorés `lookup` par tour | Résolues `lookup` par tour | Δ BPB | Jetons | Heures |
|---|---:|---:|---:|---|---|---:|---:|---:|
| v10d mixte + exploration | 0,367 → 0,383 (0,333, 0,267, 0,350, 0,350) | 1,000 → 1,000 | +0,017 [−0,033, +0,083] | 6, 0, 0, 0, 0 | 13, 8, 8, 10, 14 | +0,026 | 125 k | 0,40 |
| v10c mixte (témoin) | 0,367 → 0,350 | 1,000 → 1,000 | −0,017 [−0,050, +0,000] | — | 8, 8, 10, 9, 14 | +0,028 | 101 k | 0,36 |
| oracle mixte v10c | 0,367 → 1,000 | 1,000 → 0,967 | +0,633 [+0,500, +0,750] | — | 30 × 5 | +0,023 | 0 | 0,26 |

**Verdicts v10d (H19).**

| Hypothèse | Verdict | Chiffre |
|---|---|---|
| **H19 (i)** explorés `lookup` ≥ 10 | **échoue** | 6, tous au tour 1 |
| **H19 (ii)** gain `lookup` ≥ +0,100, intervalle excluant zéro | **échoue** | +0,017 [−0,033, +0,083] |
| **H19 (iii)** `files` ≥ 0,950 à chaque tour | passe | 1,000 partout |
| **H19 (iv)** rendement union ≥ 0,6 | **échoue** | 0,03 (+0,008 / +0,300) |
| **H19** | **échoue** | |

**Diagnostic, banc 7, checkpoint final.** Dix-huit échecs sur 30, **tous de la même forme** :
`read_file`, puis `done` refusé trois fois, sans jamais une `note`. Sur cette graine et cette
amorce, ce que la politique rate n'est pas le choix d'un champ, c'est **l'étape de note
elle-même** : le modèle appelle `done` dès l'observation lue. Il n'y a donc aucun span
copié à explorer sur ces tâches, et le mécanisme de §16, qui tire le départ du span à la
reprise, n'a rien à tirer : 6 épisodes contradictoires au tour 1 (des tâches où la note
avait lieu), zéro ensuite, 36 tâches de banc sur 60 jamais réussies. Le bassin mixte, lui,
tient `files` à 1,000 sur les cinq tours, avec plus de jetons (+24 %) et sans surcoût
d'oubli (+0,026 contre +0,028).

Ce que cela précise : la reprise qui explore corrige une **erreur de sélection** dans une
étape que la politique atteint (v11b) ; elle ne crée pas une étape que la politique n'émet
jamais. Le mode « `done` prématuré » est celui de la promotion canonique (§9, §15) : il se
corrige par l'amorce (plus d'exposition à cette famille) ou par un a priori de décodage
(refuser `done` avant toute note, à mesurer), pas par l'exploration du pointeur. Fin de la
ligne v10 : la boucle multi-familles garde ce qu'elle sait, n'apprend pas plus qu'une boucle
seule, et ses échecs sont ceux de l'amorce.

## 22. Programme 3 : l'échelle de calibration du proposeur, épuisée

Pré-enregistrement : docs/33, amendements 1 à 8. **Date :** 2026-09-22. Base de code :
`claude/prophet-program-3-calibration-eadcij`. Poids : le premier run 7 M **reconstruit**
(amendement 7), même recette que docs/09 sur le texte de cette machine (7 497 documents de
prose, 2 940 de code, 213 tenus à l'écart), 1 172 pas, 40 min ; bits par octet 4,426 →
2,082 sur son propre jeu tenu à l'écart (docs/09 : 4,348 → 2,184 sur le sien ; les deux
jeux diffèrent). Premier temps d'amorce (100 trajectoires parfaites, 200 pas), entraîné
une fois : banc du générateur 0,667 (canonique 0,667), banc hors distribution 0,267.
Chaque barreau : second temps sur une copie de ce premier temps, puis les deux bancs et
une sonde de 30 propositions échantillonnées ; graine 0.

| Second temps (pas / propositions) | Solveur (canonique) | Hors distribution | Sonde, grammaire de l'amendement 7 : malformées / invalides / **valides** | Sonde, grammaire corrigée (amendement 8) : malformées / invalides / **valides** |
|---|---:|---:|---:|---:|
| 50 / 50 | 0,883 (0,883) | 0,550 | 30 / 0 / **0** | 27 / 3 / **0** |
| 100 / 50 | 0,900 (0,883) | 0,600 | 16 / 14 / **0** | 9 / 21 / **0** |
| 100 / 25 | 0,850 (0,850) | 0,417 | 15 / 15 / **0** | 7 / 23 / **0** |
| 200 / 100 | 0,950 (0,950) | 0,417 | 3 / 20 / **7** | 0 / 21 / **9** |

Les deux échelles ont les mêmes poids à chaque barreau (perte finale du second temps
identique au dernier chiffre : l'entraînement ne décode rien) ; les scores des
bancs y sont identiques aussi : à ces poids, la correction ne change rien au solveur. Compute :
≈ 2 h 20 CPU en tout, premier run compris.

**Verdict (règle de l'amendement 1).** Aucun barreau ne satisfait les deux conditions. Le
meilleur, 200 / 100 sous la grammaire corrigée, donne 9 propositions valides sur 30
(0,30 < 0,5), et son solveur à 0,950 n'est pas strictement sous 0,95. **Le proposeur ne
démarre pas à 7 M.** H20 au tour 0 vaut au mieux 0,30 ; H25 échouerait aussi : aucune des
16 propositions valides des deux échelles n'a un nombre de champs ou une clé absents de
l'amorce. Les barreaux de la session précédente, sur d'autres poids et connus seulement
par ses messages (**non reproduits**), disaient la même chose : 0 valide sur trois
barreaux (50 / 50 : 7 / 23 / 0 ; 100 / 50 : solveur 0,733, 17 / 13 / 0 ; 100 / 25 :
solveur 0,617, hors distribution 0,417, 8 / 22 / 0 ; 200 / 100 : perdu).

**Lecture, depuis les propositions enregistrées (`samples`).**

1. **Un défaut de décodage, réel mais pas décisif** (amendement 8). Sous l'ancienne
   grammaire, les 30 malformées du premier barreau ont toutes un caractère de contrôle
   brut dans une chaîne (27 dans la valeur `file`) : le span était mort dès ce jeton. La
   correction les supprime toutes (0 dans l'échelle rejouée) et fait passer 3 à 8
   malformées par barreau en invalides ou en valides, dont deux valides de plus au dernier
   barreau. Elle n'ouvre aucun barreau.
2. **La forme s'apprend par étapes, le contenu non.** À 50 pas, la valeur `file` se
   prolonge par le but du solveur (`… .json and note the value of the field year, then
   finish.`, 18 spans sur 30) et par des fragments du corpus (licences, tableaux). À 100
   pas, l'appel se ferme le plus souvent, mais les 44 invalides de l'échelle rejouée ont toutes pour nom
   de fichier `}},`, la fermeture d'appel du solveur glissée dans la valeur, et le contenu
   est décalé d'un champ : `keys` reçoit une année, `values` les noms de clés, `ask` la
   liste des clés. À 200 pas et 100 propositions, fichier et clés sont justes (toujours
   `city,year,code`) ; les erreurs restantes sont aux **bords des listes** : la dernière
   valeur déborde (`jasper.json`, 13 invalides sur 21) et `ask` garde la virgule de la
   liste (`year,`, 14 sur 21). Les 9 valides sont des combinaisons nouvelles du
   vocabulaire du générateur (aucune ne recopie une tâche de l'amorce), toutes à trois
   champs, avec ses clés.
3. **Le solveur monte avec le second temps**, de 0,667 à 0,85–0,95 sur son banc et de
   0,267 à 0,42–0,60 hors distribution : propositions et trajectoires du second temps
   l'entraînent bien. Au dernier barreau, il sature le haut de la fenêtre de calibration.

**Ce que cela dit, et ce que cela ne dit pas.** Les deux lectures de docs/34 §4 ne sont
pas entièrement séparées. La validité croît avec l'entraînement du proposeur (0, 0, 0,
puis 9 valides), ce qui donne en partie raison à la lecture (a), « pas assez de pas ni de
propositions » ; mais le barreau qui ouvre enfin le proposeur ferme la fenêtre du
solveur : à 7 M, sur cette échelle, les deux conditions de la règle ne se recouvrent pas.
Ce qui résiste est la partie du format la plus dure pour un petit modèle : deux listes
séparées par des virgules et alignées par position. Un format qui supprime cet
alignement (les champs comme un objet, ou un argument par paire) changerait la grammaire
des spécifications : piste à pré-enregistrer, non mesurée ici. Le programme 3 passe à
l'A100 après le programme 2 à 375 M, comme prévu par docs/33 §4, avec la même règle de
calibration ; la correction de la grammaire vaut pour tout décodage. **Fin des pilotes CPU
des programmes 2 et 3** (docs/34 §7).

## 23. Programme 3 : le format objet (H26) et la clé copiée (H27), fin du CPU

Pré-enregistrement : docs/33, amendements 9 et 10. **Date :** 2026-09-22. Mêmes poids de
départ et même premier temps que §22 : banc 0,667, remesuré à l'identique sous chaque
nouveau code. Graine 0.

**H26, les champs en objet** (`--propose-format object`) : quatre barreaux, le second temps
réentraîné sur des propositions en objet.

| Second temps (pas / propositions) | Solveur (canonique) | Hors distribution | Sonde : malformées / invalides / **valides** |
|---|---:|---:|---:|
| 50 / 50 | 0,950 (0,950) | 0,600 | 12 / 18 / **0** |
| 100 / 50 | 0,783 (0,750) | 0,450 | 1 / 29 / **0** |
| 100 / 25 | 0,900 (0,867) | 0,517 | 1 / 29 / **0** |
| 200 / 100 | **0,933** (0,900) | 0,483 | 0 / 18 / **12** |

**H26 échoue sur son critère** : au mieux 12 valides sur 30 (0,40 < 0,5). Au dernier
barreau, pourtant, le solveur est **dans** la fenêtre (0,933 < 0,95, canonique 0,900) :
seule la validité manque. Le détail champ par champ, sur les appels qui se lisent :

| Barreau | Champ | Listes (§22) | Objet |
|---|---|---:|---:|
| 100 / 50 | champs justes | 0 / 21 | **23 / 29** |
| 100 / 50 | `file` juste | 0 / 21 | 0 / 29 |
| 200 / 100 | `file` juste | 30 / 30 | 30 / 30 |
| 200 / 100 | champs justes | 14 / 30 | **30 / 30** |
| 200 / 100 | `ask` parmi les clés | — | **12 / 30** |

Le format objet lève le verrou des champs, que le format en listes ne levait pas. Au
dernier barreau, les 18 invalides n'échouent **que** sur `ask` (`yearbor`, `harbor`,
`cinder`, `}}`). Aux barreaux intermédiaires, le nom de fichier commence par la fermeture du
schéma d'outil qui précède l'appel (`]}}`, 24 sur 29 au barreau 2). À 50 pas, l'objet
`fields` recopie le schéma JSON d'un outil (`{"type":"object","properties":…}`), l'objet que
le modèle voit le plus souvent.

**H27, la clé demandée copiée** (`--propose-copy ask`, sans aucun entraînement : les quatre
points de contrôle d'H26 re-sondés). Les bancs du solveur sont identiques, barreau par
barreau (0,950, 0,783, 0,900, 0,933), comme attendu d'un changement qui ne touche que la
sonde.

| Barreau | Sonde sans copie (H26) | Sonde avec copie de `ask` (H27) |
|---|---:|---:|
| 50 / 50 | 12 / 18 / 0 | 20 / 10 / 0 |
| 100 / 50 | 1 / 29 / 0 | 0 / 30 / 0 |
| 100 / 25 | 1 / 29 / 0 | 3 / 27 / 0 |
| 200 / 100 | 0 / 18 / **12** | 0 / 24 / **6** |

**H27 échoue** et fait pire : 6 valides au lieu de 12. Le pointeur copie la **valeur** d'un
champ au lieu de sa clé (`"ask":"violet"` 8 fois, `delta`, `cinder`, `orchid`), parfois un
morceau de l'appel ou du but. C'est l'habitude qu'il a prise sur les épisodes du solveur :
copier la valeur qu'il vient de lire. Le mode (f) de §14 réapparaît, dans l'autre sens.

**Verdict, règle pré-enregistrée (amendement 10).** **Le proposeur ne démarre pas à 7 M**,
après trois échelles et deux formats : au mieux 12 propositions valides sur 30, aucune
nouvelle (H25), alors que le solveur est dans sa fenêtre. **Le CPU s'arrête pour le
programme 3.**

**Ce que cette ligne a établi, et qui vaut à toute échelle.**

1. **Deux défauts silencieux du décodage contraint**, corrigés pour tout décodage : les
   caractères de contrôle bruts et les échappements inconnus dans une chaîne
   (amendement 8), et des tableaux et objets imbriqués lus sans leur syntaxe
   (amendement 9). La règle qui en sort : **viable doit vouloir dire complétable**.
2. **La forme d'une spécification décide de ce qu'un petit modèle apprend.** À dose égale,
   les champs en objet, dans la forme que le solveur lit, passent de 0 à 23 justes sur 29 ;
   deux listes alignées par position restent à 0.
3. **Chaque champ à inventer hérite du contexte qui le précède** : la fin du schéma d'outil
   (`]}}`), le gabarit du but, l'objet le plus fréquent du prompt. Plus un proposeur est
   petit, plus ses premiers jetons appartiennent au prompt plutôt qu'à la tâche.
4. **Une référence n'est ni une valeur à tirer ni une valeur à copier telle quelle.**
   Tirée, la clé dérive (`yearbor`) ; copiée, le pointeur prend la valeur voisine. Le
   pointeur n'a jamais appris à désigner une *clé* : ses cibles d'entraînement sont des
   valeurs, lues puis notées. Cibler des clés à l'entraînement est une piste pour l'échelle
   A100 ; elle n'est pas mesurée ici.
5. **La validité monte avec la dose d'entraînement du proposeur** (listes : 0, 0, 0, 9 ;
   objet : 0, 0, 0, 12), et le solveur monte avec elle vers le haut de sa fenêtre. À 7 M,
   les deux conditions de la règle ne se recouvrent qu'à peine, au dernier barreau.

## 24. `done` prématuré : un défaut de décodage, pas une étape que la politique n'émet pas (H28)

Pré-enregistrement : docs/31, amendement 24. **Date :** 2026-09-22. Point de contrôle : le
premier temps de l'amorce reconstruite (§22), `lookup`, bancs gloutons des graines 7 et 11,
sous `no_repeat_action` comme tous les pilotes depuis v7. Un seul changement : l'option
`no_repeat_emitted`, qui interdit au pas suivant l'action émise plutôt que le substitut que
la porte a inscrit.

| Mesure, à poids égaux | `no_repeat_action` (tel que les pilotes l'ont lu) | + `no_repeat_emitted` |
|---|---:|---:|
| Banc `lookup` | 0,667 | **0,967** (0,933 et 1,000) |
| Succès canonique | 0,667 | 0,667 |
| Banc hors distribution | 0,267 | **0,617** |
| Échecs « `done` prématuré » (`read_file`, `done` refusé ×3, aucune note) | **16** sur 60 | **0** |
| Autres échecs | 2 mauvaises notes, 2 notes justes trop tardives | 2 notes du mauvais champ, le budget de 4 pas épuisé |

**H28 passe** (critère : ≥ 0,767 et moins de 16 échecs de ce mode). Le succès canonique ne
bouge pas : une réussite qui passe par un `done` refusé n'est pas canonique, et c'est
elle que la promotion garde (§9). Le changement ne touche donc pas ce que la boucle
apprend ; il touche ce que le banc mesure et ce que la génération peut trouver.

**Ce que cela révise.** Le mode « `done` prématuré » de §21 (v10d : 18 échecs sur 30, tous
de cette forme) a été lu comme « une étape que la politique n'émet jamais », que
l'exploration du pointeur ne pouvait pas créer. Sur les poids reconstruits, la même forme
disparaît entièrement dès que le décodeur n'autorise plus la répétition immédiate d'un
`done` refusé : la politique émet la note au pas suivant quand on lui en laisse la place.
Les chiffres de v7 à v14 et de v10c/v10d restent tels qu'enregistrés (ils ont été mesurés
avec ce défaut, sur d'autres poids) ; leur lecture du mode `done` prématuré est à prendre
avec cette réserve. L'option entre dans la recette A100 (docs/31, amendement 25).
