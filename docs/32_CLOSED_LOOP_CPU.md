# 32 — La boucle fermée à 7 M sur CPU : premiers chiffres

**Statut : quatre pilotes terminés (v1 tolérant, v2 grammaire compacte, v3 taux ÷ 4, v4
promotion canonique), trois graines retenues, bras KLPO à chaque fois. Tout chiffre ici vient de `rounds.jsonl` ; rien n'a été
retouché après coup.** Pré-enregistrement et amendements : docs/31. **Date :** 2026-09-21.
Base de code : `claude/codex-results-analysis-5jz95y`. Poids de départ : le 7 M de docs/09
(`prophet-cpu-first-run`, pas 1163), CPU 4 cœurs.

## 0. Ce qu'on sait maintenant, en cinq lignes

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

Pour l'échelle A100 (docs/31 §3), trois choses sont acquises dès maintenant : la marge de
départ se calibre avant de lancer (tour 0 seul, par graine et par famille) ; la recette par
tour doit être mesurée sur l'oubli avant tout (un planning neuf à taux plein par tour est
la cause du +0,44) ; le banc doit rapporter la part malformée à chaque tour, sans quoi les
deux modes d'échec se confondent.
