# Joostar megoldások

1. **Hány olyan partit játszottak 2023.10.12. és 2024.02.19. között, amikor a vesztes félnek legalább 3 pontnyi minőség hátránya volt csak „standard” alapállású partikat figyelembe véve? --> eredmény: egyetlen szám. Kiegészítés: használjuk a szokásos pontszámítást a figurák értékéhez, miszerint minden gyalog 1-et, minden futó és huszár 3-at, minden bástya 5-öt és minden vezér 9-et ér**  
   Válasz: Partik száma: 3023254

2. **Bal oldali lóval ütéses meccsek: Mennyivel nagyobb arányban nyertek azok a  játékosok akik ütöttek a bal oldali lóval azokhoz képest, akik nem ütöttek bal oldali lóval a játék során (bal oldali ló alatt a fehérnek a B1-es, feketének a G8-as mezőről induló ló számít)?**  
   Válasz: Győzelmi arány különbség a bal lóval ütők és nem ütők között: 0.05853086446188427

3. **A 10 perces mérkőzések során hány esetben veszítette el világos a sáncoláshoz való jogát a játszma első három teljes lépésén (az első 6 fél-lépésen) belül?**  
   Válasz: Elvesztett sáncolási jogok száma: 621

4. **Bástyák: Az összes fehér és fekete bástya lépése között mennyi a különbség megtett hosszban?**  
   Válasz: Lépéshosszkülönbség fehér és fekete bástyák között (fehér - fekete): 28467732

5. **Hány alkalommal ért véget egy játszma háromszori lépésismétléssel olyan játékosok részvételével, akiknek van egy olló emoji van a nevében?**  
   Válasz: Háromszori lépésismétléssel véget érő partik száma: 2,592

6. **Hány játszma végződött háromszori lépésismétlés miatt döntetlennel 2024.03.12. és 2024.11.19. között csak „standard” alapállású partikat figyelembe véve? --> eredmény: egyetlen szám**  
   Válasz: Háromszori lépésismétlések száma: 166,401

7. **A versenygyőztesek által megnyert játszmákban a mattadás pillanatában átlagosan hány világos vezér tartózkodott a táblán?**  
   Válasz: Matt pillanatában a táblán lévő fehér vezérek átlagos száma: 0.6450

8. **Hány alkalommal végződött döntetlennel a játszma március 20-án olyan esetekben, ahol a mérkőzést lezáró utolsó lépés egy gyalog vezérré történő átváltozása volt?**  
   Válasz: 72 alkalommal végződött így a játszma!

9. **Melyik felhasználó(k) szenvedte(k) el a legtöbb vereséget időtúllépés miatt olyan játszmákban, ahol a mérkőzés kezdetén élt(ek) a 'Berserk' opcióval? (holtverseny esetén abc szerint első 10)**  
   Válasz:
```
======================================================================
  A LEGTÖBB BERSERK-IDŐTÚLLÉPÉSES VERESÉGET SZENVEDTÉK:
  -> 5547 vereséggel
----------------------------------------------------------------------
  1. zelkovahi
======================================================================
```

10. **Logit regresszió: Függő változó: nyert-e an meccs, Magyarázó változók: A játékos ütéseinek száma, a játékos színe,  átlagosan mennyi időt használ fel a játékos egy lépésre. Mik ennek a modellnek a becsült paraméterei?**  
    Válasz: 
```
 (Játékos szintű - Győzelem Valószínűsége)
 Függő változó : is_win (1=nyert, 0=nem)
 Konstans (Intercept) : -0.437137
 total_captures együt : 0.068560
 is_white együttható  : 0.144702
 avg_time_per_move    : -0.094959
```

11. **Feladás: Ki az, aki a legtöbbször adta fel a partit, hányan vannak akik nem adták fel soha, és hányan tartoznak a mediánba a feladások számát tekintve?**  
    Válasz: 
```
----------------------------------------------------------------------
  1. A LEGTÖBBSZÖR adta fel    : siddeep (13,564 alkalommal)
  2. SOHA nem adták fel        : 213,888 játékos
  3. Feladások MEDIÁNJA        : 2
  -> Mediánba eső játékosok    : 84,740 fő
----------------------------------------------------------------------
```

12. **Melyik az a legnagyobb „kör” ahol körbeverés történt egy adott naptári éven belül közép-európai időzóna szerint csak „standard” alapállású partikat figyelembe véve? --> eredmény: naptári év és játékosnevek listája a körbeverés sorrendjében, az időben első parti győztesének nevével kezdve. kiegészítés: ha több azonos méretű kör is létezik, bármelyik legnagyobb kör elfogadható válaszként**  
    Válasz: - 

13. **Meccsenként inkább azok nyernek nagyobb arányban akik több vagy kevesebb időt használnak fel a másik játékoshoz képest?**  
    Válasz:
```
A KEVESEBB időt használó játékos nyer nagyobb arányban.
Különbség: +39.7372%
```

14. **Mely napokon (datumokon) történt legalább egy olyan játszma, ahol az eredetileg az a2 mezőn álló világos gyalog eljutott a g8 mezőre és ott átváltozott? (elso 10 datum)**  
    Válasz: Nem történt ilyen játszma a vizsgált időszakban.

15. **Gyalogos átváltás: Hányszor nem királynőre váltották a gyalogot a beéréskor, és melyik a három legnépszerűbb bábu, amire átváltottak gyalogost királynő helyett (a váltások számát is meg kell adni)? (csak azokat az eseteket kell figyelembe venni amikor egyértelmű hogy mire lett átváltva a gyalogos)**  
    Válasz:
```
Összes 'nem Királynőre' történő átváltás száma: 207847 db

A 3 legnépszerűbb bábu:
1. Bástya (R): 120443 alkalommal
2. Huszár (N): 63976 alkalommal
3. Futó (B): 23428 alkalommal
```

16. **Melyik játékosé, melyik időszakban és hány partin keresztül tartott a leghosszabb döntetlen széria csak „standard” alapállású partikat figyelembe véve? --> eredmény: játékosnév, időszak (tól-ig), streak szám (ha több játékosnak azonos streak száma volt, akkor az nyert, akinek a széria utolsó partijában a legmagasabb élőpontszáma volt) Kiegészítés: leghosszabb alatt a lejátszott játszmák darabszám és nem a naptári időszakot értjük. Folyamatban lévő streak is versenybe szállhat**  
    Válasz: 
```
VÉGEREDMÉNY (Játékosnév, időszak, streak szám):
chessvideworld, 2023.10.26 10:31:53 - 2023.10.26 10:51:39, 10
(A nyertes Élő-pontszáma a széria végén: 2104)
```

17. **Logit regresszió: Függő változó: Lépésekre generált dummy változó (1 ha azzal a lépéssel ütött, 0 ha nem) Magyarázó változók: meccs kezdete óta eltelt idő másodpercben, bábu színe (white=1) dummy. Mik ennek a modellnek a becsült paraméterei?**  
    Válasz: 
```
 (Lépés szintű - Ütés Valószínűsége)
 Függő változó : is_capture (1=ütött, 0=nem)
 Konstans (Intercept) : -1.190382
 elapsed_time együtth : -0.001220
 is_white együttható  : -0.019440
 ```
 
18. **Melyik játékosé, melyik időszakban és hány partin keresztül tartott a leghosszabb nyeretlenség széria csak „standard” alapállású partikat figyelembe véve? --> eredmény formátum: játékosnév, időszak (tól-ig), streak szám (ha több játékosnak azonos streak száma volt, akkor az nyert, akinek a játékosneve leghamarabb követi a „Lili” becenevet a magyar abc szabályai szerint) Kiegészítés: leghosszabb alatt a lejátszott játszmák darabszám és nem a naptári időszakot értjük. Folyamatban lévő streak is versenybe szállhat**  
    Válasz:
```
(Játékosnév, időszak, streak szám):
Varga-91, 2024.12.27 12:33:24 - 2024.12.27 14:03:41, 167
```

19. **Hány játszma végződött az 50 lépéses szabály miatt döntetlennel 2026.03.15 és 2026.10.14. között csak „standard” alapállású partikat figyelembe véve? --> eredmény: egyetlen szám**  
    Válasz: 54 db játszma végződött az 50 lépéses szabály miatt döntetlennel. 

20. **Hogyan alakult a megfigyelt években 04.21 és 05.18. között (CET szerint) a vezércsellel kezdett partik százalékos aránya az összes játszmához képest csak „standard” alapállású partikat figyelembe véve? --> eredmény: minden év megjelölt időszakára, aminél szerepel legalább 1db játszma az adatbázisban 1db arányszám (vezércsellel kezdett partik száma / összes parti) kiegészítés: vezércsellel kezdett partinak minősül minden olyan játszma, ahol az első lépések 1. d4-d5; 2. c4 volt**  
    Válasz:
```
┌──────────┬─────────────┬──────────┬────────────────┐
│ Év       ┆ Összes      ┆ Vezércsel┆ Arány          │
╞══════════╪═════════════╪══════════╪════════════════╡
│ 2024     ┆ 1070352     ┆ 37485    ┆ 3.5%           │
│ 2025     ┆ 2190611     ┆ 71980    ┆ 3.29%          │
└──────────┴─────────────┴──────────┴────────────────┘
```

21. **Az adatbázisban lévő partik közül hány olyan játszma volt, ami közép-európai idő szerint potenciálisan elhúzódhatott (volna) egyik évről a másikra a játszma time kontrollja és a megtett lépések alapján csak „standard” alapállású partikat figyelembe véve? --> eredmény: minden olyan szilveszterre, ami beleesik az adatfelvétel time range-ébe 1db szám**  
    Válasz: 
```
┌─────────────────┬──────────────────────┐
│ szilveszter éve ┆ átnyúló partik száma │
╞═════════════════╪══════════════════════╡
│ 2023            ┆ 42                   │
│ 2024            ┆ 4                    │
│ 2025            ┆ 67                   │
└─────────────────┴──────────────────────┘
```

22. **Téglalap alakú lépések: Ki volt az, aki a legtöbbször írt le egy téglalapot egy bábuval (az összes játékon leírt téglalapok összegét kell tekinteni), és mekkora volt a legnagyobb leírt téglalap a területét tekintve? Téglalapszám holtverseny esetén azt a  játékos nevét kell megadni aki ezt a téglalap számot először érte el.**  
    Válasz: -

23. **Melyik játékos(ok) ad(tak) legtöbbször sáncolással mattot? (holtverseny esetén abc szerint első 10)**  
    Válasz:
```
┌──────────────────┬────────────┐
│ győztes neve     ┆ mattok     │
╞══════════════════╪════════════╡
│ HuberVilla       ┆ 6          │
│ Fly-Low-Hit-Hard ┆ 4          │
│ RumijaBarCG21    ┆ 4          │
│ VadimirUlaov     ┆ 4          │
│ chessyriy        ┆ 4          │
│ AntonBerg        ┆ 3          │
│ BlackGambit1     ┆ 3          │
│ FakriID          ┆ 3          │
│ NEWBYRONDUPIES   ┆ 3          │
│ vova1968         ┆ 3          │
└──────────────────┴────────────┘
```

24. **A 3 perces mérkőzések körében hány alkalommal hajtott végre világos 'en passant' ütést olyan játszmákban, amelyek indiai típusú megnyitással kezdődtek? (Az ECO kód első betűje a megnyitás típusát jelöli)**  
    Válasz: Világos 15654 alkalommal ütött 'en passant'!