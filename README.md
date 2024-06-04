## Szemantikus keresés a cikkekben

### Megfontolások

Ahhoz, hogy nem statisztikai módszerekkel keresni tudjunk a szövegben, és értelmet nyerjen az összehasonlításuk, először matematikai formába kell őket hozni egy olyan módszerrel, ahol hasonló vagy ellentétes jelentésű, de máshogy leírt szavak között is megtaláljuk az összefüggéseket és az ellentéteket, figyelve a logikai összefüggésekre is (tagadás, és, vagy stb.). Mint kiderült, az "igaz" nem tökéletes ellentéte a "hamis" vagy a "jó" szónak a "rossz". ("A dolgozat nem lett rossz." és "A dolgozat jó lett." között is érezhetjük, hogy bár hasonlóak, más jegyre utalhatnak.
Ha az előző lépés sikerült, és képesek vagyunk cikkeket összehasonlítani a keresett kifejezésünkkel vagy egy témával, akkor már nagyon közel vagyunk akár egy ajánló rendszerhez. Szerettem volna a cikkeket kategóriákba is besorolni, és az alapján is keresni, és javaslatokat tenni. Lássunk is neki.

### Adattisztítás, transzformálás, mondatokra bontás (NLTK), tokenizálás, beágyazás sentence transformerrel

Az adatok általában nincsenek olyan formátumban vagy adatszerkezetben, amit egy modellnek közvetlenül átadhatunk, ezért azt előbb tisztítani, transzformálni, és tokenizálni kell. Az általam feldolgozott Covid-19 témájú 9000 angol cikk külön json fájlokban volt tárolva, rengeteg felesleges információval és metaadattal. Rendelkezett címmel, absztrakttal (rövid összefoglalás) és bekezdésenként magával a tartalmával. Egyből jön a dimenzióprobléma, hogy nem mindegyiknek volt absztraktja, és a bekezdések száma se egyezett meg. Használtam egy NLTK nevű API csomagot, amivel a bekezdéseket és az abstractot mondatokra bontottam (újabb dimenzió probléma, mert a mondatok száma is változó), így már tudunk a mondatok szintjén gondolkodni. Ezt már csak tokenizálni kell, és átadni a modellnek, aki leképezi a mondatokat esetünkben 768 dimenziós vektorokra. Szerencsére nem kell eltávolítani a kötőszavakat vagy írásjeleket a mondatokból nyelvi modellek esetén. Tokenizáció lényege, hogy nem szavak vagy egy karakteres betűk szintjén gondolkodunk, pl.: tudós, tudomány, hanem tokenek szintjén, ami `n` darab karaktert tartalmaz a tanító adathalmazban való gyakorisága és a maximális tokenek száma alapján. Lehet pl.: tud+ós, tud+omány, így a cos hasonlóságot már a tokenek szintjén tudjuk nézni, ahol ezek a szavak nem állnak távol egymástól, és más rokon értelmű szavakat is össze tudunk hasonlítani. (A legkisebb token egy karakter, így az sem túl nagy probléma, ha egy helytelen vagy addig nem ismert szót írunk be, ellentétben más módszerekkel.) Mivel változtattam a cikkek adatszerkezetén, ezért letároltam cikkenként különböző json fájlokban őket, és a másik mappában pedig csak a beágyazott vektorokat (ugyanolyan fájlnevekkel). A vektorokat pki-ban tároltam, ami lényegesen hatékonyabb, mint a szöveg alapú tárolás, torch tensorokat tárolok benne. A numpy tömbök bináris tárolása is szóba jött, de sajnos a dinamikus adatszerkezet miatt ez nem volt lehetséges. Az, hogy melyik dimenzió mit tartalmaz és mit jelent, nem fárasztok vele senkit, a `article_embedding.py` fájlban megtekintheti. Röviden sok dimenziós heterogén dinamikus lista.

Beletettem a repoba a 2020-03-13/comm_use_subset.tar.gz tömörített fájlt, ami kicsomagolás után tartalmazza a 9000 kizárólag angol cikk adatait json formátumban. Ugyanabban a mappában olvashatunk az adatszerkezetről. Nem szükséges kézzel kibontani, scriptból megoldom. Minden cikk beágyazását elkészítettem, és ebből 100-t töltöttem fel Githubra a tárhely végessége miatt.

### Hasonlóság keresés a cikkek között

Alapvetően azonos dimenziójú vektorizált mondatokat hasonlítunk össze a cos hasonlóság alapján. Miután betöltöttük a cikkeket és a beágyazásaikat, meg kell határozni egy keresési feltételt. Ez egy általunk leírt mondat. Ezt összehasonlítja a többi cikkel (általában a címével), sorba rendezi őket, és ajánlja a mondathoz a leghasonlóbbakat. De itt nem álltam meg, és definiáltam 20 különböző Covid-19 témával kapcsolatos kategóriát, mert milyen jó, hogyha kategóriákra lebontva kapunk különböző ajánlásokat. Természetesen a kategóriákat is vektorizálni kellett a modellel, és hasonlóságot mérni a bemeneti keresési feltételhez. Nyílvánvalóan ha rákeresek valamire, akkor van `n` darab kategória, ami közel esik hozzám, és a cikkeket is be lehet így sorolni kategóiákba, de, hogy ne mindig ugyanazoakt ajánlja, ezért az egyes kategória ajánlásainál figyelembe vettem, hogy azok mennyire állnak közel a kategóriához és a keresett mondathoz, amit megadtunk. Ha már megvannak a beágyazások a `article_embedding.py` futtatása segítségével, vagy csak leclone-oztuk a tárolót, akkor futassuk a `read_and_sugges_article.py` interaktív fájlt (pip csomagokat töltsük le). GPU ajánlott, de CPU-val is boldogul. Az utóbbi tartalmaz kommenteket, többnyire a torch beépített függvényeit és pl. sorba rendezéshez lambda function-öket használtam a feltétel precíz megadásához.

### Tesztek, felmerülő alternatív próbálkozások és eredményeik

A tárhely végessége és a depricated megoldások miatt a legtöbb fájl nem került bele a repoba. Mivel nem voltam jártas az AI területén, meghoztam azt a coinflip döntést, hogy a kettő közül melyik keretrendszert szeretném használni. Ez a torch-ra esett a C++ API, rugalmasabb függvényhasználat és az állítólag kevesebb memóriafogyasztása miatt. Több modellt is kipróbáltam a sentence transformerek közül, többnyelvűt nem nagyon akartam, mert csak angol cikkek voltak, túl nagy modelt sem, mert hiába ugyanakkora vagy hasonló a leképező vektor, nincs végtelen időnk (és a többnyelvűek rendszerint nagyobbak is vagy ugyanakkorákat tekintve kevésbé hatékonyak). A scriptek írása közben nem ágyaztam be újra a 9000 cikket mindig, csak kisebb részeket, hogy a módszerem helyes-e. Próbáltam olyat, hogy együtt tárolom a mondatokat és a beágyazásait egy közös adatszerkezetben, egyik ilyen volt a `old_embeddings_output.json` is. Vannak alternatív módszerek mondatok vektorizálására mint a Word2Vec vagy a GloVe. Én a Word2Vec-et kipróbáltam, nem volt rossz, de rosszul kezelte, ha egy szó nem volt benne vagy ha a mondatok ellentétes jelentésüek voltak például egy tagadószóval, így ezt elvetettem. Mivel bekezdésekre voltak bontva az eredeti json fájlok, próbáltam azt beágyazni a sentence transformerrel. Hát igen, SENTENCE (mondat) transformer (ki gondolta volna, hogy bekezdés beágyazásánál elveszik a hasonlóság valahova a harmadik tizedesjegyre, de egy próbát megért). Szóval NLTK könyvtárával felbontottam mondatokra a cikkek minden részét, és végeztem pár tesztet és kiíratást. Majdnem tökéletesek lettek, jól kezelte például a tizedesjegyeket, zárójeleket, vesszőket stb. Az NLTK nem nyelvi model, inkább statisztikai módszereken alapul, érdemes használni. Eloszlásgrafikont nem rajzoltam, mert nem volt grafikus felületem. Ezek rendre a `test_nltk.py` és `old_nltk_articles.py` fájlok. A szövegek a tisztítás után megtekinthetőek a `jsonschema-text.json` fájlban ez körülbelül a negyedik séma volt, mire ez lett (a vektorok pki-ben vannak, mint kiderült, nem érdemes szövegként tárolni őket, és numpy tömbök nem elég dinamikusak). A `info_from_articles.py` futtatásával információt kaphatunk a cikkekről, hogy például hány bekezdésből állnak, mi a címük etc. A `torch_info_vectors.py` csak a beágyazásokról mutat egy példát. Mint mondtam voltak olyanok is, hogy cikkekhez, vagy kategóriákhoz is hasonlítottam a bemenetet, ezeket ki lehet próbálni a `topic_similarity.py`, `sugges_articles_and_topics.py` és `sentence_to_article_similarity.py` fájlok futtatásával. A végleges persze a  `read_and_sugges_article.py` lett, ami ha a tárolót letöltöttük, azonnal futtatható.

### Program használata

Töltsük le a tárolót, és ha nem csak az első 100 cikk beágyazása kell nekünk, és van elég jó GPU-nk, akkor futassuk a `article_embedding.py` fájlt. A `first100article_embedding.py` az első százat csinálja meg újra, ami már része a reponak.
Futassuk a `read_and_sugges_article.py` interaktív fájlt. Kész vagyunk, a parancssor interaktív, egy lehetséges kimenete (vége megvágva):

```bash
python3 read_and_sugges_article.py 
Loading the model...
Model is ready.

Enter a sentence: immunity to pathogens taught by specialized human dendritic cell subsets

Top 10 similar articles:
Article 2: immunity to pathogens taught by specialized human dendritic cell subsets (Similarity: 1.0000)
Article 29: A Systems Immunology Approach to Plasmacytoid Dendritic Cell Function in Cytopathic Virus Infections (Similarity: 0.6371)
Article 89: Rabies-based vaccine induces potent immune responses against Nipah virus (Similarity: 0.4184)
Article 56: Unique Epitopes Recognized by Antibodies Induced in Chikungunya Virus-Infected Non-Human Primates: Implications for the Study of Immunopathology and Vaccine Development (Similarity: 0.3780)
Article 94: The differentiated airway epithelium infected by influenza viruses maintains the barrier function despite a dramatic loss of ciliated cells OPEN (Similarity: 0.3740)
Article 12: Rational Design of a Live Attenuated Dengue Vaccine: 29-O-Methyltransferase Mutants Are Highly Attenuated and Immunogenic in Mice and Macaques (Similarity: 0.3508)
Article 30: viruses Viruses and Autoimmunity: A Review on the Potential Interaction and Molecular Mechanisms (Similarity: 0.3482)
Article 72: Harnessing host-virus evolution in antiviral therapy and immunotherapy (Similarity: 0.3415)
Article 99: A Recombinant Influenza A/H1N1 Carrying A Short Immunogenic Peptide of MERS-CoV as Bivalent Vaccine in BALB/c Mice (Similarity: 0.3411)
Article 39: Experimental infection of dromedaries with Middle East respiratory syndrome-Coronavirus is accompanied by massive ciliary loss and depletion of the cell surface receptor dipeptidyl peptidase 4 OPEN (Similarity: 0.3410)

Top 3 similar categories:
Category 2: Herd immunity (Similarity: 0.4198)
Category 18: Vaccine hesitancy (Similarity: 0.3244)
Category 12: Vaccine efficacy (Similarity: 0.3226)

Recommended articles per category:

Category: Herd immunity
Article 2: immunity to pathogens taught by specialized human dendritic cell subsets (Avg Similarity: 0.7099)
Article 29: A Systems Immunology Approach to Plasmacytoid Dendritic Cell Function in Cytopathic Virus Infections (Avg Similarity: 0.5285)
Article 89: Rabies-based vaccine induces potent immune responses against Nipah virus (Avg Similarity: 0.4191)

Category: Vaccine hesitancy
Article 56: Unique Epitopes Recognized by Antibodies Induced in Chikungunya Virus-Infected Non-Human Primates: Implications for the Study of Immunopathology and Vaccine Development (Avg Similarity: 0.3512)
Article 94: The differentiated airway epithelium infected by influenza viruses maintains the barrier function despite a dramatic loss of ciliated cells OPEN (Avg Similarity: 0.3492)
Article 12: Rational Design of a Live Attenuated Dengue Vaccine: 29-O-Methyltransferase Mutants Are Highly Attenuated and Immunogenic in Mice and Macaques (Avg Similarity: 0.3376)

Category: Vaccine efficacy
Article 30: viruses Viruses and Autoimmunity: A Review on the Potential Interaction and Molecular Mechanisms (Avg Similarity: 0.3354)
Article 72: Harnessing host-virus evolution in antiviral therapy and immunotherapy (Avg Similarity: 0.3320)
Article 99: A Recombinant Influenza A/H1N1 Carrying A Short Immunogenic Peptide of MERS-CoV as Bivalent Vaccine in BALB/c Mice (Avg Similarity: 0.3318)

Would you like to read from the top n similar articles or from top categories? (Enter 'n' or 'categories'): categories
Enter the category number from the top categories: 1
Enter the article number within the category 'Herd immunity': 1


Title: Immunity to pathogens taught by specialized human dendritic cell subsets


Abstract:
Dendritic cells (DCs) are specialized antigen-presenting cells (APCs) that have a key role in immune responses because they bridge the innate and adaptive arms of the immune system.
They mature upon recognition of pathogens and upregulate MHC molecules and costimulatory receptors to activate antigen-specific CD4 + and CD8 + T cells.
It is now well established that DCs are not a homogeneous population but are composed of different subsets with specialized functions in immune responses to specific pathogens.
Upon viral infections, plasmacytoid DCs (pDCs) rapidly produce large amounts of IFN-α, which has potent antiviral functions and activates several other immune cells.
However, pDCs are not particularly potent APCs and induce the tolerogenic cytokine IL-10 in CD4 + T cells.
In contrast, myeloid DCs (mDCs) are very potent APCs and possess the unique capacity to prime naive T cells and consequently to initiate a primary adaptive immune response.
Different subsets of mDCs with specialized functions have been identified.
In mice, CD8α + mDCs capture antigenic material from necrotic cells, secrete high levels of IL-12, and prime Th1 and cytotoxic T-cell responses to control intracellular pathogens.
Conversely, CD8α − mDCs preferentially prime CD4 + T cells and promote Th2 or Th17 differentiation.
BDCA-3 + mDC2 are the human homologue of CD8α + mDCs, since they share the expression of several key molecules, the capacity to cross-present antigens to CD8 + T-cells and to produce IFN-λ.
However, although several features of the DC network are conserved between humans and mice, the expression of several toll-like receptors as well as the production of cytokines that regulate T-cell differentiation are different.
Intriguingly, recent data suggest specific roles for human DC subsets in immune responses against individual pathogens.
The biology of human DC subsets holds the promise to be exploitable in translational medicine, in particular for the development of vaccines against persistent infections or cancer.

Paragraph 1:
iNTRODUCTiON Human beings are constantly exposed to a myriad of pathogens, including bacteria, fungi, and viruses. These foreign invaders or cohabitants contain molecular structures that are sensed by the innate immune system, which mounts a first-line defense and also activates a pathogen-specific, adaptive immune response. The adaptive immune system is composed of B cells that produce specific antibodies, CD8 + T cells that can kill pathogen-infected cells, and CD4 + T cells that produce effector cytokines and coordinate the immune response. T cells express antigen receptors (T-cell antigen receptors, TCR) that recognize specific peptides presented on MHC molecules. CD8 + T cells recognize peptides presented by MHC class-I molecules that are ubiquitously expressed, whereas CD4 + T cells are activated by peptide-MHC class-II complexes, which are largely restricted to antigen-presenting cells (APCs). Dendritic cells (DCs) can express very high levels of MHC and costimulatory molecules, and it is generally accepted that they are the relevant cells to induce the activation ("priming") of antigen-specific "naive" T cells (1, 2) and induce their differentiation into various types of effector T cells.

Paragraph 2:
The elimination or containment of different types of pathogens requires dedicated classes of adaptive immune responses (3) . Thus, pathogens like viruses or intracellular bacteria require CD4 + and CD8 + T cells that produce IFN-γ and kill infected cells (Th1 and CTL, respectively). IL-12 is the critical cytokine that induces this type of response, but IL-12 production by DC is tightly controlled and requires several stimuli derived from pathogens and from CD4 + helper T cells (4) (5) (6) (7) (8) (9) . Conversely, extracellular bacteria and fungi require a different type of response that can be mediated by Th17 cells (10) (11) (12) . These effector cells are induced by proinflammatory cytokines produced by DC and macrophages (13) and attract neutrophils that in turn phagocytose extracellular bacteria (14) . A third type of effector response is the Th2 response, which is required to expel extracellular parasites such as helminths by activating eosinophils and basophils and by inducing antibodies of the IgE class (15) . IL-4 is the critical cytokine that induces this response (16) , but IL-4 is normally not produced by DC (17, 18) . Finally, these different effector responses have to be controlled by specialized regulatory T cells, in particular by IL-10-producing T cells ("Tr1 cells"), which are generated from effector cells and are important to avoid excessive tissue damage by adaptive immune responses (19) (20) (21) (22) . Cytokines that promote this type of regulatory T-cell response are IFN-α, IL-27, and IL-10 (23) (24) (25) , and all these cytokines can be produced by DCs (26, 27) .

Paragraph 3:
Professional APCs have to present pathogen-derived peptides on MHC molecules to activate antigen-specific T cells. DCs are phagocytic in the immature state, i.e., under steady-state conditions and upon initial pathogen encounter, and can take up antigenic material by pinocytosis or by surface receptor-mediated internalization (28) . Proteins from pathogens are then shuttled to lysosomes where they are chopped to peptides and loaded on MHC class-II molecules (29, 30) . These peptide-MHC complexes are then transported to the plasma membrane to activate specific CD4 + T cells. The presentation of peptides derived from exogenous proteins on MHC class-I, a process called cross-presentation (31, 32) , is a largely unique feature of DCs and is particularly important to activate CD8 + T cells in viral infections. Virus-infected cells express viral proteins in the cytosol where they are degraded to peptides by the proteasome, translocated to the endoplasmic reticulum by TAP proteins, and loaded on MHC class-I molecules (31) . However, since DCs are not necessarily infected by viruses, they must be able to process virus-derived proteins also from external sources, such as virus-infected cells, to activate CD8 + T cells. The mechanism of cross-presentation is still incompletely understood, but two distinct pathways via vacuoles and peptide translocation from phagolysosomes to the cytosol have been described (32) . It is believed that cross-presentation is the most important pathway leading to the induction of cytotoxic T-cell responses, and excellent reviews have been published on this relevant topic (31) (32) (33) .
```

### Források

Dataset: [Covid-19: 2020-03-13](https://allenai.org/data/cord-19)
Sentence transformer: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
Fontosabb pip csomagok:
 - NLTK
 - torch
 - sentence_transformers
 - numpy
 - pickle
 - tarfile
 - json

Debian és Ubuntu alapú disztribúciókon biztosan működik, Windows rendszeren meg elméletileg nincs akadálya.
