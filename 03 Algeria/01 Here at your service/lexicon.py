"""Name gazetteer: every capitalised name the corpus may use, with its register class.

A land / nature, B living, C built, D time / numbers / events, E speech / arts, F goods.
Multi-word names are matched first; single words are the fallback.
"""

A_NAMES = """
Chernobyl Bactrian Aswan
North Pole
South Pole
Pole Ziban M'zab Mzab Soviet Thagaste
Soviet Union
Arctic Ocean
Indian Ocean
South Pacific
Algeria Morocco Tunisia Libya Egypt Mauritania Mali Niger Chad Sudan Ethiopia Kenya Nigeria Ghana Senegal Cameroon
Angola Mozambique Tanzania Uganda Rwanda Somalia Zambia Zimbabwe Botswana Namibia Madagascar Gabon Congo Eritrea
Djibouti Malawi Benin Togo Guinea Liberia Burundi Lesotho Eswatini Tchad
South Africa
Ivory Coast
Burkina Faso
Sierra Leone
Western Sahara
France Spain Portugal Italy Germany Austria Switzerland Belgium Netherlands Holland Luxembourg Denmark Norway Sweden
Finland Iceland Ireland England Scotland Wales Britain Poland Czechia Slovakia Hungary Romania Bulgaria Greece Turkey
Albania Serbia Croatia Slovenia Bosnia Montenegro Macedonia Ukraine Belarus Russia Moldova Estonia Latvia Lithuania
Cyprus Malta Monaco Andorra Georgia Armenia Azerbaijan
United Kingdom
Great Britain
Czech Republic
China Japan Korea India Pakistan Bangladesh Nepal Bhutan Afghanistan Iran Iraq Syria Lebanon Israel Palestine Jordan
Kuwait Qatar Bahrain Oman Yemen Mongolia Kazakhstan Uzbekistan Turkmenistan Kyrgyzstan Tajikistan Vietnam Laos Cambodia
Thailand Myanmar Burma Malaysia Singapore Indonesia Philippines Taiwan Tibet Siberia
Saudi Arabia
Sri Lanka
North Korea
South Korea
United Arab Emirates
Canada Mexico Cuba Jamaica Haiti Guatemala Honduras Nicaragua Panama Colombia Venezuela Ecuador Peru Bolivia Chile
Argentina Uruguay Paraguay Brazil Guyana Suriname Australia Fiji Samoa Tonga Greenland Alaska Hawaii California
Texas Florida Quebec Bavaria Tuscany Sicily Sardinia Corsica Crete Provence Andalusia Catalonia Normandy Brittany
Patagonia Anatolia Mesopotamia Arabia Persia Scandinavia Balkans Caucasus Siberia Manchuria Bengal Punjab Kashmir
New Zealand
Costa Rica
El Salvador
Puerto Rico
Dominican Republic
Papua New Guinea
United States
America Americas Africa Europe Asia Antarctica Oceania Eurasia Arctic Antarctic
North America
South America
Latin America
Central America
Middle East
North Africa
West Africa
East Africa
Southeast Asia
Maghreb Sahel Sahara Kabylie Kabylia Hoggar Ahaggar Tassili Aures Atlas Mzab Tell Tamanrasset Djurdjura Chelia Tahat
Tassili n'Ajjer
Grand Erg
Mitidja Tademait Touat Gourara Tidikelt Ouarsenis Hodna Saoura
Kalahari Gobi Namib Atacama Mojave Sonoran Thar Taklamakan Negev Sinai Arabian Syrian Libyan Nubian Danakil
Alps Pyrenees Andes Himalaya Himalayas Rockies Urals Carpathians Apennines Appalachians Caucasus Karakoram Drakensberg
Everest Kilimanjaro Olympus Fuji Etna Vesuvius Elbrus Aconcagua Denali Matterhorn Toubkal Sinai Ararat
Mont Blanc
Table Mountain
Rocky Mountains
Atlas Mountains
Mediterranean Atlantic Pacific Indian Arctic Baltic Caribbean Caspian Adriatic Aegean Ionian Tyrrhenian
Red Sea
Black Sea
Dead Sea
North Sea
Arabian Sea
South China Sea
Gulf of Mexico
Persian Gulf
Bay of Bengal
Nile Amazon Danube Rhine Seine Thames Volga Congo Niger Zambezi Limpopo Orange Senegal Mississippi Missouri
Yangtze Ganges Indus Tigris Euphrates Jordan Tiber Loire Rhone Elbe Oder Vistula Dnieper Don Ural Mekong Yellow
Chelif Medjerda Moulouya Orinoco Parana Colorado Hudson Yukon Mackenzie
Victoria Titicaca Baikal Superior Michigan Huron Erie Ontario Tanganyika Malawi Geneva Como Garda Ness
Gibraltar Suez Bosporus Panama Magellan Dover
Sun Mercury Venus Mars Jupiter Saturn Uranus Neptune Pluto Ceres Titan Europa Io Ganymede Sirius Polaris Orion
Milky Way
Andromeda
Big Dipper
Greenwich Equator
Cape of Good Hope
Horn
Iberia Gaul Numidia Mauretania Byzantium
Algerian Moroccan Tunisian Egyptian Libyan Saharan Mediterranean Atlantic African European Asian American Canadian
Mexican Brazilian Argentinian Chilean Peruvian Australian Nigerian Kenyan Ethiopian Malian Senegalese Ghanaian
Andalusian Iberian Siberian Alpine Arctic Antarctic Tropical Amazonian Himalayan Andean Scandinavian Balkan
Swiss Belgian Austrian Irish Scottish Welsh Swedish Norwegian Danish Finnish Icelandic Polish Hungarian Romanian
Bulgarian Croatian Serbian Ukrainian Iraqi Iranian Syrian Lebanese Jordanian Saudi Yemeni Omani Qatari Emirati
Afghan Pakistani Bangladeshi Nepalese Tibetan Mongolian Vietnamese Thai Cambodian Indonesian Malaysian Filipino
Cuban Jamaican Colombian Venezuelan Bolivian Ecuadorian Uruguayan Paraguayan Panamanian Haitian Hawaiian Alaskan
Californian Texan Bavarian Tuscan Sicilian Corsican Cretan Sardinian Catalan Basque Kabyle Numidian Carthaginian
Phoenician Byzantine Persian Ottoman Mesopotamian Anatolian Caucasian Sumerian Babylonian Assyrian
Mediterranean Saharan Sahelian Maghrebi Berber Amazigh Tuareg
Celsius Fahrenheit
Northern Southern Eastern Western Central North South East West Upper Lower Great Little Grand Old New Lake Mount
"""

B_NAMES = """
Rowling
J K Rowling
Tom Sawyer
Mark Antony
Antony
Abdelkader Massinissa Jugurtha Juba Syphax Dihya Kahina Tin Hinan Augustine Apuleius Tertullian Cyprian Donatus
Emir Abdelkader
Saint Augustine
Augustine of Hippo
Ahmed Ben Bella
Houari Boumediene
Mohamed Boudiaf
Abdelaziz Bouteflika
Abdelmadjid Tebboune
Chadli Bendjedid
Ferhat Abbas
Messali Hadj
Larbi Ben
Djamila Bouhired
Hassiba Ben Bouali
Ali la Pointe
Frantz Fanon
Kateb Yacine
Assia Djebar
Mohammed Dib
Mouloud Feraoun
Mouloud Mammeri
Malek Haddad
Yasmina Khadra
Kamel Daoud
Boualem Sansal
Rachid Mimouni
Tahar Djaout
Jacques Derrida
Louis Althusser
Cheb Khaled
Cheb Mami
Cheb Hasni
Rachid Taha
Souad Massi
Lounis Ait Menguellet
Idir Matoub Lounes
Matoub Lounes
Warda
Lili Boniche
Dahmane El Harrachi
Hadj Mohamed El Anka
Zinedine Zidane
Riyad Mahrez
Rabah Madjer
Lakhdar Belloumi
Hassiba Boulmerka
Noureddine Morceli
Taoufik Makhloufi
Imane Khelif
Mohammed Racim
Baya Mahieddine
Etienne Dinet
Ibn Khaldun
Ibn Battuta
Ibn Sina
Ibn Rushd
Avicenna Averroes Alhazen Fibonacci
Muhammad Moses Jesus Abraham Noah David Solomon Mary Joseph Adam Eve Buddha Confucius Krishna God Allah
Prophet Pope Saint Sultan Emir Caliph Pharaoh Bey Dey Pasha Khan Tsar Czar Kaiser Shah Imam Sheikh Mr Mrs Dr Sir Lady Lord Queen
Homer Plato Socrates Aristotle Pythagoras Euclid Archimedes Herodotus Hippocrates Galen Ptolemy Sophocles Euripides Aesop
Alexander Caesar Cleopatra Hannibal Nero Augustus Charlemagne Attila Saladin Suleiman Ramses Tutankhamun
Nefertiti Hammurabi Cyrus Darius Xerxes Pericles Spartacus Jugurtha Hamilcar Scipio Hadrian Trajan Septimius Severus
Julius Caesar
Napoleon Bonaparte
Napoleon
Genghis Khan
Kublai Khan
Marco Polo
Christopher Columbus
Vasco da Gama
Ferdinand Magellan
James Cook
Neil Armstrong
Yuri Gagarin
Buzz Aldrin
Amelia Earhart
Charles Lindbergh
Shakespeare Tolstoy Dante Leonardo Vinci Einstein Beethoven Newton Washington
William Shakespeare
Leo Tolstoy
Dante Alighieri
Leonardo da Vinci
Albert Einstein
Ludwig van Beethoven
Isaac Newton
George Washington
Yves Saint Laurent
Albert Camus
Camus
Miguel de Cervantes
Cervantes Austen Hugo Goethe Dostoevsky Dickens Hemingway Orwell Twain Melville Kafka Joyce Proust Flaubert Balzac Zola
Voltaire Rousseau Moliere Racine Baudelaire Rimbaud Verlaine Pushkin Chekhov Gogol Tolkien Rowling Achebe Soyinka Mahfouz
Borges Neruda Marquez Rumi Hafez Khayyam Tagore Gibran Kipling Wilde Byron Keats Shelley Wordsworth Chaucer Milton Poe
Whitman Dickinson Steinbeck Faulkner Fitzgerald Salinger Orwell Huxley Bradbury Asimov Verne Dumas Stendhal Sartre
Beauvoir Brecht Ibsen Lorca Cervantes Petrarch Boccaccio Virgil Ovid Seneca Cicero Sappho Confucius Laozi Basho
Jane Austen
Victor Hugo
Charles Dickens
Ernest Hemingway
George Orwell
Mark Twain
Herman Melville
Franz Kafka
James Joyce
Marcel Proust
Gustave Flaubert
Emile Zola
Alexandre Dumas
Jules Verne
Fyodor Dostoevsky
Anton Chekhov
Alexander Pushkin
Chinua Achebe
Wole Soyinka
Naguib Mahfouz
Gabriel Garcia Marquez
Pablo Neruda
Jorge Luis Borges
Rabindranath Tagore
Khalil Gibran
Omar Khayyam
Oscar Wilde
Mary Shelley
Edgar Allan Poe
Lewis Carroll
Agatha Christie
Arthur Conan Doyle
Antoine de Saint
Picasso Monet Rembrandt Vermeer Botticelli Michelangelo Raphael Caravaggio Goya Dali Munch Klimt Matisse Cezanne Renoir
Degas Manet Gauguin Kahlo Rivera Warhol Pollock Turner Constable Velazquez Rubens Titian Bosch Bruegel Delacroix Magritte
Pablo Picasso
Claude Monet
Vincent van Gogh
Van Gogh
Gogh
Salvador Dali
Edvard Munch
Frida Kahlo
Henri Matisse
Paul Cezanne
Eugene Delacroix
Johannes Vermeer
Sandro Botticelli
Mozart Bach Vivaldi Handel Haydn Chopin Liszt Schubert Schumann Brahms Wagner Verdi Puccini Rossini Tchaikovsky Debussy
Ravel Bizet Strauss Mahler Stravinsky Dvorak Grieg Sibelius Elgar Holst Gershwin Satie Berlioz Mendelssohn Rachmaninoff
Wolfgang Amadeus Mozart
Johann Sebastian Bach
Antonio Vivaldi
Frederic Chopin
Giuseppe Verdi
Pyotr Tchaikovsky
Claude Debussy
Maurice Ravel
Georges Bizet
Elvis Presley
Michael Jackson
Bob Marley
Fela Kuti
Umm Kulthum
Fairuz
Beatles Mozart Madonna Elvis Adele Shakira Beyonce
Galileo Copernicus Kepler Darwin Curie Pasteur Edison Tesla Faraday Maxwell Bohr Planck Heisenberg Hawking Mendel
Mendeleev Lavoisier Fleming Freud Jung Pavlov Turing Lovelace Babbage Bell Marconi Nobel Gutenberg Watt Volta Ampere
Ohm Kelvin Joule Hubble Halley Linnaeus Lamarck Wegener Hooke Boyle Dalton Rutherford Fermi Oppenheimer Feynman
Galileo Galilei
Nicolaus Copernicus
Johannes Kepler
Charles Darwin
Marie Curie
Pierre Curie
Louis Pasteur
Thomas Edison
Nikola Tesla
Michael Faraday
James Watt
Alexander Graham Bell
Alexander Fleming
Sigmund Freud
Alan Turing
Ada Lovelace
Gregor Mendel
Dmitri Mendeleev
Alfred Nobel
Johannes Gutenberg
Stephen Hawking
Edwin Hubble
Niels Bohr
Max Planck
Rosalind Franklin
Wright
Wright brothers
Lincoln Jefferson Roosevelt Kennedy Obama Churchill Gandhi Mandela Lenin Stalin Mao Bismarck Gaulle Hitler Mussolini
Elizabeth Victoria Louis Henry Charles Philip Peter Catherine Isabella Ferdinand Akbar Ashoka Mansa Musa Sundiata
Abraham Lincoln
Thomas Jefferson
Winston Churchill
Mahatma Gandhi
Nelson Mandela
Martin Luther King
Charles de Gaulle
Queen Victoria
Queen Elizabeth
Louis XIV
Mansa Musa
Haile Selassie
Kwame Nkrumah
Gamal Abdel Nasser
Anwar Sadat
Habib Bourguiba
Mohammed V
Pele Maradona Messi Ronaldo Zidane Jordan Ali Bolt Federer Nadal Djokovic Serena Williams Owens Phelps Comaneci
Muhammad Ali
Usain Bolt
Lionel Messi
Cristiano Ronaldo
Diego Maradona
Michael Jordan
Roger Federer
Rafael Nadal
Jesse Owens
Romans Greeks Egyptians Phoenicians Carthaginians Vikings Aztecs Incas Maya Mayans Mongols Ottomans Normans Celts
Persians Bedouins Berbers Amazigh Imazighen Kabyles Chaouis Mozabites Tuareg Touareg Moors Arabs Turks Franks Goths
Vandals Huns Saxons Zulu Maasai Yoruba Igbo Hausa Fulani Bantu Pygmies Inuit Sami Cherokee Sioux Apache Navajo Kurds
Algerians Moroccans Tunisians Egyptians French English Spanish Italians Germans Americans Africans Europeans Asians
Christians Muslims Jews Buddhists Hindus Catholics Protestants Pharaohs Sultans Caliphs Emirs Romans Byzantines
Roman Greek Viking Aztec Inca Mayan Mongol Norman Celtic Bedouin Moorish Frankish Vandal Hun Zulu Maasai Yoruba
Muslim Christian Jewish Catholic Protestant Orthodox Hindu Buddhist Sufi Sunni Shia
Fennec Barbary
Santa Claus
"""

C_NAMES = """
Aswan High Dam
High Dam
Sistine Chapel
Thagaste Souk Ahras Sidi Ferruch Tagaste Kalaa
Souk Ahras
Sidi Ferruch
Algiers Oran Constantine Annaba Blida Batna Setif Tlemcen Bejaia Biskra Ghardaia Tamanrasset Djanet Ouargla Bechar
Tipaza Timgad Djemila Cherchell Mostaganem Skikda Jijel Guelma Tebessa Medea Tiaret Laghouat Djelfa Adrar Tindouf
El Oued
Tizi Ouzou
Beni Isguen
Bou Saada
Hassi Messaoud
In Salah
Sidi Bel Abbes
Mondovi Hippo Cirta Icosium Caesarea Lambaesis Tiddis Cuicul Thamugadi Hippo Regius
Casablanca Rabat Marrakesh Marrakech Fez Fes Tangier Meknes Agadir Essaouira Chefchaouen Ouarzazate
Tunis Sfax Sousse Kairouan Carthage Tripoli Benghazi Cairo Alexandria Luxor Aswan Giza Memphis Thebes Khartoum
Timbuktu Bamako Dakar Niamey Nouakchott Lagos Abuja Accra Nairobi Mombasa Zanzibar Kampala Kigali Kinshasa Luanda
Maputo Harare Lusaka Windhoek Gaborone Pretoria Johannesburg Durban Soweto Addis Ababa Asmara Mogadishu Dodoma
Cape Town
Addis Ababa
Dar es Salaam
Abidjan Yamoussoukro Ouagadougou Conakry Freetown Monrovia Lome Cotonou Yaounde Douala Libreville Brazzaville Antananarivo
Paris Marseille Lyon Toulouse Nice Bordeaux Lille Strasbourg Nantes Avignon Versailles Cannes Montpellier
London Manchester Liverpool Birmingham Edinburgh Glasgow Dublin Belfast Cardiff Oxford Cambridge York Bristol Leeds
Madrid Barcelona Seville Granada Cordoba Valencia Toledo Malaga Bilbao Lisbon Porto Rome Milan Naples Venice Florence
Turin Pisa Genoa Bologna Verona Palermo Pompeii Berlin Munich Hamburg Frankfurt Cologne Vienna Salzburg Zurich Geneva
Bern Basel Brussels Antwerp Bruges Amsterdam Rotterdam Utrecht Copenhagen Oslo Bergen Stockholm Helsinki Reykjavik
Warsaw Krakow Prague Budapest Bucharest Sofia Athens Sparta Istanbul Ankara Constantinople Troy Belgrade Zagreb
Sarajevo Ljubljana Tirana Skopje Kyiv Kiev Minsk Moscow Petersburg Vilnius Riga Tallinn Chisinau Valletta Nicosia
Saint Petersburg
Saint Louis
Tbilisi Yerevan Baku Beirut Damascus Aleppo Amman Jerusalem Bethlehem Baghdad Babylon Basra Mosul Tehran Isfahan
Riyadh Jeddah Mecca Medina Dubai Doha Muscat Kuwait Manama Sanaa Kabul Islamabad Karachi Lahore Delhi Mumbai Kolkata
Chennai Bangalore Agra Varanasi Kathmandu Dhaka Colombo Beijing Shanghai Hong Kong Guangzhou Xian Tokyo Kyoto Osaka
Hiroshima Nagasaki Seoul Pyongyang Hanoi Bangkok Manila Jakarta Singapore Kuala Lumpur Samarkand Bukhara Tashkent
Almaty Astana Ulaanbaatar Lhasa Taipei Rangoon Yangon Petra Palmyra Persepolis Ur Nineveh Jericho Angkor Wat
Washington
New York
New Delhi
Los Angeles
San Francisco
Las Vegas
New Orleans
Rio de Janeiro
Buenos Aires
Mexico City
Sao Paulo
Chicago Boston Philadelphia Seattle Miami Houston Dallas Detroit Atlanta Toronto Montreal Vancouver Ottawa Havana
Lima Cusco Bogota Caracas Quito Santiago Brasilia Montevideo Asuncion Sydney Melbourne Canberra Perth Wellington
Auckland Honolulu Anchorage
Eiffel Tower
Big Ben
Colosseum Coliseum Kremlin Louvre Alhambra Kasbah Casbah Acropolis Parthenon Pantheon Sphinx Stonehenge Vatican
Sorbonne Versailles Titanic Internet Hagia Sophia Mezquita Kaaba Westminster Buckingham Notre Dame Taj Mahal
Machu Picchu
Great Wall
Great Pyramid
Leaning Tower
Tower Bridge
Golden Gate Bridge
Golden Gate
Brooklyn Bridge
Statue of Liberty
Burj Khalifa
Sydney Opera House
Opera House
Forbidden City
Sagrada Familia
Brandenburg Gate
Arc de Triomphe
Notre Dame
Hagia Sophia
Blue Mosque
Great Mosque
Ketchaoua Mosque
Martyrs Memorial
Maqam Echahid
Notre Dame d'Afrique
Basilica
Panama Canal
Suez Canal
Trans-Saharan
Christ the Redeemer
Mount Rushmore
Empire State Building
White House
Wall Street
Abu Simbel
Valley of the Kings
Mausoleum
Tassili
Apollo Concorde Boeing Airbus Mercedes Toyota Ford Ferrari Tesla Google Apple Microsoft Amazon Facebook Wikipedia
Hubble
Sputnik
Voyager
"""

D_NAMES = """
Hastings Armada Hijra
Battle of Hastings
Spanish Armada
Waterloo Trojan
Trojan War
Civil War
American Civil War
Olympic Games
Games
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
January February March April May June July August September October November December
Ramadan Eid Mawlid Ashura Yennayer Christmas Easter Hanukkah Passover Diwali Thanksgiving Halloween Carnival Nowruz
Eid al Fitr
Eid al Adha
New Year
Olympics Olympic Paralympics World Cup Wimbledon Tour de France Super Bowl
African Cup of Nations
Africa Cup of Nations
Renaissance Enlightenment Reformation Middle Ages Stone Age Bronze Age Ice Age Neolithic Paleolithic Jurassic
Cold War
World War
Big Bang
Industrial Revolution
French Revolution
Russian Revolution
American Revolution
Algerian War
Hundred Years
Crusades Holocaust Reconquista Inquisition Armistice Prohibition Depression
Great Depression
Black Death
Silk Road
Arab Spring
Victorian Medieval Ancient Modern Baroque Romantic Classical Prehistoric Antiquity Islamic Golden
BC BCE AD CE
Nobel Prize
Nobel Oscar Oscars Grammy Pulitzer
First Second Third Fourth Fifth Sixth Seventh Eighth Ninth Tenth Hundred Thousand
Ninth Symphony
Fifth Symphony
"""

E_NAMES = """
Muqaddimah
The Adventures of Tom Sawyer
The Trial
Around the World in Eighty Days
The Three Musketeers
The Old Man and the Sea
Nineteen Eighty-Four
Arabic Tamazight Berber Kabyle Darija French English Spanish Portuguese German Italian Russian Chinese Mandarin
Cantonese Japanese Korean Hindi Urdu Bengali Punjabi Tamil Swahili Amharic Hausa Yoruba Zulu Xhosa Afrikaans Latin
Greek Hebrew Aramaic Persian Farsi Turkish Kurdish Dutch Flemish Danish Swedish Norwegian Finnish Icelandic Polish
Czech Hungarian Romanian Bulgarian Serbian Croatian Ukrainian Thai Vietnamese Indonesian Malay Tagalog Somali Esperanto
Sanskrit Coptic Punic Phoenician Gaelic Welsh Catalan Basque Occitan Breton Creole Wolof Lingala Tigrinya Berber
Tifinagh Latin Cyrillic Braille Morse
Islam Christianity Judaism Buddhism Hinduism Sikhism Taoism Shinto Sufism Catholicism Protestantism Orthodoxy
Quran Koran Bible Torah Talmud Gospel Psalms Vedas Upanishads Iliad Odyssey Aeneid Gilgamesh Beowulf Mahabharata Ramayana
Hamlet Macbeth Othello Lear Romeo Juliet Faust Candide Ulysses Frankenstein Dracula Emma Nedjma Inferno Paradiso
Divine Comedy
Arabian Nights
Thousand and One Nights
Romeo and Juliet
King Lear
Don Quixote
War and Peace
Anna Karenina
Crime and Punishment
Les Miserables
Pride and Prejudice
Oliver Twist
Great Expectations
Moby Dick
The Stranger
The Plague
The Little Prince
The Odyssey
The Iliad
The Republic
The Prince
Harry Potter
The Hobbit
Lord of the Rings
Things Fall Apart
Alice in Wonderland
Old Man and the Sea
One Hundred Years of Solitude
Mona Lisa
Starry Night
The Starry Night
The Scream
Guernica
Last Supper
The Last Supper
Night Watch
The Night Watch
Water Lilies
Sunflowers
Persistence of Memory
The Persistence of Memory
Birth of Venus
The Birth of Venus
Girl with a Pearl Earring
The Thinker
David
Ode to Joy
Four Seasons
The Four Seasons
Magic Flute
The Magic Flute
Swan Lake
Nutcracker
The Nutcracker
Bolero
Carmen
Aida
Messiah
Clair de Lune
Moonlight Sonata
Brandenburg Concertos
Requiem
Kassaman
Rai Chaabi Andalusi Malouf Gnawa Gnaoua Jazz Blues Reggae Flamenco Tango Salsa Samba Hip Hop Rock Opera Ballet
Impressionism Cubism Surrealism Romanticism Realism Baroque Modernism Expressionism
Algebra Geometry Biology Physics Astronomy Philosophy Geography Economics Medicine Latin Calligraphy
Pythagorean Newtonian Darwinian
UNESCO UNICEF NATO UN WHO FIFA NASA ESA OPEC Unesco
United Nations
African Union
Arab League
European Union
Red Cross
Red Crescent
Constitution
Declaration of Independence
Magna Carta
Nobel
Wikipedia
DNA RNA ATP
English Channel
"""

F_NAMES = """
Deglet Nour
Deglet Noor
Medjool
Chorba Chakhchoukha Rechta Mhajeb Makroud Baklava Tajine Tagine Merguez Harira Bourek Dolma Mechoui Zlabia Kalbelouz
Dinar Dirham Euro Euros Dollar Dollars Pound Yen Yuan Rupee Ruble Franc Lira Peso Real Riyal Rand Naira Shilling
Kente Champagne Cognac Parmesan Cheddar Roquefort Camembert Brie Feta Mozzarella Gouda
Coca Cola
"""

CLASS_TEXT = {"A": A_NAMES, "B": B_NAMES, "C": C_NAMES, "D": D_NAMES, "E": E_NAMES, "F": F_NAMES}

# names that are languages: class E when used as a language, otherwise the class of the place / people
LANGUAGE_WORDS = set("""
Arabic Tamazight Berber Kabyle Darija French English Spanish Portuguese German Italian Russian Chinese Mandarin
Cantonese Japanese Korean Hindi Urdu Bengali Punjabi Tamil Swahili Amharic Hausa Yoruba Zulu Xhosa Afrikaans Latin
Greek Hebrew Aramaic Persian Farsi Turkish Kurdish Dutch Flemish Danish Swedish Norwegian Finnish Icelandic Polish
Czech Hungarian Romanian Bulgarian Serbian Croatian Ukrainian Thai Vietnamese Indonesian Malay Tagalog Somali
Sanskrit Coptic Punic Phoenician Gaelic Welsh Catalan Basque Occitan Breton Creole Wolof Lingala Tigrinya
""".split())

# class of a language word when it is NOT used as a language (adjective of a country / people)
LANG_ADJ_CLASS = {w: "A" for w in LANGUAGE_WORDS}
for w in ("Berber", "Kabyle", "Hausa", "Yoruba", "Zulu", "Kurdish", "Basque", "Wolof", "Phoenician", "Punic", "Coptic", "Greek"):
    LANG_ADJ_CLASS[w] = "B"


STOP_SINGLE = set("The A An In Of On At To And With For El Al Le La Les De Da Du Des Van Von Di Del Es Ben Bin Ibn Abu Ait N Sea Lake Mount".split())


def _build():
    multi: dict[tuple, str] = {}
    single: dict[str, str] = {}
    for cls, text in CLASS_TEXT.items():
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            words = line.split()
            # a line with several capitalised words separated by spaces is either a multiword
            # name (few words) or a list of single names; lists have > 4 entries or no lowercase
            if len(words) <= 4 and not (len(words) > 1 and all(w[0].isupper() for w in words) and len(words) >= 3 and cls in ("A",) and False):
                pass
            if text_is_list(line):
                for w in words:
                    single.setdefault(w, cls)
            else:
                multi.setdefault(tuple(words), cls)
                for w in words:
                    if w[0].isupper() and w not in STOP_SINGLE:
                        single.setdefault(w, cls)
    return multi, single


def text_is_list(line: str) -> bool:
    """A line of 5+ capitalised words is a list of single names; otherwise it is one name."""
    words = line.split()
    return len(words) >= 5 and all(w[:1].isupper() for w in words)


MULTI, SINGLE = _build()
MAX_MULTI = max(len(k) for k in MULTI)
