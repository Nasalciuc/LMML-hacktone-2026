import json
import os
import re
import sys
import unicodedata


def fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower()


def words(s):
    return set(fold(s).split())


FIRST = words("""
aaron abel abigail abraham adan adolfo adrian adriana agustin agustina aida alberto alejandra alejandro
alexa alexander alexandra alexis alfonso alfredo alicia alina alison allan alonso alvaro amanda amelia ana
anabel anabelle andrea andres andy angela angelica angie anibal antonella antonia antonio araceli ariana
ariel armando arnoldo arturo astrid aurora axel barbara beatriz benjamin bernardo berta brandon brenda
bryan camila candela carla carlos carmen carolina catalina cecilia cesar christian cindy cinthia clara
claudia claudio cristian cristina cristopher dafne damian daniel daniela dario david debora denis diana
diego dilan douglas eduardo efrain elena eliana elias elisa elizabeth elsa emanuel emilia emiliano emilio
emily emma enrique erick erika ernesto esteban estefania eugenia eva evelyn fabian fabiola fabricio
federico felipe fernanda fernando fiorella francisco francella franklin gabriel gabriela gerardo german
gilberto giselle gonzalo graciela gregorio guadalupe guillermo gustavo hannia hector helena henry hernan
hilda hugo ignacio ileana ines irene isaac isabel isabella isaias ismael ivan ivania ivonne jacqueline
jaime javier jennifer jenny jeremy jesus jimena joaquin johan johanna jonathan jorge jose josefina josue
juan juana julia julian juliana julio karen karina karla katherine kevin laura leonardo leticia lilliam
liliana lorena lorenzo lourdes lucas lucia luciana luis luisa manuel marcela marco marcos margarita maria
mariana mariano maribel maricela mario marisol marta martin martina mateo matias mauricio maximiliano
mayra melany melissa miguel milena miriam monica nancy natalia nathalia nelson nicolas nicole noelia
norma octavio olga oscar pablo pamela paola patricia paula pedro priscilla rafael ramiro ramon randall
raquel raul rebeca ricardo roberto rocio rodolfo rodrigo rolando ronald ruben samuel sandra santiago
sara sebastian sergio silvia sofia sonia stephanie susana tatiana teresa tomas ursula valentina valeria
vanessa veronica vicente viviana walter ximena yadira yesenia yolanda zulema pilar alba rosa luz paz
victoria mercedes dolores gloria blanca amparo consuelo esperanza marina flor minor dulce cruz angel
jeffry jeffrey jason johnny jefferson joseph josefa wendy yuliana yamileth yessenia xiomara gerson
alejandrina rosario fatima ivania guiselle grettel lidieth marianela shirley sharon priscila ronny
james john robert michael william richard thomas charles christopher matthew anthony donald steven paul
andrew joshua kenneth brian george edward timothy jeffrey ryan jacob nicholas eric stephen larry justin
scott gregory patrick dennis jerry tyler henry peter adam nathan zachary kyle harold carl keith gerald
ethan arthur terry sean lawrence austin noah jesse albert bruce jordan dylan alan ralph wayne eugene
logan randy louis russell vincent philip bobby bradley mary linda susan jessica sarah ashley kimberly
donna michelle dorothy deborah rebecca cynthia kathleen amy anna samantha christine debra rachel
carolyn janet catherine heather diane julie joyce kelly lauren joan olivia judith megan cheryl martha
frances hannah ann kathryn alice janice doris madison denise beverly theresa marilyn danielle brittany
natalie sophia charlotte ava mia harper chloe zoe lucy oliver liam mason elijah owen caleb luke
""")

SURN = words("""
solano hernandez araya quesada hidalgo jimenez obando guzman urena vargas zuniga cordero segura fonseca
nunez chaves brenes mena castro paniagua rodriguez gonzalez fernandez lopez martinez sanchez perez gomez
ramirez torres flores rivera diaz morales ortiz gutierrez chavez ramos ruiz alvarez mendoza vasquez
vazquez castillo moreno romero herrera medina aguilar garcia rojas mora alfaro alpizar arias arce
barrantes benavides bolanos bonilla calderon campos carvajal cerdas chacon chinchilla cespedes corrales
delgado duran elizondo esquivel fallas garro granados guevara gamboa leiton leon madrigal marin mata
matarrita mendez miranda molina monge montero montoya morera munoz murillo navarro ovares pacheco porras
quiros rivas robles rosales salas salazar sandi sequeira serrano solis soto tenorio umana valverde vega
venegas villalobos vindas zamora acuna aguero alvarado avila badilla blanco cabezas camacho carmona
cubero espinoza estrada fuentes jara leal lizano loria lobo mesen meza mejia naranjo oviedo pereira
prado retana rios saborio sanabria solorzano tellez trejos ugalde valerio valenzuela villegas abarca
anchia angulo artavia astua azofeifa bermudez bogantes calvo chaverri coto cordoba cortes dittel
echeverria garita guillen jaen lara maroto montenegro moya ocampo orozco ortega padilla pineda pizarro
santamaria valle cruz castro rosas vindas alvarado sossa sosa cabrera reyes vidal vargas
smith johnson williams jones miller davis wilson anderson taylor moore jackson harris clark lewis
robinson walker allen wright adams baker nelson carter mitchell roberts turner phillips campbell parker
evans edwards collins stewart morris murphy rogers morgan cooper peterson bailey kelly howard cox
richardson watson brooks bennett hughes sanders myers foster thompson martin garcia
""")

AMBIG = words("""
rosa luz paz cruz angel leon victoria mercedes dolores gloria alba blanca amparo consuelo esperanza
marina flor minor dulce pilar christian sharon mayra lourdes guadalupe rosario fatima
""")

PARTICLES = {'de', 'del', 'la', 'las', 'los', 'van', 'von', 'da', 'di', 'dos', 'das', 'mc'}

CAPSTOP = words("""
pura vida voyages voyage costa rica hi hello hey dear hola estimado estimada estimados estimadas querido
querida team equipo please por favor thanks thank you gracias regards saludos atentamente cordialmente
i my mi me the el la los las we our nuestro nuestra this esta este it is a an and y or o to for from of
de del in en on at with con by para avenida calle parque iglesia supermercado panaderia banco hotel tour
tours canopy volcano volcan playa beach rio river national nacional park reserva aeropuerto airport
monday tuesday wednesday thursday friday saturday sunday lunes martes miercoles jueves viernes sabado
domingo january february march april may june july august september october november december enero
febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre pv cr iban
mr mrs ms sr sra srta dona dr dra lic ing sir madam best kind warm cheers sincerely yes no ok
also note ps subject asunto re fw fwd booking reserva reservation reference referencia urgent urgente
refund reembolso payment pago complaint queja invoice factura good buenos buenas dias tardes noches
morning afternoon evening love shuttle cedula id phone telefono email correo account cuenta address
direccion name nombre passenger pasajero client cliente customer guest my your his her their they he she
we us who what when where why how which there here that these those can could would should will
manager gerente support soporte service servicio customer agency agencia office oficina staff guide
dirección señor señora señorita don
""")

STOP = words("""
and or but i my me you your we our us he she they it is are was were be been am will would can could
should shall may might must have has had do does did not no yes so if then than that this these those
there here who whom which what when where why how all any some each every also too very just only from
to of in on at by for with about as into like through after over between out against during without
before under around among the a an please thanks thank call reach email write travel traveling
travelling travels join joining joins come coming comes go going book booked booking stay staying
arrive arriving want wants need needs like asked ask help again today tomorrow tonight now soon later
y o pero yo mi mis me te tu su sus nos nuestro nuestra el la los las un una unos unas en con por para
que como cuando donde quien es son fue era soy estoy esta estan va van viaja viajara vendra tambien muy
ya no si gracias favor hola le les lo al se mas ser estar tener hay hoy manana conmigo with me us
will would wants is and also too as well who
""")

PLACES = [fold(p) for p in """
san jose|manuel antonio|santa teresa|santa ana|santa elena|santa cruz|san carlos|san isidro|san ramon
|puerto jimenez|ciudad quesada|puerto viejo|juan santamaria|daniel oduber|braulio carrillo
|rincon de la vieja|playa hermosa|playa flamingo|playa conchal|los chiles|la fortuna|monteverde
|tortuguero|arenal|tamarindo|quepos|limon|heredia|alajuela|cartago|puntarenas|guanacaste|liberia
|nicoya|escazu|golfito|drake|corcovado|cahuita|uvita|dominical|jaco|samara|nosara|montezuma|turrialba
|poas|irazu|rio celeste|barva|sarapiqui|pavones|osa|santa rosa|palo verde|cano negro|ostional
|mal pais|grecia|atenas|orotina|perez zeledon|san pedro|curridabat|moravia|tibas|desamparados
|coronado|santo domingo|belen|tres rios|paraiso|siquirres|guapiles|bribri|sixaola|upala|canas
|bagaces|tilaran|filadelfia|hojancha|nandayure|santa barbara|san vito|buenos aires
|ciudad neily|coto brus|parrita|garabito|miramar|san mateo|costa rica|pura vida voyages
|pura vida|central america|la sabana|rohrmoser|los yoses|barrio escalante|sabanilla|guadalupe
|san francisco|san rafael|san antonio|san juan|san miguel|san pablo|san vicente|santa lucia
|zapote|pavas|hatillo|uruca|la uruca|aserri|alajuelita|santo tomas|mercedes norte|san joaquin
|santa maria|piedades|ciudad colon|puriscal|dota|tarrazu|leon cortes
|vazquez de coronado|montes de oca|goicoechea|carara|cocos|isla del coco|isla tortuga|chirripo
|cabo blanco|la selva|bijagua|ojochal|matapalo|herradura|tarcoles|portalon|potrero|brasilito
|nuevo arenal|el castillo|cuajiniquil|playas del coco|el coco|ocotal|papagayo
""".replace('\n', '').split('|') if p.strip()]
PLACES = set(p.strip() for p in PLACES)

PLACE_HEADS = words("""
playa volcan volcano rio river parque park hotel restaurante restaurant reserva refugio isla island cerro
lago laguna lake catarata cataratas falls sendero trail puerto bahia bay punta monte villa finca lodge
cabinas soda aeropuerto airport barrio residencial urbanizacion condominio escuela colegio liceo
universidad hospital clinica clinic museo museum teatro iglesia church mercado supermercado panaderia
banco calle avenida ruta carretera estadio centro mall plaza tienda san santa santo
""")

CITY_SET = PLACES | words("""
heredia alajuela cartago limon puntarenas guanacaste quepos escazu liberia nicoya jaco golfito
""")

MONTHS_EN = (r'jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|'
             r'sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?')
MONTHS_ES = (r'enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|'
             r'noviembre|diciembre|ene|abr|ago|dic')
MON = '(?:' + MONTHS_EN + '|' + MONTHS_ES + ')'

DATE_RXS = [re.compile(p, re.I) for p in [
    r'\b' + MON + r'\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+(?:de\s+|del\s+)?\d{4}\b',
    r'\b\d{1,2}(?:st|nd|rd|th|º|°)?\s+(?:of\s+|de\s+)?' + MON + r'\.?,?\s+(?:of\s+|de\s+|del\s+)?\d{4}\b',
    r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b',
    r'\b\d{4}[/.-]\d{1,2}[/.-]\d{1,2}\b',
    r'\b\d{1,2}/\d{1,2}/\d{2}\b',
]]

PV_RX = re.compile(r'\bPV[- ]?[A-Z0-9]{3,10}\b', re.I)
EMAIL_RX = re.compile(r"[\w.+'-]+(?:[ \t]?@[ \t]?|[ \t]*[(\[]at[)\]][ \t]*)[\w-]+(?:(?:\.|[ \t]*[(\[]dot[)\]][ \t]*)[\w-]+)+", re.I)
IBAN_RX = re.compile(r'\bCR[ \t.-]*\d{2}(?:[ \t.-]*\d){14,22}\b', re.I)
IBAN2_RX = re.compile(r'\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b')
RUN_RX = re.compile(r'(?<![\w])(?:\+|00)?[ \t]*\(?\+?\d[\d \t().\-/]{5,30}\d(?![\w])')

DIR = (r'(?:norte|sur|este|oeste|noreste|noroeste|sureste|suroeste|north|south|east|west|'
       r'northeast|northwest|southeast|southwest)')
UNIT = r'(?:metros|metro|mts?\.?|m\.?|varas|meters|metres|yards|km)'
DIST = r'\d+\s*' + UNIT + r'\s+(?:al\s+|hacia\s+el\s+|to\s+the\s+)?' + DIR
DISTS = DIST + r'(?:\s*(?:,|y|and)\s*\d+\s*(?:' + UNIT + r'\s*)?(?:al\s+|to\s+the\s+)?' + DIR + r')*'
OSTOP = (r'(?:to|and|for|para|y|keep|please|if|so|in|on|with|or|o|but|pero|because|porque|since|then|'
         r'as|that|que|is|es|our|my|mi|nuestro|nuestra|your|su|tu|when|cuando|by|at|en|we|i|it|this|'
         r'esta|este|until|hasta|before|after|antes|despues|from|desde)')
OW = r"(?:[^\W\d_][\w'’&.-]*|\d+[A-Za-z]?|#\d+)"
OBJ = OW + r'(?:[ \t]+(?!' + OSTOP + r'\b)' + OW + r'){0,6}'
CONN = r'(?:de\s+la|del|de\s+los|de\s+las|de|al|a\s+la|a\s+los|a\s+las|a|of\s+the|of|from\s+the|from|the)\s+'

ADDR_ANCHORS = [re.compile(p) for p in [
    r'(?i:\b' + DISTS + r')\s+(?:(?i:' + CONN + r'))?' + OBJ,
    r'(?i:\b(?:del|de\s+la|de\s+los|de\s+las|from\s+the)\s+)' + OW + r'(?:[ \t]+' + OW + r'){0,4}?\s+(?i:' + DISTS + r')',
    r'(?i:\b(?:contiguo|frente|detr[aá]s|diagonal|costado\s+' + DIR + r'|al\s+costado\s+' + DIR +
    r'|al\s+lado|junto))\s+(?i:(?:a\s+la|al|a\s+los|a\s+las|a|de\s+la|del|de\s+los|de)\s+)?' + OBJ,
    r'(?i:\b(?:avenida|avda\.?|av\.|calle|carretera|ruta|transversal|bulevar|boulevard|callej[oó]n))'
    r'\s+(?:\d+[A-Za-z]?\b|(?i:n[oº°]\.?)\s*\d+|[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:de\s+|del\s+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,2})',
    r'\b\d+\s+[A-Z][\w]+(?:\s+[A-Z][\w]+)?\s+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Lane|Drive)',
    r'(?i:\b(?:casa|house|apartamento|apartment|apto\.?|apt\.?|lote|unit|condominio|residencial|'
    r'urbanizaci[oó]n|barrio|bo\.))\s*(?i:#|n[oº°]\.?|n[uú]mero|number)?\s*(?:\d+[A-Za-z]?\b|'
    r'[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:(?i:de|del|la|las|los|el)\s+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,3})'
    r'(?i:(?:\s+(?:de\s+)?color\s+\w+|\s+esquinera|\s+con\s+port[oó]n\s+\w+)*)',
    r'(?i:\b(?:casa|port[oó]n)\s+(?:(?:de\s+)?color\s+\w+|esquinera|de\s+(?:dos|tres)\s+(?:pisos|plantas)|'
    r'blanca|verde|azul|amarilla|roja|celeste|gris|rosada|negro|negra|caf[eé])(?:\s+con\s+port[oó]n\s+\w+)?)',
]]
ADDR_EN_REL = re.compile(r'(?i:(?:across\s+from|next\s+to|in\s+front\s+of|behind|opposite|beside)\s+(?:the\s+)?)' + OBJ)
CITY_RX = re.compile(r"[A-ZÁÉÍÓÚÑ][\wáéíóúñü]*\.?(?:[ \t]+(?:de[ \t]+(?:la[ \t]+|los[ \t]+)?|del[ \t]+)?"
                     r"[A-ZÁÉÍÓÚÑ][\wáéíóúñü]*){0,3}(?=[ \t]*(?:[.,;:!?)\n]|$)|[ \t]+[a-záéíóúñ])")
POSTAL_RX = re.compile(r'\d{5}\b')
BUILDING = words('hotel residencial condominio barrio urbanizacion apartamentos edificio torre cabinas lodge hostal')
COMMA_RX = re.compile(r'[ \t]*,[ \t]*')
ADDR_FIELD_RX = re.compile(r'(?im)\b(?:direcci[oó]n|address|domicilio|se[nñ]as)[ \t]*:[ \t]*([^\n]+?)[ \t]*\.?[ \t]*$')

TOKEN_RX = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
SENT_RX = re.compile('\uE000[a-z]*\uE001')

STRONG_CUES = [
    r'\bmy\s+(?:full\s+)?name\s+is', r'\bmi\s+nombre(?:\s+completo)?\s+es', r'\bme\s+llamo',
    r'\b(?:full\s+)?name\s*:', r'\bnombre(?:\s+completo)?\s*:', r'\bpasajer[oa]s?\s*:', r'\bpassengers?\s*:',
    r'\btitular(?:\s+de\s+la\s+cuenta)?\s*:', r'\baccount\s+holder\s*:', r'\bholder\s*:', r'\bclient[e]?\s*:',
    r'\bcustomer\s*:', r'\bcontact[o]?\s*:', r'\bguest\s*:', r'\bhu[eé]sped\s*:', r'\bfrom\s*:', r'^de\s*:',
    r'\bsigned\s*:', r'\bfirma(?:do)?\s*:', r'\bunder\s+the\s+name(?:\s+of)?', r'\bin\s+the\s+name\s+of',
    r'\ba\s+nombre\s+de', r'\b(?:booked|reserved|reservation|booking)\s+under', r'\bmy\s+name\s*[:-]',
]
REL = (r'\b(?:my|our)\s+(?:(?:\d+|two|three|little|older|younger|eldest|youngest|teenage|adult)[\s-]*'
       r'(?:year[\s-]*old\s+)?)?(?:son|daughter|wife|husband|partner|mother|mom|mum|father|dad|brother|'
       r'sister|friend|colleague|boyfriend|girlfriend|fianc[eé]e?|cousin|aunt|uncle|grandson|granddaughter|'
       r'grandmother|grandma|grandfather|grandpa|niece|nephew|child|kid|stepson|stepdaughter|companion|'
       r'coworker|co-worker|travel\s+companion|spouse|mother-in-law|father-in-law|sister-in-law|'
       r'brother-in-law|guide|driver)s?'
       r'|\b(?:mi|nuestr[oa])\s+(?:hij[oa]|espos[oa]|pareja|madre|mam[aá]|padre|pap[aá]|herman[oa]|amig[oa]|'
       r'novi[oa]|prometid[oa]|prim[oa]|t[ií][oa]|abuel[oa]|sobrin[oa]|compañer[oa]|colega|niet[oa]|'
       r'suegr[oa]|cuñad[oa]|jef[ea]|gu[ií]a|chofer)s?(?:\s+(?:mayor|menor|peque[ñn][oa]))?'
       r'|\btravel(?:l)?ing\s+with|\bviaj(?:a|o|amos|ar[aá])\s+con|\bacompañad[oa]\s+(?:de|por)'
       r'|\b(?:along\s+)?with\s+my\s+(?:son|daughter|wife|husband|partner|friend)')
WEAK_CUES = [
    r"\bI\s+am", r"\bI['’]m", r'\bthis\s+is', r"\bit['’]s", r'\bsoy', r'\bhere\s+is', r'\bnamed', r'\bcalled',
    r'\bse\s+llama', r'\bllamad[oa]', r'\bask\s+for', r'\bpregunt(?:e|ar|en)\s+por', r'\bspeak\s+(?:to|with)',
    r'\bhablar\s+con', r'\batte?\s*[.:]', r'\batt\s*[.:]',
    r'(?-i:\b(?:Sr|Sra|Srta|Dr|Dra|Lic|Licda|Ing|Mr|Mrs|Ms|Mx|Prof)\.?)',
    r'(?-i:\b(?:Señor|Señora|Señorita|Sr|Don|Doña|Miss|Sir|Madam)(?=[ \t]))',
    r'^[ \t]*(?:hi|hello|hey|dear|hola|estimad[oa]s?|querid[oa]|buen(?:os|as)(?:\s+(?:d[ií]as|tardes|noches))?|'
    r'good\s+(?:morning|afternoon|evening))',
    r'\b(?:regards|saludos|atentamente|cordialmente|thanks|gracias|cheers|best|sincerely|pura\s+vida|'
    r'abrazos|bendiciones)\s*[,!.:\-–—]+',
    r'^[ \t]*(?:--|—|–|-)',
]
CUE_GAP = r'[ \t]*[,:\-–—]?[ \t]*(?:(?:is|named|called|se\s+llama|llamad[oa]|es)[ \t]+)?'
STRONG_RX = re.compile('(?:' + '|'.join(STRONG_CUES) + ')' + CUE_GAP, re.I | re.M)
REL_RX = re.compile('(?:' + REL + ')' + CUE_GAP, re.I | re.M)
WEAK_RX = re.compile('(?:' + '|'.join(WEAK_CUES) + ')' + CUE_GAP, re.I | re.M)
CLOSING_RX = re.compile(
    r'^[ \t]*(?:(?:best|kind|warm|warmest)\s+regards|regards|saludos(?:\s+cordiales)?|un\s+saludo|'
    r'atentamente|cordialmente|gracias|muchas\s+gracias|thanks|thank\s+you|many\s+thanks|cheers|best|'
    r'sincerely|yours\s+(?:truly|sincerely)|respectfully|pura\s+vida|un\s+abrazo|bendiciones|take\s+care|'
    r'abrazos|con\s+cari[ñn]o|love|all\s+the\s+best|best\s+wishes|saludos\s+cordiales)[ \t]*[,!.]*[ \t]*$',
    re.I | re.M)


def sentinel(i):
    return '' + chr(ord('a') + i % 26) * (i // 26 + 1) + ''


def keep(saved, value):
    saved.append(value)
    return sentinel(len(saved) - 1)


def protect_pv(text):
    saved = []
    return PV_RX.sub(lambda m: keep(saved, m.group(0)), text), saved


def restore_pv(text, saved):
    keys = {sentinel(i): v for i, v in enumerate(saved)}
    return SENT_RX.sub(lambda m: keys.get(m.group(0), m.group(0)), text)


def classify_run(s):
    d = re.sub(r'\D', '', s)
    n = len(d)
    if n < 8:
        return None
    if d.startswith('00506') and n == 13:
        return '[PHONE]'
    if d.startswith('506') and n == 11:
        return '[PHONE]'
    if n == 8:
        return '[PHONE]'
    if 9 <= n <= 12:
        return '[ID]'
    return '[ACCOUNT]'


def redact_runs(text):
    out = []
    pos = 0
    for m in RUN_RX.finditer(text):
        s = m.group(0)
        lead = len(s) - len(s.lstrip(' \t'))
        core = s.strip(' \t')
        tag = classify_run(core)
        if tag is None:
            groups = re.split(r'[ \t]{2,}|[ \t]*/[ \t]*', core)
            if len(groups) > 1:
                parts = [classify_run(g) or g for g in groups]
                if any(p.startswith('[') for p in parts):
                    out.append(text[pos:m.start()] + s[:lead] + ' '.join(parts))
                    pos = m.end()
            continue
        out.append(text[pos:m.start()] + s[:lead] + tag)
        pos = m.end()
    out.append(text[pos:])
    return ''.join(out)


def redact_addresses(text):
    spans = []
    for rx in ADDR_ANCHORS:
        for m in rx.finditer(text):
            if m.end() > m.start():
                spans.append([m.start(), m.end()])
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    result = []
    for s, e in merged:
        while True:
            c = COMMA_RX.match(text, e)
            if not c:
                break
            p = c.end()
            best = None
            for rx in ADDR_ANCHORS + [ADDR_EN_REL, POSTAL_RX]:
                m = rx.match(text, p)
                if m and m.end() > p:
                    best = max(best or 0, m.end())
            if best is None:
                m = CITY_RX.match(text, p)
                if m:
                    first = fold(m.group(0).split()[0].rstrip('.'))
                    if fold(m.group(0)) in CITY_SET or first not in CAPSTOP:
                        best = m.end()
            if best is None:
                break
            e = best
        base = max(0, s - 60)
        m = re.search(r"([A-ZÁÉÍÓÚÑ][\wáéíóúñü]*(?:[ \t]+(?:de[ \t]+|del[ \t]+|las?[ \t]+|los[ \t]+|el[ \t]+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñü]*){0,3})[ \t]*,[ \t]*$",
                      text[base:s])
        if m:
            c0, c1 = base + m.start(1), base + m.end(1)
            for w in re.finditer(r'[A-ZÁÉÍÓÚÑ][\wáéíóúñü]*', text[c0:c1]):
                tail = text[c0 + w.start():c1]
                if fold(tail) in CITY_SET or fold(w.group(0)) in BUILDING:
                    s = c0 + w.start()
                    break
        if result and s <= result[-1][1]:
            result[-1][1] = max(result[-1][1], e)
        else:
            result.append([s, e])
    out = []
    pos = 0
    for s, e in result:
        seg = text[s:e]
        while seg and seg[-1] in ' \t.':
            if seg.endswith('.') and re.search(r'\b(?:m|mts|av|apto|apt|bo|st|rd|ave)\.$', seg, re.I):
                break
            seg = seg[:-1]
            e -= 1
        out.append(text[pos:s] + '[ADDRESS]')
        pos = e
    out.append(text[pos:])
    text = ''.join(out)
    text = ADDR_FIELD_RX.sub(lambda m: m.group(0).replace(m.group(1), '[ADDRESS]') if '[ADDRESS]' not in m.group(1) else m.group(0), text)
    return text


class Tok:
    __slots__ = ('s', 'e', 't', 'f', 'cap', 'place', 'ph')

    def __init__(self, m, text):
        self.s, self.e, self.t = m.start(), m.end(), m.group(0)
        self.f = fold(self.t)
        self.cap = self.t[0].isupper()
        self.place = False
        self.ph = text[max(0, self.s - 1):self.s] == '[' and text[self.e:self.e + 1] == ']'


def tokenize(text):
    return [Tok(m, text) for m in TOKEN_RX.finditer(text)]


def mark_places(text, toks):
    n = len(toks)
    i = 0
    while i < n:
        hit = 0
        for L in (4, 3, 2, 1):
            if i + L <= n and adjacent_all(text, toks, i, i + L - 1):
                if ' '.join(t.f for t in toks[i:i + L]) in PLACES:
                    hit = L
                    break
        if hit:
            for k in range(i, i + hit):
                toks[k].place = True
            i += hit
            continue
        if toks[i].cap and toks[i].f in PLACE_HEADS:
            j = i + 1
            cnt = 0
            while j < n and cnt < 3 and adjacent(text, toks, j - 1, j) and (toks[j].cap or toks[j].f in PARTICLES):
                toks[j].place = True
                j += 1
                cnt += 1
        i += 1


def adjacent(text, toks, i, j):
    gap = text[toks[i].e:toks[j].s]
    return re.fullmatch(r'[ \t]+|[ \t]*-[ \t]*', gap) is not None


def adjacent_all(text, toks, i, j):
    return all(adjacent(text, toks, k, k + 1) for k in range(i, j))


EXTRA = set()

MAILWORDS = words("""
don senor senora
info reservas reservations contact contacto admin travel ventas sales hello hola mail correo support
booking bookings office team user cliente client gmail yahoo hotmail outlook icloud noreply viajes tours
familia family home casa trabajo work personal mr mrs the and
""")


def known(f):
    return f in FIRST or f in SURN or f in EXTRA


def email_hints(locals_, text):
    toks = [m.group(0) for m in TOKEN_RX.finditer(text)]
    folded = [fold(t) for t in toks]
    present = set(f for f in folded if len(f) >= 3 and f not in STOP)
    capped = set(fold(t) for t in toks if t[0].isupper())
    pieces = set()
    for local in locals_:
        for piece in re.split(r"[._\-+\d']+", fold(local)):
            if len(piece) >= 3:
                pieces.add(piece)
    hints = set()
    for piece in pieces:
        if piece in MAILWORDS or piece in CAPSTOP or piece in COMMONCAP or piece in PLACES:
            continue
        if piece in present:
            hints.add(piece)
        for k in range(len(folded) - 1):
            a, b = folded[k], folded[k + 1]
            if len(a) >= 2 and len(b) >= 3 and (a + b == piece or b + a == piece):
                hints.update([a, b])
            elif len(b) >= 3 and a[:1] + b == piece and b in capped:
                hints.add(b)
        for cut in (1, 2):
            if len(piece) - cut >= 3:
                if piece[cut:] in capped:
                    hints.add(piece[cut:])
                if piece[:-cut] in present and (piece[:-cut] in FIRST or piece[:-cut] in capped or cut == 1):
                    hints.add(piece[:-cut])
    return {h for h in hints if h not in CAPSTOP and h not in COMMONCAP and h not in STOP}


def namey(t, mode):
    if t.f in PARTICLES or t.ph:
        return False
    if t.cap:
        if t.f in CAPSTOP:
            return False
        if mode == 'weak' and t.place:
            return False
        return True
    if mode in ('strong', 'sig'):
        return known(t.f) or (t.f not in STOP and len(t.f) > 1)
    return known(t.f)


COMMONCAP = words("""
central national nacional park parque airport aeropuerto voyages team tour tours hotel lodge resort
volcano volcan beach playa river rio falls waterfall catarata canopy rafting zipline tirolesa snorkel
snorkeling excursion booking reservation reserva reference subject asunto whatsapp gmail yahoo outlook
hotmail visa mastercard paypal sinpe movil bank banco iban swift account cuenta invoice factura
dear hello hi hola thanks gracias regards saludos please new old good great best kind warm happy
north south east west norte sur este oeste street avenue road calle avenida ruta highway carretera
monday tuesday wednesday thursday friday saturday sunday lunes martes miercoles jueves viernes sabado
domingo january february march april may june july august september october november december enero
febrero marzo abril mayo junio julio agosto septiembre setiembre octubre noviembre diciembre
pura vida costa rica america united states canada mexico spain espana europe
cloud forest reserve hot springs springs wildlife refuge island bay coast pacific caribbean gulf golfo
hanging bridges bridge sky tram walk night coffee chocolate sloth sanctuary waterfall gardens garden
tree house trek hike hiking trip package paquete transfer shuttle bus van boat ferry flight vuelo
kayak paddle surf lesson lessons class tour tours adventure adventures eco ecotour safari cruise canal
canals mangrove mangroves birdwatching birding turtle turtles nesting whale whales dolphin dolphins
crocodile volcanoes mountain mountains montana cerro lake lago laguna jungle selva rainforest rain
spa resort inn suites suite room rooms villa villas apartment apartments cabin cabins camping camp
deluxe premium standard family familia group grupo private privado shared compartido full half day
dia morning afternoon sunset sunrise tour operator operations customer service care desk front
department office manager reservations bookings billing accounts finance support help center
airlines airline air express travel travels viajes turismo tourism tourist tourists guide guides
national international nacional internacional institute instituto university universidad
""")


def propery(t):
    if t.ph or t.place or t.f in CAPSTOP or t.f in COMMONCAP or t.f in PLACE_HEADS:
        return False
    w = t.t
    if len(w) < 2 or not w[0].isupper():
        return False
    if "'" in w or '’' in w:
        return False
    if w.isupper():
        return known(t.f)
    return not any(c.isupper() for c in w[1:])


def grow(text, toks, i, mode, first_lower_ok):
    n = len(toks)
    t = toks[i]
    if not t.cap and not first_lower_ok(t):
        return []
    if not namey(t, mode):
        return []
    span = [i]
    j = i
    while len(span) < 6:
        k = j + 1
        parts = []
        while k < n and toks[k].f in PARTICLES and len(parts) < 2 and adjacent(text, toks, k - 1, k):
            parts.append(k)
            k += 1
        if k >= n or not adjacent(text, toks, k - 1, k):
            break
        nt = toks[k]
        if parts and not nt.cap and not (mode in ("strong", "sig") and known(nt.f)):
            break
        if not namey(nt, mode):
            break
        if not nt.cap and not known(nt.f) and (len(span) >= 4 or any(toks[x].cap for x in span)):
            break
        span.extend(parts)
        span.append(k)
        j = k
    return span


def find_names(text):
    toks = tokenize(text)
    mark_places(text, toks)
    n = len(toks)
    by_start = {t.s: i for i, t in enumerate(toks)}
    spans = []

    def tok_at(pos):
        return by_start.get(pos)

    for m in STRONG_RX.finditer(text):
        i = tok_at(m.end())
        if i is not None:
            spans.append(grow(text, toks, i, 'strong', lambda t: t.f not in STOP))
    for m in REL_RX.finditer(text):
        i = tok_at(m.end())
        if i is not None:
            spans.append(grow(text, toks, i, 'strong', lambda t: t.f in FIRST or t.f in EXTRA))
    for m in WEAK_RX.finditer(text):
        i = tok_at(m.end())
        if i is None:
            continue
        t = toks[i]
        if t.cap:
            spans.append(grow(text, toks, i, 'weak', lambda t: False))
        elif t.f not in STOP and (known(t.f) or any(
                known(toks[k].f) and toks[k].f not in PARTICLES and adjacent_all(text, toks, i, k)
                for k in range(i + 1, min(n, i + 4)))):
            spans.append(grow(text, toks, i, 'strong', lambda t: True))
    for m in CLOSING_RX.finditer(text):
        rest = text[m.end():]
        lm = re.match(r'\s*\n([^\n]*)', rest)
        if not lm:
            continue
        line = lm.group(1)
        ls = m.end() + lm.start(1)
        lt = [i for i, t in enumerate(toks) if ls <= t.s < ls + len(line)]
        if not lt or len(lt) > 7 or '@' in line:
            continue
        spans.append(grow(text, toks, lt[0], 'sig', lambda t: t.f not in STOP))
    lines = [l for l in text.split('\n') if l.strip()]
    if lines:
        last = lines[-1]
        ls = text.rfind(last)
        lt = [i for i, t in enumerate(toks) if ls <= t.s < ls + len(last)]
        if 1 <= len(lt) <= 4 and all(toks[i].cap for i in lt) and re.fullmatch(r"[\w\s.'’-]+", last.strip()):
            if any(known(toks[i].f) for i in lt):
                spans.append(grow(text, toks, lt[0], 'sig', lambda t: False))

    for i, t in enumerate(toks):
        if t.place:
            continue
        if t.cap and (t.f in FIRST and t.f not in AMBIG or t.f in EXTRA):
            spans.append(grow(text, toks, i, 'lex', lambda t: False))
        elif t.cap and t.f in FIRST and t.f in AMBIG:
            if i + 1 < n and adjacent(text, toks, i, i + 1) and toks[i + 1].cap and propery(toks[i + 1]):
                spans.append(grow(text, toks, i, 'lex', lambda t: False))
        elif t.cap and t.f in SURN and i > 0 and adjacent(text, toks, i - 1, i):
            p = toks[i - 1]
            if p.cap and p.f not in CAPSTOP and not p.place and p.f not in PARTICLES:
                spans.append(grow(text, toks, i - 1, 'lex', lambda t: False))
        elif not t.cap and (t.f in FIRST and t.f not in AMBIG or t.f in EXTRA) and i + 1 < n \
                and adjacent(text, toks, i, i + 1) and known(toks[i + 1].f) and not toks[i + 1].cap:
            spans.append([i, i + 1])
    i = 0
    while i < n:
        if not (toks[i].cap and propery(toks[i])):
            i += 1
            continue
        run, j = [i], i
        while j + 1 < n and adjacent(text, toks, j, j + 1):
            k = j + 1
            while toks[k].f in PARTICLES and k + 1 < n and k - j < 3 and adjacent(text, toks, k, k + 1):
                k += 1
            if not (toks[k].cap and propery(toks[k])):
                break
            run.extend(range(j + 1, k + 1))
            j = k
        nxt = toks[j + 1] if j + 1 < n and adjacent(text, toks, j, j + 1) else None
        prv = toks[i - 1] if i > 0 and adjacent(text, toks, i - 1, i) else None
        blocked = any(x is not None and x.cap and (x.f in COMMONCAP or x.f in PLACE_HEADS) for x in (nxt, prv))
        if not blocked and 2 <= len([x for x in run if toks[x].f not in PARTICLES]) <= 4:
            spans.append(run)
        i = j + 1
    parts = set()
    for sp in spans:
        for i in sp:
            f = toks[i].f
            if f.endswith("'s") or f.endswith('’s'):
                f = f[:-2]
            if f in PARTICLES or len(f) < 2 or f in CAPSTOP or toks[i].ph:
                continue
            if f in STOP and not known(f):
                continue
            parts.add(f)
    return parts


def email_parts(local, text_folded_caps):
    out = set()
    for piece in re.split(r'[._\-+\d\']+', fold(local)):
        if len(piece) < 3:
            continue
        if piece in FIRST or piece in SURN:
            if piece not in AMBIG or piece in text_folded_caps:
                out.add(piece)
            continue
        if piece in text_folded_caps and len(piece) >= 4:
            out.add(piece)
            continue
        for k in range(3, len(piece) - 2):
            a, b = piece[:k], piece[k:]
            if (a in FIRST or a in SURN) and (b in FIRST or b in SURN):
                out.update([a, b])
                break
        else:
            if len(piece) > 4 and piece[1:] in SURN:
                out.add(piece[1:])
            elif len(piece) > 5 and piece[2:] in SURN:
                out.add(piece[2:])
    return out


def glued(f, parts):
    n = len(f)
    if n < 6:
        return False
    ok = [False] * (n + 1)
    ok[0] = True
    cnt = [0] * (n + 1)
    for i in range(n):
        if not ok[i]:
            continue
        for p in parts:
            if len(p) >= 3 and f.startswith(p, i):
                ok[i + len(p)] = True
                cnt[i + len(p)] = max(cnt[i + len(p)], cnt[i] + 1)
    return ok[n] and cnt[n] >= 2


BIRTH_RX = re.compile(r'(?i)born|birth|nac[ií]|nacimiento|dob|d\.o\.b|f\.?\s*nac|cumplea')


def date_rep(m, text, saved):
    y = re.findall(r'\d{4}', m.group(0))
    year = int(y[0]) if y else int(m.group(0)[-2:]) + 1900
    if year < 2023:
        return '[DOB]'
    line_start = text.rfind('\n', 0, m.start()) + 1
    if BIRTH_RX.search(text[max(line_start, m.start() - 80):m.start()]):
        return '[DOB]'
    return keep(saved, m.group(0))


def redact(text):
    text, saved = protect_pv(text)
    emails = [m.group(0) for m in EMAIL_RX.finditer(text)]
    locals_ = [re.split(r'\s*(?:@|\(at\)|\[at\])', e, maxsplit=1)[0] for e in emails]
    text = EMAIL_RX.sub('[EMAIL]', text)
    EXTRA.clear()
    EXTRA.update(email_hints(locals_, text))
    text = IBAN_RX.sub('[ACCOUNT]', text)
    text = IBAN2_RX.sub(lambda m: '[ACCOUNT]' if len(re.sub(r'\D', '', m.group(0))) >= 12 else m.group(0), text)
    for rx in DATE_RXS:
        text = rx.sub(lambda m: date_rep(m, text, saved), text)
    text = redact_addresses(text)
    text = redact_runs(text)

    parts = find_names(text)
    caps = set(fold(m.group(0)) for m in TOKEN_RX.finditer(text) if m.group(0)[0].isupper())
    for local in locals_:
        parts |= email_parts(local, caps)
    parts |= EXTRA
    parts = {p for p in parts if len(p) >= 2 and p not in PARTICLES}

    if parts:
        def rep(m):
            w = m.group(0)
            if text[max(0, m.start() - 1):m.start()] == '[' and text[m.end():m.end() + 1] == ']':
                return w
            f = fold(w)
            if f in parts or glued(f, parts):
                return '[NAME]'
            if "'" in f or '’' in f:
                head = re.split(r"['’]", f)[0]
                if head in parts:
                    return '[NAME]' + w[len(head):]
            return w
        for _ in range(3):
            text = TOKEN_RX.sub(rep, text)
            more = set()
            for m in re.finditer(r"\[NAME\][ \t]+(?:(?:de|del|de la|de los|de las)[ \t]+)?([^\W\d_]+)"
                                 r"|([^\W\d_]+)[ \t]+\[NAME\]", text):
                w = m.group(1) or m.group(2)
                f = fold(w)
                if known(f) and f not in CAPSTOP and f not in STOP and f not in PARTICLES and f not in parts:
                    more.add(f)
            for m in re.finditer(r"\[NAME\][ \t]+([^\W\d_]+)[ \t]+\[NAME\]", text):
                f = fold(m.group(1))
                if f not in CAPSTOP and f not in STOP and f not in PARTICLES and len(f) > 2:
                    more.add(f)
            if not more:
                break
            parts |= more
        text = re.sub(r'\[NAME\](?:(?:[ \t]+|[ \t]*-[ \t]*)(?:(?:de|del|de la|de los|de las|la|van|von|da)[ \t]+)?\[NAME\])+',
                      '[NAME]', text)
    return restore_pv(text, saved)


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        src, dst = args[0], args[1]
    elif os.path.exists('test-input.json'):
        src, dst = 'test-input.json', 'test-output.json'
    else:
        src, dst = 'train-input.json', 'train-pred.json'
    with open(src, encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for d in data['documents']:
        try:
            r = redact(d['text'])
        except Exception:
            r = fallback(d['text'])
        if not r.strip():
            r = '[REDACTED]'
        out[d['id']] = r
    with open(dst, 'w', encoding='utf-8') as f:
        json.dump({'redacted_documents': out}, f, ensure_ascii=False, indent=2)


def fallback(text):
    text, saved = protect_pv(text)
    text = EMAIL_RX.sub('[EMAIL]', text)
    text = re.sub(r'\d', '#', text)
    return restore_pv(text, saved)


if __name__ == '__main__':
    main()
