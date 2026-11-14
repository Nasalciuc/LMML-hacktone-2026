"""Synthetic messages in the style of the training set, with harder variations.

python gen_synth.py N SEED OUTPREFIX [--holdout-only]
writes OUTPREFIX-input.json, OUTPREFIX-gold.json, OUTPREFIX-items.json
"""
import json
import random
import sys
import unicodedata

KNOWN_FIRST = ['Carlos', 'Isabella', 'Diego', 'Luis', 'Adriana', 'Gabriel', 'Rocío', 'Lucía', 'Mauricio', 'Valeria',
               'Tatiana', 'Fernanda', 'Natalia', 'Rodrigo', 'Javier', 'Sofía', 'Esteban', 'Gabriela', 'Ignacio',
               'Pilar', 'Emilio', 'Álvaro', 'María', 'José', 'Ana', 'Andrés', 'Daniela', 'Jimena', 'Óscar', 'Raúl',
               'Mónica', 'Héctor', 'Josué', 'Melissa', 'Randall', 'Priscilla', 'Paola', 'Juan', 'Sebastián',
               'Michael', 'Sarah', 'Emily', 'Thomas', 'Olivia', 'Rosa', 'Mercedes', 'Ángel', 'Luz']
HELD_FIRST = ['Keylor', 'Yendry', 'Greivin', 'Dayana', 'Yorleny', 'Maikol', 'Xinia', 'Olman', 'Wilber',
              'Marjorie', 'Kattia', 'Yerlin', 'Jafeth', 'Heiner', 'Arelys']
KNOWN_SUR = ['Solano', 'Hernández', 'Araya', 'Quesada', 'Hidalgo', 'Jiménez', 'Obando', 'Guzmán', 'Ureña', 'Vargas',
             'Zúñiga', 'Cordero', 'Segura', 'Fonseca', 'Núñez', 'Chaves', 'Brenes', 'Mena', 'Castro', 'Paniagua',
             'Rodríguez', 'Mora', 'Rojas', 'Alfaro', 'Calderón', 'Campos', 'Flores', 'Villalobos', 'Bolaños',
             'Esquivel', 'Montero', 'Murillo', 'Salazar', 'Umaña', 'Smith', 'Johnson', 'Carrillo', 'Valle']
HELD_SUR = ['Matamoros', 'Xatruch', 'Zeledón', 'Barquero', 'Picado', 'Ulate', 'Cascante', 'Arguedas', 'Sibaja',
            'Jinesta']
CITIES = ['Quepos', 'Heredia', 'Limón', 'Escazú', 'Puntarenas', 'San José', 'Alajuela', 'Cartago', 'Liberia',
          'Santa Ana', 'Grecia', 'Nicoya', 'Turrialba', 'San Pedro', 'Curridabat', 'Pérez Zeledón']
SHOPS = ['Panadería La Palmera', 'Supermercado El Trigal', 'Supermercado La Esquina', 'Panadería Del Valle',
         'Ferretería El Clavo', 'Pulpería La Amistad', 'Farmacia Fischel', 'Soda Doña Tere', 'Banco Nacional',
         'Escuela Juan Rafael Mora', 'Plaza de Deportes', 'Clínica Bíblica', 'Gasolinera Uno']
GREET = ['Hello Pura Vida Voyages team,', 'Hi Pura Vida Voyages,', 'Dear Pura Vida Voyages team,',
         'Hola equipo de Pura Vida Voyages,', 'Buenas tardes, Pura Vida Voyages:', 'Estimados señores de Pura Vida Voyages,']
TOPICS = [
    'I would like to book a guided hike to the Arenal volcano for two people, and I also want to reserve a shuttle from San José. Please add both to our itinerary.',
    'We already booked the Tortuguero canal excursion, and now I want to book a second day trip. Could you check that our reservation covers the guided walk as well, and send the updated itinerary for the week?',
    'I cancelled my trip last month and I am still waiting for my refund. I have written twice asking you to reimburse the deposit, but no reimbursement has arrived yet.',
    'The tour was called off by your team, so I am requesting a refund of the full amount. Please reimburse everything to my account as we agreed.',
    'We need to reschedule our canopy tour because our flight changed. Could we postpone it by two days? A different date later that week would also work for us.',
    'I am writing to change the date of our visit to Manuel Antonio. If we postpone, can you move the tour to the new dates at the end of the month?',
    'I want to file a complaint about the snorkeling excursion. The guide arrived very late and the poor service ruined the day — we are deeply disappointed.',
    'This is a formal complaint about our stay in Tamarindo. The hotel was unacceptable and the bad experience left our family upset.',
    'I received the invoice for our package and I want to pay the deposit today. Could you confirm the wire transfer details before I send the payment?',
    'My bank says the charge went through twice. Please check the billing on my account and confirm the corrected payment when it is settled.',
    'Quiero reservar el tour al volcán Arenal para dos personas y también el transporte desde San José hasta La Fortuna.',
    'Cancelé el viaje a Monteverde hace un mes y todavía espero el reembolso del depósito.',
    'Necesitamos reprogramar el tour de canopy porque nos cambiaron el vuelo; ¿podemos posponerlo dos días?',
    'Quiero poner una queja: el guía en Tortuguero llegó tarde y el servicio fue inaceptable.',
    'Recibí la factura del paquete y quiero hacer el pago del depósito hoy mismo por transferencia.',
]
URGENCY = ['Please act immediately.', 'It is time-sensitive, so please respond as soon as possible.',
           'This is urgent, so please treat it as a priority.', 'No hurry at all — handle it at your convenience.',
           'There is no rush; whenever you can is fine.', 'Es urgente, por favor respondan lo antes posible.',
           'No hay prisa, cuando puedan.', None, None]
MONTHS_EN = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']
MONTHS_ES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'setiembre', 'octubre',
             'noviembre', 'diciembre']
REL_EN = ['son', 'daughter', 'wife', 'husband', 'friend', 'sister', 'brother', 'mother', 'partner']
REL_ES = ['hijo', 'hija', 'esposa', 'esposo', 'amiga', 'hermano', 'mamá', 'pareja']
CLOSE = ['Regards,', 'Saludos,', 'Atentamente,', 'Pura vida,', 'Gracias,', 'Best regards,', 'Thanks!', 'Cheers,']


def strip(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


class Doc:
    def __init__(self):
        self.src, self.gold, self.items = [], [], []

    def add(self, *parts):
        for p in parts:
            if isinstance(p, tuple):
                cat, val = p
                self.src.append(val)
                self.gold.append('[' + cat + ']')
            else:
                self.src.append(p)
                self.gold.append(p)

    def item(self, cat, val):
        if [cat, val] not in self.items:
            self.items.append([cat, val])


def person(r, held_p):
    pool_f = HELD_FIRST if r.random() < held_p else KNOWN_FIRST
    pool_s = HELD_SUR if r.random() < held_p else KNOWN_SUR
    first = [r.choice(pool_f)]
    if r.random() < 0.2:
        first.append(r.choice(KNOWN_FIRST + HELD_FIRST))
    sur = [r.choice(pool_s)]
    x = r.random()
    if x < 0.35:
        sur.append(r.choice(KNOWN_SUR + HELD_SUR))
    elif x < 0.42:
        sur = r.choice([['del', 'Valle'], ['de la', 'Cruz'], [sur[0], 'de la', 'Rosa']])
    return first, sur


def full(p):
    return ' '.join(p[0] + p[1])


def short(p):
    return p[0][0] + ' ' + next(s for s in p[1] if s[0].isupper())


def phone(r):
    d = r.choice('2678') + ''.join(r.choice('0123456789') for _ in range(7))
    a, b = d[:4], d[4:]
    forms = [f'+506 {a}-{b}', f'{a}-{b}', d, f'+506{d}', f'(506) {a} {b}', f'506-{a}-{b}', f'{a} {b}',
             f'+506 {a} {b}', f'00506 {a}-{b}', f'{a}.{b}']
    return d, forms


def cedula(r):
    p = r.choice('123456789')
    a = ''.join(r.choice('0123456789') for _ in range(4))
    b = ''.join(r.choice('0123456789') for _ in range(4))
    return [f'{p}-{a}-{b}', f'{p}{a}{b}', f'{p} {a} {b}', f'0{p}-{a}-{b}']


def dob(r):
    y, m, d = r.randint(1940, 2006), r.randint(1, 12), r.randint(1, 28)
    return [f'{MONTHS_EN[m - 1]} {d}, {y}', f'{d} {MONTHS_EN[m - 1]} {y}', f'{d:02d}/{m:02d}/{y}',
            f'{y}-{m:02d}-{d:02d}', f'{d} de {MONTHS_ES[m - 1]} de {y}', f'{d:02d}-{m:02d}-{y}',
            f'{MONTHS_ES[m - 1]} {d}, {y}', f'{d:02d}.{m:02d}.{y}']


def iban(r):
    d = ''.join(r.choice('0123456789') for _ in range(20))
    g = 'CR' + d[:2] + ' ' + ' '.join(d[i:i + 4] for i in range(2, 18, 4)) + ' ' + d[18:]
    return [g, 'CR' + d, g.replace(' ', '-'), 'cr' + d]


def address(r):
    city = r.choice(CITIES)
    street = r.choice([f'Avenida {r.randint(1, 40)}', f'Calle {r.randint(1, 40)}', 'Calle Central',
                       'Avenida Segunda', f'Calle {r.randint(1, 30)}A'])
    m = r.choice([100, 150, 200, 250, 300, 400, 50, 75])
    d = r.choice(['norte', 'sur', 'este', 'oeste'])
    lm = r.choice([
        f'{m} metros {d} del Parque Central',
        f'Contiguo a la {r.choice(SHOPS)}' if r.random() < 0.5 else f'Contiguo al {r.choice(SHOPS)}',
        f'Frente al {r.choice(SHOPS)}',
        f'{m} metros {d} de la Iglesia de {city}',
        f'del {r.choice(SHOPS)} {m} m {d} y 50 {r.choice(["este", "oeste"])}',
        f'{m} m al {d} de la escuela',
        f'Diagonal a la {r.choice(SHOPS)}',
        f'Costado {d} del {r.choice(SHOPS)}',
    ])
    house = r.choice(['', '', f'casa {r.randint(1, 99)}', 'casa color verde', f'apartamento {r.randint(1, 12)}B'])
    style = r.random()
    if style < 0.5:
        chunks = [street, lm] + ([house] if house else []) + [city]
    elif style < 0.7:
        chunks = [f'Barrio {r.choice(["Escalante", "Los Yoses", "La California", "Amón", "Otoya"])}', lm, city]
    elif style < 0.85:
        chunks = [f'Residencial {r.choice(["Los Robles", "Las Flores", "Monte Real", "Vista Verde"])}',
                  f'casa {r.randint(1, 60)}', city]
    else:
        chunks = [street, f'{m} meters {r.choice(["north", "south", "east", "west"])} of the Central Park', city]
    return ', '.join(chunks)


def pv(r):
    if r.random() < 0.5:
        return 'PV-' + ''.join(r.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3)) + f'{r.randint(0, 99):02d}'
    return f'PV-{r.randint(10000, 99999)}'


def make_doc(r, held_p):
    doc = Doc()
    me = person(r, held_p)
    other = person(r, held_p)
    ref = pv(r)
    my_full = full(me)
    doc.item('NAME', my_full)
    blocks = []

    def name_mention(p, form=None):
        v = form or r.choice([full(p), full(p), short(p)])
        return ('NAME', v)

    blocks.append([r.choice(['My name is ', 'Me llamo ', 'Soy ', 'This is ', 'Mi nombre es ', "I'm "]),
                   name_mention(me, my_full if r.random() < 0.8 else my_full.lower()), '.'])
    fn, ln = strip(me[0][0]).lower(), strip([s for s in me[1] if s[0].isupper()][0]).lower()
    em = r.choice([f'{fn}.{ln}@correo.cr', f'{fn}{ln}{r.randint(1, 99)}@gmail.com', f'{fn[0]}.{ln}@yahoo.com',
                   f'{ln}.{fn}@hotmail.com', f'{fn}_{ln}@outlook.com', f'{fn}{ln[0]}@ice.co.cr'])
    doc.item('EMAIL', em)
    blocks.append([r.choice(['You can reach me at ', 'Mi correo es ', 'Write to ']), ('EMAIL', em),
                   r.choice(['.', ' if anything changes.'])])
    if r.random() < 0.8:
        d, forms = phone(r)
        f1 = r.choice(forms)
        doc.item('PHONE', f1)
        b = [r.choice(['My phone number is ', 'Mi celular es ', 'Please link my callback number ', 'Call me on ']),
             ('PHONE', f1), r.choice([' and you can call me any morning.', '.', ' to the booking.'])]
        if r.random() < 0.3:
            b += [' WhatsApp works too: ', ('PHONE', r.choice(forms)), '.']
        blocks.append(b)
    if r.random() < 0.6:
        cf = cedula(r)
        doc.item('ID', cf[0])
        b = [r.choice(['My cédula number is ', 'Mi cédula es ', 'ID: ']), ('ID', r.choice(cf)),
             r.choice([' if you need to verify my identity.', '.'])]
        if r.random() < 0.25:
            b += [' (sin guiones: ', ('ID', cf[1]), ')']
        blocks.append(b)
    if r.random() < 0.6:
        df = dob(r)
        doc.item('DOB', df[3])
        b = [r.choice(['My date of birth is ', 'Nací el ', 'Date of birth: ']), ('DOB', r.choice(df)),
             r.choice([', in case you need it for the passenger list.', '.'])]
        if r.random() < 0.2:
            b += [' That is ', ('DOB', r.choice(df)), ' on the form.']
        blocks.append(b)
    if r.random() < 0.6:
        ib = iban(r)
        doc.item('ACCOUNT', ib[1])
        b = [r.choice(['The balance should go to account ', 'Mi cuenta IBAN es ', 'The payment should come from account ']),
             ('ACCOUNT', r.choice(ib)), '.']
        if r.random() < 0.2:
            b += [' Sin espacios: ', ('ACCOUNT', ib[1]), '.']
        blocks.append(b)
    if r.random() < 0.6:
        a = address(r)
        doc.item('ADDRESS', a)
        blocks.append(r.choice([
            ['Please send the paperwork to ', ('ADDRESS', a), '.'],
            ['Please add the pickup at ', ('ADDRESS', a), ' to our booking.'],
            ['Mi dirección es ', ('ADDRESS', a), '.'],
            ['We are staying at ', ('ADDRESS', a), ' until Friday.'],
        ]))
    if r.random() < 0.6:
        doc.item('NAME', full(other))
        k = r.random()
        if k < 0.4:
            blocks.append(['My ', r.choice(REL_EN), ' ', name_mention(other, full(other)), ' will travel with me.'])
        elif k < 0.6:
            blocks.append(['Mi ', r.choice(REL_ES), ' ', name_mention(other, full(other)), ' viaja conmigo.'])
        elif k < 0.75:
            blocks.append(['Please put ', ('NAME', full(other)), ' on the passenger list as well.'])
        elif k < 0.9:
            blocks.append(['My ', r.choice(REL_EN), ', ', ('NAME', full(other)), ', will join us. If I do not answer, ',
                           ('NAME', other[0][0]), ' can confirm.'])
        else:
            blocks.append([('NAME', full(other)), ' will join us on the second day.'])
    blocks.append(r.choice([
        ['Please keep reference ', ref, ' on file.'],
        ['Keep reference ', ref, ' attached to this message.'],
        ['La referencia es ', ref, '.'],
    ]))
    if r.random() < 0.3:
        blocks.append(['If I am not around, ask for ', ('NAME', r.choice([me[0][0], me[0][0].lower()])), ' at reception.'])
    if r.random() < 0.2:
        hon = r.choice(['Sr.', 'Sra.', 'Mr.', 'Ms.', 'Don', 'Doña'])
        blocks.append(['The invoice should say ', hon, ' ', ('NAME', [s for s in me[1] if s[0].isupper()][0]), '.'])
    r.shuffle(blocks)
    if r.random() < 0.15:
        doc.add('From: ', ('NAME', my_full), ' <', ('EMAIL', em), '>\nSubject: ', ref, '\n\n')
    doc.add(r.choice(GREET), '\n\n', r.choice(TOPICS), '\n\n')
    for b in blocks:
        doc.add(*b)
        doc.add('\n\n')
    u = r.choice(URGENCY)
    if u:
        doc.add(u, '\n\n')
    sig = r.choice([my_full, my_full, me[0][0], me[0][0].lower(), short(me)])
    if r.random() < 0.15:
        doc.add('— ', ('NAME', sig))
    else:
        doc.add(r.choice(CLOSE), '\n', ('NAME', sig))
    return doc


def main():
    n, seed, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    held_p = 1.0 if '--holdout-only' in sys.argv else 0.3
    r = random.Random(seed)
    docs, gold, items = [], {}, {}
    for i in range(n):
        d = make_doc(r, held_p)
        did = f'syn_{seed}_{i:04d}'
        docs.append({'id': did, 'text': ''.join(d.src)})
        gold[did] = ''.join(d.gold)
        items[did] = d.items
    json.dump({'documents': docs}, open(prefix + '-input.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'redacted_documents': gold}, open(prefix + '-gold.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(items, open(prefix + '-items.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
