"""Hand-written free-form messages that do not follow the training templates.

python hard_cases.py OUTPREFIX   -> OUTPREFIX-input.json / -gold.json / -items.json
Markup: {N:name} {E:email} {P:phone} {I:cedula} {D:dob} {A:address} {C:account}
"""
import json
import re
import sys

CAT = {'N': 'NAME', 'E': 'EMAIL', 'P': 'PHONE', 'I': 'ID', 'D': 'DOB', 'A': 'ADDRESS', 'C': 'ACCOUNT'}

CASES = [
"""Hola equipo,

Les escribe {N:Yamileth Chinchilla Rojas}. Quiero reservar el tour de rafting en el río Pacuare para el 14 de marzo de 2026, somos 4 personas. Mi teléfono es {P:(506) 8834 1290} y mi correo {E:yami.chinchilla@gmail.com}.

¡Pura vida!
{N:Yami}""",

"""Hi there,

I booked the Monteverde cloud forest night walk (ref PV-48211) but our flight AV 627 lands late, so we need to reschedule to the next evening if possible. You can text me at {P:+1 415 555 0199} or call my Costa Rican number {P:8712 3456}.

Thanks,
{N:Mark Davidson}""",

"""Buenas,

Soy {N:Greivin Arce}, mi cédula es {I:206540321}. Ayer me cobraron dos veces el pago del tour a Tortuguero, referencia PV-90233. Por favor revisen el cobro y me devuelven la diferencia a la cuenta {C:CR05015202001026284066}.

Gracias,
{N:G. Arce}""",

"""Dear team,

This is a complaint. Our guide, {N:Minor}, was rude and the boat to Tortuguero left without us. My wife {N:Lucía Bermúdez} and I are very disappointed. Booking PV-11872.

Send the written reply to {A:Barrio Los Yoses, de la Fuente de la Hispanidad 200 m sur y 25 oeste, casa esquinera, San Pedro}.

{N:Esteban Rojas Mora}
{P:7020-1188}""",

"""Pura Vida Voyages,

I want a refund for PV-30011. The zipline at Arenal was closed. Account holder: {N:Hannia Villalobos}, IBAN {C:CR21 0152 0200 1026 2840 66}.

My DOB is {D:04/11/1983} and cédula {I:1-1234-0567} if needed.

{N:hannia}""",

"""Hola!

Me llamo {N:josé pablo zamora} y quiero cambiar la fecha de mi tour en Manuel Antonio, ¿se puede posponer una semana? Mi número es {P:8888-2121}. Nací el {D:3 de julio de 1990}.

Saludos cordiales,
{N:José Pablo Zamora}""",

"""Hello,

Quick question about the invoice for PV-77120: can I pay by SINPE Móvil to {P:7101-2020}? My name is {N:Sarah O'Connor}, email {E:sarah.oconnor@outlook.com}.

Best,
{N:Sarah}""",

"""Estimados,

Por favor envíen la factura a nombre de {N:Kattia Mora Calvo}, cédula {I:3 0456 0789}, dirección {A:Avenida 10, 50 metros norte de la Clínica Bíblica, San José}.

El pago lo hará mi esposo {N:Olman Cascante}.

Atentamente,
{N:Kattia Mora}""",

"""Team,

URGENT: my daughter {N:Valentina} lost her passport before the Tamarindo snorkel trip (PV-55190). Please call me immediately at {P:+506 6123 4567}.

{N:Ricardo Elizondo}""",

"""Hi Pura Vida Voyages,

We had a great time in La Fortuna! But the shuttle from San José never came and we had to pay a taxi. I'd like a refund of the shuttle portion, booking PV-20934.

Contact: {N:Dayana Ulate} ({E:dayanau@ice.co.cr}, {P:6044 7711})""",

"""Hola, buenos días.

Mi mamá, doña {N:Xinia Sibaja}, viaja con nosotros al volcán Poás. Ella nació el {D:1948-02-29}. ¿Hay descuento de adulto mayor? Reserva PV-60117.

Muchas gracias,
{N:Maikol Sibaja}
{E:maikol.sibaja@yahoo.com}""",

"""Dear Sir or Madam,

I am writing to formally complain about the Corcovado tour. The lodge in Drake Bay had no electricity. I demand a full refund to {C:CR33 0815 0001 1234 5678 90}.

Yours sincerely,
{N:Victoria Hughes}
{A:Calle 5, 100 meters south of the Central Park, Liberia}""",

"""Hey,

{N:Diego} here, {N:Carla Montoya}'s husband. She booked the canopy tour PV-14455 but she's sick, can we move it to Friday? Call me {P:83456789} no rush.

{N:Diego Fallas}""",

"""Buenas tardes,

Necesito el comprobante del pago del depósito (PV-81220). Mis datos: {N:Marjorie Picado Sánchez}, céd. {I:4-0198-0456}, tel. {P:2260 4411}, correo {E:mpicado@correo.cr}.
Dirección: {A:Residencial Monte Real, casa 23, Heredia}.

Gracias de antemano.""",

"""Hello,

My son {N:jafeth} and his friend {N:Heiner Barquero} are booked on the Arenal hike (PV-33019). Can you confirm the pickup at {A:Hotel Las Colinas, 300 m oeste de la iglesia de La Fortuna} at 7:00 am?

Thanks!
{N:Arelys Barquero}""",

"""Good afternoon,

I need to change our Tortuguero package dates; we were going on April 2, 2026 and now need April 9, 2026. Reference PV-72001. The booking is under {N:Thomas Whitaker}.

Regards,
{N:Tom Whitaker}
Mobile: {P:+44 7700 900123}""",

"""Hola PV,

El guía {N:Randall} nos dejó esperando dos horas en Cahuita. Es inaceptable. Quiero que me reembolsen. Mi IBAN es {C:CR 83 0152 0212 3456 7890 12}. Referencia PV-40404.

{N:Yorleny Matamoros}
cel {P:8 7 6 5 4 3 2 1}""",

"""Hi,

Please add my mother {N:Rosa Elena Quirós} (born {D:17.05.1952}) to the Manuel Antonio tour, PV-51515. Her cédula is {I:105230876}.

Thank you,
{N:Luis Diego Quirós}""",
]


def parse(src):
    out_s, out_g, items = [], [], []
    pos = 0
    for m in re.finditer(r'\{([NEPIDAC]):([^{}]+)\}', src):
        out_s.append(src[pos:m.start()])
        out_g.append(src[pos:m.start()])
        cat, val = CAT[m.group(1)], m.group(2)
        out_s.append(val)
        out_g.append('[' + cat + ']')
        if [cat, val] not in items:
            items.append([cat, val])
        pos = m.end()
    out_s.append(src[pos:])
    out_g.append(src[pos:])
    return ''.join(out_s), ''.join(out_g), items


def main():
    prefix = sys.argv[1]
    docs, gold, items = [], {}, {}
    for i, c in enumerate(CASES):
        s, g, it = parse(c)
        did = f'hard_{i:02d}'
        docs.append({'id': did, 'text': s})
        gold[did] = g
        items[did] = it
    json.dump({'documents': docs}, open(prefix + '-input.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump({'redacted_documents': gold}, open(prefix + '-gold.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(items, open(prefix + '-items.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
