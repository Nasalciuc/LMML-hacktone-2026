import csv
from PIL import Image, ImageDraw
D="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/task_japan/dataset/anonymous/"
T="C:/Users/user/projects/LMML-hacktone-2026/30 Japan/01 City of Subtitles/attribution_table.csv"
rows=[r for r in csv.DictReader(open(T)) if r['photographer']=='toorisugari']
tw,th=640,480
for wk in range(8,13):
    rr=[r for r in rows if r['file'].startswith('w%02d'%wk)]
    sheet=Image.new('RGB',(3*tw,3*th))
    for i,r in enumerate(rr):
        im=Image.open(D+r['file']).convert('RGB').resize((tw,th),Image.LANCZOS)
        d=ImageDraw.Draw(im); d.rectangle((0,0,90,14),fill=(0,0,0)); d.text((3,2),r['file'][:7],fill=(255,255,0))
        sheet.paste(im,((i%3)*tw,(i//3)*th))
    sheet.save('tw%02d.jpg'%wk,quality=90)
