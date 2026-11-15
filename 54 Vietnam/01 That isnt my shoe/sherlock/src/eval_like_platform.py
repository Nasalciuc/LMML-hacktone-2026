import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _here)
_sys.path.insert(0, _os.path.dirname(_here))
from paths import DATA, WEIGHTS, RUNS, EXPORT, SRC, FEATS, SUBMISSION  # noqa: E402,F401
import os, csv, warnings, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
warnings.filterwarnings('ignore')
D = f'{DATA}'
rows = list(csv.DictReader(open(os.path.join(D, 'train-output.csv'))))[:30]
for name in ['sherlock_labels', 'sherlock_probs4']:
    m = torch.load(f'{SUBMISSION}/{name}.pt', map_location='cpu', weights_only=False)
    m.eval()
    res = []
    for tf_name, tf in (('ToTensor', T.ToTensor()), ('ToTensor+Normalize', T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])),
                        ('PILToTensor(uint8)', T.PILToTensor()), ('Resize224+ToTensor', T.Compose([T.Resize(224), T.ToTensor()]))):
        preds = []
        for r in rows:
            x = tf(Image.open(os.path.join(D, 'train-input', r['filename'])).convert('RGB')).unsqueeze(0)
            with torch.no_grad():
                o = m(x)
            preds.append(o[0] if isinstance(o, list) else list(m.classes4)[int(o.argmax(1))])
        acc = np.mean([p == r['label'] for p, r in zip(preds, rows)])
        res.append(f'{tf_name}: acc={acc:.2f} unk={np.mean([p == "unknown" for p in preds]):.2f}')
    print(name, '|', ' | '.join(res))
