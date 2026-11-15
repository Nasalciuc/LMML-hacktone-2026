import json,sys,importlib.util
sys.path.insert(0,'.')
from arena import Arena
spec=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
g,sd,game=int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
with Arena() as a:
    s=a.reset(g,sd,game); mem={}
    log=[]
    while not s['game_done']:
        o=s['observation']; act=m.agent(o,mem)
        if o['round']<80:
            own=o['own']
            ncraft=sum(own['jobs']); prod=[(i,w[2]) for i,w in enumerate(own['warehouses']) if w[2]]
            if act['op'] in ('build_workshop','craft','deliver','build_boat'):
                log.append('r%d %s score=%d'%(o['round'],json.dumps(act),own['score']))
        s=a.step(act)
    r=s['result']
    print('result',r['scores'],'hp',r['half_points'])
    print('nboats_end',len(o['own']['boats']),'workshops',[i for i,x in enumerate(o['own']['workshops']) if x])
    for L in log: print(' ',L)
