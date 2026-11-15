"""Does the Professor react to our moves? Compare its claim timeline under different our-policies."""
import json, sys
sys.path.insert(0,'.')
from arena import Arena

def bfs_next(routes,a,b):
    n=len(routes); dist={a:0}; first={a:None}; seen=set()
    while True:
        cur=None
        for v in dist:
            if v not in seen and (cur is None or dist[v]<dist[cur]): cur=v
        if cur is None: break
        seen.add(cur)
        for w in range(n):
            if routes[cur][w] and w not in seen:
                d=dist[cur]+routes[cur][w]
                if w not in dist or d<dist[w]: dist[w]=d; first[w]=w if cur==a else first[cur]
    return first.get(b)

def play(a,group,seed,game,policy):
    s=a.reset(group,seed,game); o0=s['observation']
    prof_moves=[]  # (round, tuple of enemy boat states)
    claims=[]; prev=[-1]*8
    while not s['game_done']:
        o=s['observation']
        prof_moves.append((o['round'],tuple((b['position'],b['destination'],b['eta']) for b in o['enemy_boats'])))
        for c in range(8):
            if o['claims'][c]!=-1 and prev[c]==-1:
                claims.append((o['round'],c)); prev[c]=o['claims'][c]
        act=policy(o)
        if act not in o['legal_actions']: act={'op':'wait'}
        s=a.step(act)
    return tuple(prof_moves), tuple(claims), tuple(s['result']['scores'])

def p_wait(o): return {'op':'wait'}
def p_boat(o): return {'op':'build_boat'} if (o['round']==0 and {'op':'build_boat'} in o['legal_actions']) else {'op':'wait'}
def make_sail(routes):
    # keep both boats shuttling to far sites to contest stock
    def pol(o):
        for b,bt in enumerate(o['own']['boats']):
            if bt['eta']==0:
                # sail to a manufacturing site or wherever
                sails=[x for x in o['legal_actions'] if x['op']=='sail' and x['boat']==b]
                if sails: return sails[(o['round']+b)%len(sails)]
        return {'op':'wait'}
    return pol

if __name__=='__main__':
    cases=[(0,11),(2,41326234),(6,2146175226)]
    with Arena() as a:
        for g,sd in cases:
            s=a.reset(g,sd,0); routes=s['observation']['map']['routes']
            sail=make_sail(routes)
            for game in range(5):
                pm_w,cl_w,sc_w=play(a,g,sd,game,p_wait)
                pm_b,cl_b,sc_b=play(a,g,sd,game,p_boat)
                pm_s,cl_s,sc_s=play(a,g,sd,game,sail)
                react_boat = pm_w!=pm_b
                react_sail = pm_w!=pm_s
                # first round where prof differs under sail
                diffr=None
                for i in range(min(len(pm_w),len(pm_s))):
                    if pm_w[i]!=pm_s[i]: diffr=pm_w[i][0]; break
                print('g%d sd%d game%d  profscore wait=%d boat=%d sail=%d  reacts_to_boat=%s reacts_to_sail=%s firstdiff_r=%s'%(
                    g,sd,game,sc_w[1],sc_b[1],sc_s[1],react_boat,react_sail,diffr))
