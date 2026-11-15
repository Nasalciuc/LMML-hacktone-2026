"""For fixed (group,seed), play 'wait' all 5 games; log Professor final score, claim order, and boat moves."""
import json, sys
sys.path.insert(0,'.')
from arena import Arena

def run(group, seed):
    with Arena() as a:
        for game in range(5):
            s=a.reset(group,seed,game)
            o0=s['observation']
            claim_order=[]  # (round, contract, who)
            traj=[]  # per round list of enemy boats
            prev=[-1]*8
            while not s['game_done']:
                o=s['observation']
                cl=o['claims']
                for c in range(len(cl)):
                    if cl[c]!=-1 and prev[c]==-1:
                        claim_order.append((o['round'],c,'me' if cl[c]==o['seat'] else 'prof'))
                        prev[c]=cl[c]
                traj.append((o['round'],[ (b['boat'],b['position'],b['destination'],b['eta']) for b in o['enemy_boats']]))
                s=a.step({'op':'wait'})
            r=s['result']
            co=' '.join('r%d:c%d/%s'%(rd,c,w) for rd,c,w in claim_order)
            print('g%d seed%d game%d scores=%s hp=%d claims: %s'%(group,seed,game,r['scores'],r['half_points'],co))
        print('  map contracts:', json.dumps(o0['map']['contracts']))
        print('  roles:', o0['map']['roles'], 'homes:', o0['map']['homes'], 'seat:', o0['seat'])

if __name__=='__main__':
    # a few groups/seeds
    cases=[(0,11),(0,767302775),(2,41326234),(6,2146175226)]
    for g,sd in cases:
        run(g,sd)
        print()
