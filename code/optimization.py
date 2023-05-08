'''# Code for NSGA-2
# Code for NSGA-3
from nsga2 import MOOCS_nsga2
def moo_ACS():
    if args.moo == 'nsga-2':
        clients = MOOCS_nsga2()
    elif args.moo == 'nsga-3':
        clients = nsga3()
    else:
        print("wrong Multi-objective optimizer")
    return clients'''