#!/usr/bin/env python

import sys
commands1 = []
commands2 = []
for line in sys.stdin:
    x = line.split('.')[0]
    if len(x) == 3:
        commands1.append("git mv {}.constraints {}_methyl_methyl.constraints".format(x,x))
    else:
        if 'ACE' in x:
            continue
        elif 'NME' in x:
            continue
        split = x.split('_')
        commands1.append("git mv {x}_{y}_{z}.constraints {x}_{z}_{y}.constraints_new".format(x=split[0], y=split[2], z=split[1]))
        commands2.append("git mv {x}_{z}_{y}.constraints_new {x}_{z}_{y}.constraints".format(x=split[0], y=split[2], z=split[1]))

print("#!/bin/bash")
for c in commands1:
    print(c)
for c in commands2:
    print(c)
