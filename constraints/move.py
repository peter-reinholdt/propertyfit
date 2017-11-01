#!/usr/bin/env python
import sys
import sh

filename = sys.argv[1]
out = filename
if "methylcharged" in filename:
    out = "{}_methyl_charged.constraints".format(filename[:3])
elif "chargedmethyl" in filename:
    out = "{}_charged_methyl.constraints".format(filename[:3])
elif "methylneutral" in filename:
    out = "{}_methyl_neutral.constraints".format(filename[:3])
elif "neutralmethyl" in filename:
    out = "{}_neutral_methyl.constraints".format(filename[:3])
if filename != out:
    sh.cp(filename, out)
