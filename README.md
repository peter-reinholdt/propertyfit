# PropertyFit

[![Build Status](https://travis-ci.org/peter-reinholdt/propertyfit.svg?branch=master)](https://travis-ci.org/peter-reinholdt/propertyfit)

Python program to fit multipole moments and polarizabilities.

## Naming conventions

List of amino acid namings. Names from Charmm, Amber and Gromos should also work. 
If the name of an amino acid is longer than four characters, then "C", "N", "c", "n", "A" or "B" cannot be put in front.  
Atom names from these force fields should also be fine.

### Terminals

C"XXX", charged C-terminal amino acid

N"XXX", charged N-terminal amino acid

c"XXX", neutral C-terminal amino acid

n"XXX", neutral N-terminal amino acid

A"XXX", methyl capped C-terminal I think (might not be fitted)

B"XXX", methyl capped N-terminal I think (might not be fitted) 

### Common protonation states

ALA, Alanine

ARG, Arginine (+1 charge)

ASN, Asparagine

ASP, Aspartic acid (-1 charge)

CYS, Cystine

GLN, Glutamine

GLU, Glutamic acid (-1 charge)

GLY, Glycine

HIS, Histidine (+1 charge), protonated at delta nitrogen and epsilon nitrogen

ILE, Isoleucine

LEU, Leucine

LYS, Lysine (+1 charge)

MET, Methionine

PHE, Phenylalanine

PRO, Proline

SER, Serine

THR, Threonine

TRP, Tryptophan

TYR, Tyrosine

VAL, Valine

### Other supported protonation states

ASH, protonated Aspartic acid

CYD, de-protonated Cystine (-1 charge)

CYX, sulfur-bridged Cystine

HID, Histidine, protonated at delta nitrogen

HIE, Histidine, protonated at epsilon nitrogen

GLH, protonated Glutamic acid

LYD, de-protonated Lysine