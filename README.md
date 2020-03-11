# PropertyFit

[![Build Status](https://travis-ci.org/peter-reinholdt/propertyfit.svg?branch=master)](https://travis-ci.org/peter-reinholdt/propertyfit)

Python program to fit multipole moments and polarizabilities.

## Naming conventions

See the tables below for explanations of the amino acid abbrevtiations.
The standard names (e.g., `VAL`) can be prefixed to indicate terminal amino acids, for example `CVAL` corresponds to a C-terminal Valine.
Th officially supported names are given in the lists below, but names from Charmm, Amber and Gromos should also work. 
Most atom type names from these force fields should also be fine.

### Terminals
|Prefix|Type|
|------|----|
|   | internal amino acid |
| C | charged C-terminal amino acid|
| N | charged N-terminal amino acid|
| c | neutral C-terminal amino acid|
| n | neutral N-terminal amino acid|
| A | methyl capped C-terminal|
| B | methyl capped N-terminal|

### Common protonation states
|Abbreveviation|Full Name|Charge|
|--------------|---------|------:|
|ALA|Alanine|0|
|ARG|Arginine|+1|
|ASN|Asparagine|0|
|ASP|Aspartic acid|-1|
|CYS|Cysteine|0|
|GLN|Glutamine|0|
|GLU|Glutamic acid|-1|
|GLY|Glycine|0|
|HIS/HIP|Histidine|1|
|ILE|Isoleucine|0|
|LEU|Leucine|0|
|LYS|Lysine|1|
|MET|Methionine|0|
|PHE|Phenylalanine|0|
|PRO|Proline|0|
|SER|Serine|0|
|THR|Threonine|0|
|TRP|Tryptophan|0|
|TYR|Tyrosine|0|
|VAL|Valine|0|

### Other supported protonation states
|Abbreveviation|Full Name|Charge|
|--------------|---------|------:|
|ASH| protonated Aspartic acid | 0 |
|CYD|de-protonated Cysteine |-1|
|CYX|sulfur-bridged Cysteine |0|
|HID|Histidine, protonated at delta nitrogen |0|
|HIE|Histidine, protonated at epsilon nitrogen |0|
|GLH|protonated Glutamic acid |0|
|LYD|de-protonated Lysine |0|
