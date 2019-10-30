# Task 1

import pubchempy as pcp
import sys
from rdkit import Chem

cids_str = sys.argv
cids = []

# Convert input cid list to integer
for cid in cids_str[1:]:
    cid = int(cid)
    cids.append(cid)

# Download file in SDF format
pcp.download('SDF', 'output.sdf', cids, 'cid', overwrite=True)

# Download file in CSV format with some chosen features
pcp.download('CSV', 'output.csv', cids, operation='property/\
MolecularFormula,\
MolecularWeight,\
CanonicalSMILES,\
IUPACName,\
XLogP,\
ExactMass,\
MonoisotopicMass,\
TPSA,\
Complexity,\
Charge,\
HBondDonorCount,\
HBondAcceptorCount,\
RotatableBondCount,\
HeavyAtomCount,\
IsomericSMILES', overwrite=True)

# Convert from SDF to SMILES format
sdf_file = Chem.SDMolSupplier('output.sdf')
with open('output.smi', 'w') as file:
    for mol in sdf_file:
        smiles = Chem.MolToSmiles(mol)
        file.write("{}\n".format(smiles))



