"""
Python implementation of tradeSeq functions for trajectory-based differential expression analysis.

Functions:
- fitGAM: Fits GAM models per gene along pseudotime
- associationTest: Identifies genes significantly changing over time
- predictSmooth: Predicts smoothed expression values over pseudotime
- startVsEndTest: Compares early vs late expression for each gene
"""

# from . import _base, _association_test #, _pattern_test, _start_vs_end_test

# fitGAM =  _base.fitGAM 
# run_fitGAM_parallel = _base.run_fitGAM_parallel
# AssociationTest = _association_test.AssociationTest
# PseudotimeRestrictedAssociationTest = _association_test.PseudotimeRestrictedAssociationTest


from ._base import *
from ._association_test import *
from ._utils import *
