import numpy as np
from diafiltration import flux_permeate

def test_flux():
    assert np.isclose(flux_permeate(10.0), 1/60, rtol=0.05)
