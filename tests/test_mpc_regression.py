from diafiltration import closed_loop, cL_max, MP, cP_star, cL_star

def test_closed_loop():
    t,V,ML,_ = closed_loop(N=20)
    assert t[-1] < 6*3600
    assert (ML/V).max() < cL_max
    assert MP/V[-1] >= cP_star
    assert ML[-1]/V[-1] <= cL_star
