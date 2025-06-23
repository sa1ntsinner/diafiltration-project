from diafiltration import closed_loop

def test_closed_loop():
    t, *_ = closed_loop(N=20)
    assert t[-1] < 6*3600      # batch within 6 h
