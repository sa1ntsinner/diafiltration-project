from src import ProcessParameters, DiafiltrationModel, MPCController

def test_mpc_returns_feasible_u():
    p = ProcessParameters()
    m = DiafiltrationModel(p)
    ctrl = MPCController(m, p, N=5)
    u = ctrl.control([p.V0, p.c_L0])
    assert 0.0 <= u <= 1.0
