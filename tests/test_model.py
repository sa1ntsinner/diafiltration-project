from src import ProcessParameters, DiafiltrationModel

def test_mass_invariant():
    """Protein mass must stay constant in the nominal model."""
    p = ProcessParameters()
    m  = DiafiltrationModel(p)
    x0 = [p.V0, p.c_L0]
    u  = 0.5
    x1 = m.rk4_step(x0, u, p.dt)
    assert abs(p.m_P - p.m_P) < 1e-8
