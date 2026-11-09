import importlib
import numpy as np
import pytest
from malthus_optimize import differential_evolution, minimize

@pytest.mark.parametrize("kwargs", [{"constraints":[object()]},{"integrality":[True]},{"callback":lambda *a:None},{"tol":.1},{"polish":True},{"mutation":(.5,1.)}])
def test_unsupported_controls_fail_before_execution(kwargs):
    with pytest.raises(NotImplementedError):
        differential_evolution(lambda x:0.,[(-1,1)],**kwargs)

def test_budget_completion_is_not_convergence(monkeypatch):
    module=importlib.import_module("malthus_optimize.api.minimize")
    monkeypatch.setattr(module,"run_composer_optimization",lambda **k:(np.array([0.]),0.,20,2,[]))
    result=minimize(lambda x:0.,bounds=[(-1,1)],options={"maxiter":2})
    assert result.success is False and result.status == 1

def test_weighted_covariance_matches_analytic_reference(monkeypatch):
    module=importlib.import_module("malthus_optimize.api.curve_fit")
    x=np.arange(6.)
    y=2*x+1+np.array([.1,-.2,.1,.2,-.1,.05])
    sigma=np.arange(1.,7.)
    X=np.column_stack([x,np.ones(6)])
    weighted=X/sigma[:,None]
    popt=np.linalg.lstsq(weighted,y/sigma,rcond=None)[0]
    monkeypatch.setattr(module,"run_composer_optimization",lambda **k:(popt,0.,10,2,[]))
    _,cov=module.curve_fit(lambda x,a,b:a*x+b,x,y,bounds=[(-5,5),(-5,5)],sigma=sigma,absolute_sigma=True)
    np.testing.assert_allclose(cov,np.linalg.inv(weighted.T@weighted),rtol=1e-7)
