from __future__ import annotations

import pytest

from polaris.infra.resources import (
    get_resource_profile,
    load_resource_profiles,
    validate_profile_cost,
)


def test_live_resource_profiles_lock_default_execution_order():
    profiles = load_resource_profiles()

    assert tuple(profiles) == (
        "farmshare_l40_free",
        "flow_a100_weekend",
        "modal_burst",
        "cloudrift_fallback",
    )

    farmshare = profiles["farmshare_l40_free"]
    flow = profiles["flow_a100_weekend"]
    modal = profiles["modal_burst"]
    cloudrift = profiles["cloudrift_fallback"]

    assert farmshare.run_kind == "farmshare"
    assert farmshare.free is True
    assert farmshare.gpu_count == 4
    assert farmshare.max_concurrent_jobs == 4
    assert flow.run_kind == "flow"
    assert flow.gpu_model == "a100-80gb.sxm"
    assert flow.gpu_count == 4
    assert flow.max_bid_dollars_per_gpu_hour == pytest.approx(0.025)
    assert flow.initial_spend_cap_dollars == pytest.approx(10.0)
    assert modal.allowed_phases == ("phase3", "debug")
    assert cloudrift.fallback_only is True


def test_resource_profile_cost_validation_fails_closed():
    flow = get_resource_profile("flow_a100_weekend")
    farmshare = get_resource_profile("farmshare_l40_free")

    assert validate_profile_cost(flow, estimated_dollar_cost=9.99)["passed"] is True
    with pytest.raises(ValueError, match="exceeds profile initial_spend_cap_dollars"):
        validate_profile_cost(flow, estimated_dollar_cost=10.01)
    with pytest.raises(ValueError, match="FarmShare profile must remain zero-cost"):
        validate_profile_cost(farmshare, estimated_dollar_cost=0.01)
