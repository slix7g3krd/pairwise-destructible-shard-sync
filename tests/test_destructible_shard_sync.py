import pytest
import gameplay.destructible_shard_sync as mod

CASES = [
    ("predict_align", (1,)),
    ("shard_recycle", (1,)),
    ("tunneling", (1,)),
    ("seed_sync", (1,)),
    ("cap_match", (1,)),
    ("determinism", (1,)),
    ("authority", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("lifetime", (1,)),
    ("metrics", (1,)),
    ("snapshot", (1,)),
    ("interp", (1,)),
    ("budget", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
