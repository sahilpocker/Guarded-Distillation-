import numpy as np
import pandas as pd
import pytest

from cortisol import gate


def aci_ready() -> bool:
    try:
        gate.aci_update(0.0, 0.0, 0.1, 0.1)
        return True
    except NotImplementedError:
        return False


needs_aci = pytest.mark.skipif(not aci_ready(), reason="cortisol.gate.aci_update not implemented")


@pytest.fixture
def tiny_pools():
    """A small synthetic stand-in for Banking77: 6 intents (2 of them 'launch') + out-of-scope."""
    from cortisol.data import Pools
    rng = np.random.default_rng(0)
    vocab = {
        "card_arrival": ["card", "arrive", "post", "delivery", "waiting"],
        "exchange_rate": ["rate", "exchange", "euro", "dollar", "convert"],
        "pending_transfer": ["transfer", "pending", "still", "bank", "sent"],
        "top_up_failed": ["top", "up", "failed", "money", "add"],
        "getting_virtual_card": ["virtual", "card", "online", "get", "digital"],
        "apple_pay_or_google_pay": ["apple", "google", "pay", "phone", "wallet"],
    }
    rows = []
    for lab, words in vocab.items():
        for i in range(400):
            w = rng.choice(words, size=4)
            rows.append((f"{' '.join(w)} please help {i}", lab))
    oos = [(f"what is the weather like in city {i} today", "out_of_scope") for i in range(300)]
    b = pd.DataFrame(rows, columns=["text", "label"])
    o = pd.DataFrame(oos, columns=["text", "label"])
    return Pools(banking=b, oos=o, intents=tuple(sorted(vocab)))
