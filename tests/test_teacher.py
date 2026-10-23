import pandas as pd

from guarded_distillation.teacher import SimulatedTeacher, build_system_prompt, parse_label


def test_parse_label():
    labels = ["card_arrival", "exchange_rate", "out_of_scope"]
    assert parse_label("card_arrival", labels) == "card_arrival"
    assert parse_label("  `Exchange_Rate`.\n", labels) == "exchange_rate"
    assert parse_label("card_arival", labels) == "card_arrival"
    assert parse_label("I think it's about pizza", labels) == "unparseable"


def test_system_prompt_lists_every_label(tiny_pools):
    sp = build_system_prompt(tiny_pools.intents, None)
    for lab in tiny_pools.intents:
        assert f"- {lab}" in sp
    assert "- out_of_scope" in sp


def test_simulated_teacher_hits_its_accuracy_and_is_deterministic(tiny_pools):
    df = pd.concat([tiny_pools.banking, tiny_pools.oos])
    T = SimulatedTeacher(df, acc=0.9, seed=0)
    labels = [T.label(r.text, r.label) for r in df.itertuples()]
    acc = (pd.Series(labels).values == df.label.values).mean()
    assert 0.86 <= acc <= 0.94
    assert labels == [T.label(r.text, r.label) for r in df.itertuples()]
