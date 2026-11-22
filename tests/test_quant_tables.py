import json
from pathlib import Path

TABLE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "quant_tables"
    / "qwen3.5-4b_b1b770d9.json"
)
PROFILES = {"BF16", "Q8_0", "Q4_K_M", "Q3_K_M", "Q4_K_M_gate8"}


def test_quant_table_profiles_and_gate_diff():
    t = json.loads(TABLE.read_text())
    assert t["llama_cpp_commit"].startswith("b1b770d9")
    assert set(t["profiles"]) == PROFILES and set(t["argv"]) == PROFILES | {"F32"}
    # quantized profiles read the F32 GGUF: a BF16 source rounds away small fine-tuning updates
    for p in PROFILES - {"BF16"}:
        assert "<F32.gguf>" in t["argv"][p] and "<BF16.gguf>" not in t["argv"][p]
    names = {p: set(tab) for p, tab in t["profiles"].items()}
    assert all(n == names["BF16"] for n in names.values()) and len(names["BF16"]) == 378
    assert not any("mtp" in n for n in names["BF16"])
    q4, gate8 = t["profiles"]["Q4_K_M"], t["profiles"]["Q4_K_M_gate8"]
    changed = {n for n in q4 if q4[n]["type"] != gate8[n]["type"]}
    assert len(changed) == 24 and all(
        n.endswith("ssm_beta_alpha.weight") for n in changed
    )
    assert {gate8[n]["type"] for n in changed} == {"Q8_0"}
    assert "ssm_beta_alpha=q8_0" in t["argv"]["Q4_K_M_gate8"]
