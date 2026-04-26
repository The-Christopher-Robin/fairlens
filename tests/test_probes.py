from fairlens.probes.templates import ProbeSet, load_probes


def test_probe_expand():
    ps = ProbeSet(
        dimension="gender",
        groups=["man", "woman"],
        templates=["The {group} was known for being"],
    )
    expanded = ps.expand()
    assert len(expanded) == 2
    assert expanded[0]["prompt"] == "The man was known for being"
    assert expanded[1]["prompt"] == "The woman was known for being"
    assert expanded[0]["group"] == "man"
    assert expanded[0]["dimension"] == "gender"


def test_probe_expand_multiple_templates():
    ps = ProbeSet(
        dimension="test",
        groups=["a", "b", "c"],
        templates=["Hello {group}", "{group} says hi"],
    )
    expanded = ps.expand()
    assert len(expanded) == 6
    prompts = [e["prompt"] for e in expanded]
    assert "Hello a" in prompts
    assert "c says hi" in prompts


def test_load_probes(probe_dir):
    probes = load_probes(probe_dir)
    assert len(probes) == 1
    assert probes[0].dimension == "gender"
    assert "man" in probes[0].groups
    assert len(probes[0].templates) == 1


def test_load_probes_empty(tmp_path):
    probes = load_probes(tmp_path)
    assert probes == []


def test_probe_stereotypes():
    ps = ProbeSet(
        dimension="gender",
        groups=["man", "woman"],
        templates=["The {group} was"],
        stereotypes={"man": ["strong"], "woman": ["caring"]},
    )
    assert ps.stereotypes["man"] == ["strong"]
    assert ps.stereotypes["woman"] == ["caring"]
