"""Checks for the gene pool.  python test_genes.py"""
import random
import tempfile
from pathlib import Path

import genes as G

# keep tests off the real pool file
G.POOL_FILE = Path(tempfile.mkdtemp()) / "test_pool.json"


def _state(n=10):
    return {"votes": 0,
            "genes": [G._gene(f"style {i}, clause a{i}, clause b{i}, clause c{i}",
                              120, 128, i + 1) for i in range(n)]}


def test_crossover_joins_parent_clauses():
    rng = random.Random(0)
    genes = _state(2)["genes"]
    child = G.crossover(genes[0], genes[1], rng, 99)
    clauses = set(child["text"].split(", "))
    assert clauses <= set(genes[0]["text"].split(", ")) | set(genes[1]["text"].split(", "))
    assert clauses & set(genes[0]["text"].split(", "))   # has a head from A
    assert clauses & set(genes[1]["text"].split(", "))   # and a tail from B
    assert child["bpm_lo"] == (genes[0]["bpm_lo"] + genes[1]["bpm_lo"]) // 2


def test_mutate_swaps_a_clause_for_a_genre():
    rng = random.Random(1)
    genres = set(G.GENRES_FILE.read_text().splitlines())
    text = "deep house, warm analog bassline, lush pads, shuffling hi-hats"
    out = G.mutate(text, rng)
    before, after = text.split(", "), out.split(", ")
    changed = [(x, y) for x, y in zip(before, after) if x != y]
    assert len(changed) == 1, changed
    # the new clause must have gained a genre name the old one didn't have
    assert changed[0][0] != changed[0][1]
    assert any(g in changed[0][1] and g not in changed[0][0]
               for g in genres), changed


def test_pick_trial_is_random_and_respects_filter():
    rng = random.Random(2)
    gene_list = _state(6)["genes"]
    picks = {G.pick_trial(gene_list, rng)["id"] for _ in range(100)}
    assert picks == {g["id"] for g in gene_list}       # every gene reachable
    assert G.pick_trial(gene_list, rng, "style 3")["id"] == 4
    assert G.pick_trial(gene_list, rng, "no match")["id"] in picks


def test_vote_scores_without_evolving_until_5():
    state = _state(10)
    for i in range(4):
        G.vote(state, 1, True)
    assert len(state["genes"]) == 10                   # no evolve before 5 votes
    assert state["genes"][0]["up"] == 4
    G.vote(state, 1, False)                            # 5th vote
    assert state["votes"] == 5


def test_fifth_vote_drops_bottom_5_and_refills():
    rng = random.Random(4)
    state = _state(10)
    for i, g in enumerate(state["genes"]):
        g["up"] = 10 - g["id"]                         # ids 6..10 score lowest
    for _ in range(5):
        G.vote(state, 1, True, rng)                    # vote top gene, up
    ids = {g["id"] for g in state["genes"]}
    assert len(ids) == 10
    assert ids == {1, 2, 3, 4, 5} | set(range(11, 16))  # bottom 5 gone, children in
    for g in state["genes"]:
        if g["id"] > 10:
            assert g["plays"] == 0 and g["text"]       # fresh children, nonempty


def test_small_pool_never_evolved_to_nothing():
    rng = random.Random(5)
    state = _state(5)                                  # exactly EVICT_N genes
    for _ in range(10):
        G.vote(state, 1, True, rng)
    assert len(state["genes"]) == 5                    # no death spiral


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
