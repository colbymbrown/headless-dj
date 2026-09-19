"""Gene pool: the prompt population that evolves across daily mixes.

Each gene is one prompt (its variable middle text), a bpm range, and
play/vote counts. Seeded from dj.STYLES on first load; every gene starts at
score 0. One mix = one gene trial: dj.py picks a gene at random, renders the
mix, your vote on the webpage (web.py) scores it (+1 up, -1 down; no vote,
no change). Every EVOLVE_EVERY votes the bottom EVICT_N genes by net score
(up - down) are dropped and replaced with crossover children (comma-clause
splices) of the survivors, each 50% mutated by swapping one word for a
random genre from genres.txt.

  python genes.py --status      # show the pool
  python genes.py --evolve      # force an evolution round now
"""
import json
import random
from pathlib import Path

POOL_FILE = Path("gene_pool.json")
GENRES_FILE = Path("genres.txt")
EVICT_N = 5       # genes dropped + refilled each evolution round
EVOLVE_EVERY = 5  # votes between evolution rounds


def load():
    """Pool state {"votes": n, "genes": [...]}; seeds from dj.STYLES first run."""
    if not POOL_FILE.exists():
        from dj import STYLES  # lazy: dj imports this module
        gene_list = []
        for style, ((lo, hi), variants) in STYLES.items():
            for v in variants:
                gene_list.append(_gene(f"{style}, {v}", lo, hi, next_id(gene_list)))
        state = {"votes": 0, "genes": gene_list}
        save(state)
        print(f"seeded {POOL_FILE} with {len(gene_list)} genes from dj.STYLES")
    state = json.loads(POOL_FILE.read_text())
    return state


def save(state):
    POOL_FILE.write_text(json.dumps(state, indent=2))


def next_id(gene_list):
    return max((g["id"] for g in gene_list), default=0) + 1


def _gene(text, lo, hi, gid):
    return {"id": gid, "text": text, "bpm_lo": lo, "bpm_hi": hi,
            "plays": 0, "up": 0, "down": 0}


def score(g):
    return g["up"] - g["down"]


def pick_trial(gene_list, rng, contains=None):
    """Random gene. `contains` (the --style flag) narrows candidates;
    falls back to the whole pool if nothing matches."""
    cands = [g for g in gene_list if contains and contains.lower() in g["text"].lower()]
    return rng.choice(cands or gene_list)


def crossover(a, b, rng, gid):
    """Child text: head clauses of A + tail clauses of B (comma-spliced).
    BPM range: midpoint of the parents'."""
    ca = a["text"].split(", ")
    cb = b["text"].split(", ")
    ka, kb = rng.randint(0, len(ca)), rng.randint(0, len(cb))
    text = ", ".join(dict.fromkeys(ca[:ka] + cb[kb:]))  # dedup, keep order
    lo = (a["bpm_lo"] + b["bpm_lo"]) // 2
    hi = (a["bpm_hi"] + b["bpm_hi"]) // 2
    return _gene(text, lo, hi, gid)


def mutate(text, rng):
    """Replace one word with a random genre name from genres.txt."""
    genres = GENRES_FILE.read_text().splitlines()
    clauses = text.split(", ")
    c = rng.randrange(len(clauses))
    words = clauses[c].split()
    if not words:
        return text
    words[rng.randrange(len(words))] = rng.choice(genres)
    clauses[c] = " ".join(words)
    return ", ".join(clauses)


def evolve(gene_list, rng=None):
    """Drop EVICT_N worst by score, refill with crossover children of the
    survivors (50% mutated). Returns the new gene list; caller saves."""
    rng = rng or random.Random()
    gene_list.sort(key=score, reverse=True)
    survivors = gene_list[:-EVICT_N]
    parents = survivors[:max(2, len(survivors) // 2)]
    nid = next_id(gene_list)
    children = []
    while len(children) < EVICT_N:
        child = crossover(rng.choice(parents), rng.choice(parents), rng, nid)
        nid += 1
        if rng.random() < 0.5:
            child["text"] = mutate(child["text"], rng)
        if child["text"]:
            children.append(child)
    print(f"evolved: dropped {[g['text'][:40] for g in gene_list[-EVICT_N:]]}, "
          f"added {[c['text'][:40] for c in children]}")
    return survivors + children


def vote(state, gene_id, up, rng=None):
    """Score a trial (+1/-1). Every EVOLVE_EVERY votes, evolve the pool.
    # ponytail: unsynchronized file writes; fine for one localhost voter."""
    g = next(g for g in state["genes"] if g["id"] == gene_id)
    g["up" if up else "down"] += 1
    state["votes"] += 1
    if state["votes"] % EVOLVE_EVERY == 0 and len(state["genes"]) > EVICT_N:
        state["genes"] = evolve(state["genes"], rng)
    save(state)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--evolve", action="store_true")
    args = ap.parse_args()
    state = load()
    if args.evolve:
        state["genes"] = evolve(state["genes"])
        save(state)
    for g in sorted(state["genes"], key=score, reverse=True):
        print(f"  #{g['id']:3d}  {g['plays']:3d} plays  {g['up']:+d}/{g['down']:+d}  "
              f"score {score(g):+d}  {g['text']}")
    print(f"\n{len(state['genes'])} genes, {state['votes']} votes since seed; "
          f"evolves every {EVOLVE_EVERY} votes (next at "
          f"{(state['votes'] // EVOLVE_EVERY + 1) * EVOLVE_EVERY})")
