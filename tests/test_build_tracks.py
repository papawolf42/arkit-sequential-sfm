from tools.build_tracks import build_tracks


def test_chain_and_conflicting_component_are_separated():
    edges = [((1, 0), (2, 0)), ((2, 0), (3, 0)),
             ((4, 0), (5, 0)), ((5, 0), (4, 1))]
    tracks, conflicts = build_tracks(edges)
    assert tracks == [[(1, 0), (2, 0), (3, 0)]]
    assert conflicts == [[(4, 0), (4, 1), (5, 0)]]


def test_repeated_edges_do_not_duplicate_observations():
    tracks, conflicts = build_tracks([((1, 2), (2, 3)), ((2, 3), (1, 2))])
    assert tracks == [[(1, 2), (2, 3)]]
    assert conflicts == []
