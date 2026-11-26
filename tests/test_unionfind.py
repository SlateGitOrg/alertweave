from alertweave.cluster import UnionFind


def test_new_items_are_their_own_sets():
    uf = UnionFind()
    items = [uf.add() for _ in range(4)]
    assert [uf.find(i) for i in items] == items


def test_union_is_transitive_and_reports_whether_it_merged():
    uf = UnionFind()
    a, b, c, d = (uf.add() for _ in range(4))
    assert uf.union(a, b) is not None
    assert uf.union(b, c) is not None
    assert uf.union(a, c) is None
    assert uf.find(a) == uf.find(b) == uf.find(c) != uf.find(d)


def test_find_compresses_paths():
    uf = UnionFind()
    items = [uf.add() for _ in range(8)]
    for left, right in zip(items, items[1:], strict=False):
        uf.union(left, right)
    root = uf.find(items[0])
    for item in items:
        uf.find(item)
    assert all(uf.parent(item) == root for item in items)


def test_union_by_rank_keeps_trees_shallow():
    uf = UnionFind()
    count = 1024
    items = [uf.add() for _ in range(count)]
    step = 1
    while step < count:
        for start in range(0, count, 2 * step):
            uf.union(items[start], items[start + step])
        step *= 2

    def depth(item: int) -> int:
        steps = 0
        while uf.parent(item) != item:
            item = uf.parent(item)
            steps += 1
        return steps

    assert max(depth(i) for i in items) <= 10
    assert len({uf.find(i) for i in items}) == 1
