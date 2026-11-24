from scene_studio.postprocess import stats_rows


def test_area_pct(pipe, image):
    res = pipe.run(image, "cat")
    cat = res.instances[0]
    assert abs(cat.area_pct - (40 * 40) / (100 * 80) * 100) < 1e-6


def test_stats_rows_shape(pipe, image):
    rows = stats_rows(pipe.run(image, "cat, dog").instances)
    assert len(rows) == 2 and len(rows[0]) == 5
