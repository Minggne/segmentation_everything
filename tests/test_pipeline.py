def test_run_returns_instances(pipe, image):
    res = pipe.run(image, "cat, dog")
    assert [i.label for i in res.instances] == ["cat", "dog"]
    assert res.depth.shape == (image.height, image.width)


def test_depth_rank(pipe, image):
    res = pipe.run(image, "cat, dog")
    cat, dog = res.instances
    assert cat.depth_rank == 1 and dog.depth_rank == 2   # mèo nằm nửa trái = gần hơn
    assert cat.closeness > dog.closeness


def test_empty_detection(empty_pipe, image):
    res = empty_pipe.run(image, "unicorn")
    assert res.instances == []
    assert not res.union_mask().any()
    assert empty_pipe.render(res, "white_bg").size == res.image.size


def test_resize_max_side():
    from PIL import Image

    from scene_studio.utils.image import resize_max_side

    assert max(resize_max_side(Image.new("RGB", (4000, 2000)), 500).size) == 500
