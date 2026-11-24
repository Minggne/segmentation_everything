import pytest

from scene_studio.postprocess import EFFECTS, crop_cutouts, depth_colormap, overlay


@pytest.mark.parametrize("name", list(EFFECTS))
def test_all_effects(pipe, image, name):
    res = pipe.run(image, "cat, dog")
    out = pipe.render(res, name)
    assert out.size == res.image.size
    if name == "cutout":
        assert out.mode == "RGBA"


def test_unknown_effect(pipe, image):
    with pytest.raises(ValueError):
        pipe.render(pipe.run(image, "cat"), "nope")


def test_visualizers(pipe, image):
    res = pipe.run(image, "cat, dog")
    assert overlay(res).size == res.image.size
    assert depth_colormap(res.depth).size == res.image.size
    cuts = crop_cutouts(res)
    assert len(cuts) == 2 and cuts[0][0].mode == "RGBA"
