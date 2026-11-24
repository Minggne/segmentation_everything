import json

from scene_studio.export import to_coco, write_coco


def test_coco_structure(pipe, image):
    coco = to_coco(pipe.run(image, "cat, dog"), "a.png")
    assert {c["name"] for c in coco["categories"]} == {"cat", "dog"}
    assert len(coco["annotations"]) == 2
    assert coco["annotations"][0]["segmentation"]
    assert coco["images"][0]["width"] == 100


def test_write_coco(pipe, image, tmp_path):
    p = write_coco(pipe.run(image, "cat"), tmp_path / "x.json")
    assert json.loads(p.read_text())["annotations"]
