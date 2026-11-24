from scene_studio.models.detector import GroundingDINODetector


def test_build_query():
    assert GroundingDINODetector.build_query("Person, Dog ,car") == "person. dog. car."
