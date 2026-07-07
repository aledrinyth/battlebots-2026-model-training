from agario_visualiser.visualiser_submission import CameraFrame, VisualiserApp


def test_circle_in_camera_becomes_visible_on_edge_entry() -> None:
    camera = CameraFrame(
        left=10.0,
        top=10.0,
        right=30.0,
        bottom=30.0,
        scale_x=1.0,
        scale_y=1.0,
    )

    assert VisualiserApp._is_circle_in_camera(None, camera, 30.5, 20.0, 1.0)
    assert not VisualiserApp._is_circle_in_camera(None, camera, 31.1, 20.0, 1.0)


def test_circle_in_camera_respects_diagonal_corner_intersection() -> None:
    camera = CameraFrame(
        left=10.0,
        top=10.0,
        right=30.0,
        bottom=30.0,
        scale_x=1.0,
        scale_y=1.0,
    )

    assert VisualiserApp._is_circle_in_camera(None, camera, 30.6, 30.6, 1.0)
    assert not VisualiserApp._is_circle_in_camera(None, camera, 30.9, 30.9, 1.0)
