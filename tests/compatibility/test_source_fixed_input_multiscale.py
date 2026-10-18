"""Regression for resizing a fixed-input Swin model during prediction."""

from Preflight_checker.preflight import check_source


SOURCE = '''
IMG_RES = 256

class PetModel:
    def __init__(self):
        self.backbone = timm.create_model(
            "swin_small_patch4_window7_224", img_size=IMG_RES
        )

def predict(model, imgs, scales=(0.9, 1.0)):
    for scale in scales:
        new_h = int(round(imgs.size(2) * scale))
        new_w = int(round(imgs.size(3) * scale))
        xi = torch.nn.functional.interpolate(imgs, size=(new_h, new_w))
        model(xi)

predict(model, imgs)
'''


def test_rejects_nonunit_prediction_scale_for_fixed_input_swin():
    report = check_source(SOURCE, shared_memory_bytes=1 << 30)

    assert report["admitted"] is False
    assert "fixed_input_multiscale_resize" in report["issues"]
    assert any(
        diagnostic["code"] == "fixed_input_multiscale_resize"
        and "swin_small_patch4_window7_224" in diagnostic["message"]
        and "256" in diagnostic["message"]
        for diagnostic in report["diagnostics"]
    )


def test_accepts_unit_scale_for_fixed_input_swin():
    source = SOURCE.replace("scales=(0.9, 1.0)", "scales=(1.0,)")

    assert check_source(source, shared_memory_bytes=1 << 30)["admitted"] is True


def test_accepts_nonunit_scale_for_non_fixed_backbone():
    source = SOURCE.replace("swin_small_patch4_window7_224", "convnext_tiny")

    assert check_source(source, shared_memory_bytes=1 << 30)["admitted"] is True


def test_does_not_reject_unused_fixed_input_fallback():
    source = SOURCE.replace(
        'self.backbone = timm.create_model(\n            "swin_small_patch4_window7_224", img_size=IMG_RES\n        )',
        'try:\n'
        '            self.backbone = timm.create_model(BACKBONE, img_size=IMG_RES)\n'
        '        except Exception:\n'
        '            self.backbone = timm.create_model(\n'
        '                "swin_small_patch4_window7_224", img_size=IMG_RES\n'
        '            )',
    ).replace("IMG_RES = 256", 'IMG_RES = 256\nBACKBONE = "convnext_tiny"')

    assert check_source(source, shared_memory_bytes=1 << 30)["admitted"] is True


def test_rejects_swin_multiscale_when_dynamic_img_size_is_ignored():
    source = '''
BACKBONE = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"
IMG_RES = 256

class PetModel:
    def __init__(self):
        self.backbone = timm.create_model(
            BACKBONE, img_size=IMG_RES, dynamic_img_size=True
        )

def predict(model, imgs, metas, scales=(0.9, 1.0)):
    for s in scales:
        def _round32(value):
            return max(32, int(round(value / 32.0)) * 32)
        new_h = _round32(imgs.size(2) * s)
        new_w = _round32(imgs.size(3) * s)
        xi = torch.nn.functional.interpolate(imgs, size=(new_h, new_w))
        model(xi, metas)

predict(model, imgs, metas)
'''

    report = check_source(source, shared_memory_bytes=1 << 30)

    assert "fixed_input_multiscale_resize" in report["issues"]
