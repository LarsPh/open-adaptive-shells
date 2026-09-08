from types import SimpleNamespace

import pytest

from adashell_renderer.types import DrRenderBuffer
from adashell_renderer.viewer_gl import CudaGlPresenter
from adashell_renderer.viewer_support import ViewerRenderFrame


class FakeGl:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("GL_"):
            return name

        def call(*args):
            self.calls.append((name, args))
            return 12 if name == "glGenTextures" else 34

        return call


class FakeInterop:
    def __init__(self):
        self.calls = []

    def register_pbo(self, pbo):
        self.calls.append(("register", pbo))
        return 56

    def unregister_pbo(self, handle):
        self.calls.append(("unregister", handle))

    def __getattr__(self, name):
        assert name in {"copy_rgb_ptrs_to_pbo", "copy_scalar_ptr_to_pbo", "copy_vector_ptrs_to_pbo"}

        def copy(*args):
            self.calls.append((name, args))
            return (0.1, 0.2, 0.3)

        return copy


@pytest.mark.parametrize("mode,expected", [
    ("RGB", "copy_rgb_ptrs_to_pbo"),
    ("sample_count", "copy_scalar_ptr_to_pbo"),
    ("pred_gradient_norm", "copy_scalar_ptr_to_pbo"),
    ("pred_normal", "copy_vector_ptrs_to_pbo"),
])
def test_direct_presenter_dispatch_and_range_reupload(monkeypatch, mode, expected):
    gl, ext = FakeGl(), FakeInterop()
    presenter = CudaGlPresenter(gl, ext)
    monkeypatch.setattr(CudaGlPresenter, "_dr_arrays_to_cupy", lambda self, *xs: tuple(SimpleNamespace(data=SimpleNamespace(ptr=x)) for x in xs))
    buffer = DrRenderBuffer((100, 200, 300), 4, 2, 2, diagnostics={"sample_count": 400, "pred_gradient_norm": 500, "pred_normal": (600, 700, 800)})
    result = ViewerRenderFrame(SimpleNamespace(width=2, height=2), {}, {}, 0.0, buffer)
    presenter.upload(result, mode, (0.0, 1.0))
    presenter.upload(result, mode, (0.0, 2.0))
    copies = [args for name, args in ext.calls if name == expected]
    assert len(copies) == 2
    assert sum(name == "register" for name, _ in ext.calls) == 1
    if "scalar" in expected:
        assert copies[0][4:6] == (0.0, 1.0)
        assert copies[1][4:6] == (0.0, 2.0)
    assert result.details["display_direct_pbo"] == 1.0
    assert result.details["display_torch_materialize_ms"] == 0.0
    assert sum(name == "glTexSubImage2D" for name, _ in gl.calls) == 2
    presenter.close()
    presenter.close()
    assert sum(name == "unregister" for name, _ in ext.calls) == 1


def test_presenter_rejects_partial_frame_before_allocating():
    gl, ext = FakeGl(), FakeInterop()
    presenter = CudaGlPresenter(gl, ext)
    result = ViewerRenderFrame(SimpleNamespace(width=2, height=2), {}, {}, 0.0, DrRenderBuffer((1, 2, 3), 3, 2, 2))
    with pytest.raises(RuntimeError, match="full-frame"):
        presenter.upload(result, "RGB", (0.0, 1.0))
    assert gl.calls == []
    assert ext.calls == []


def test_resize_unregisters_cuda_before_deleting_gl_buffer():
    events = []
    gl, ext = FakeGl(), FakeInterop()
    gl.glDeleteBuffers = lambda *_: events.append("delete")
    ext.unregister_pbo = lambda _: events.append("unregister")
    presenter = CudaGlPresenter(gl, ext)
    presenter.ensure_size(2, 2)
    presenter.ensure_size(4, 3)
    assert events == ["unregister", "delete"]
    assert (presenter.width, presenter.height) == (4, 3)
