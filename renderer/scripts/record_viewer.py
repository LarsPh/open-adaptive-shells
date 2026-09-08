"""Record the live GL viewer framebuffer, never the surrounding desktop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def apply_camera_tour(state, frame, target, elapsed: float, duration: float) -> None:
    """Replay camera controls from a fixed pose for a repeatable walkthrough."""
    progress = min(max(elapsed / duration, 0.0), 1.0)
    state.camera.reset(frame, target=target)
    state.camera.zoom(3.0)
    state.rotate_mode = "Orbit"
    state.display_mode = "RGB"
    if progress < 0.3:
        phase = progress / 0.3
        state.camera.orbit(150 * math.sin(2 * math.pi * phase), 12 * math.sin(2 * math.pi * phase))
    elif progress < 0.4:
        phase = (progress - 0.3) * 10
        state.camera.zoom(3 * math.sin(math.pi * phase))
    elif progress < 0.5:
        phase = (progress - 0.4) * 10
        state.camera.pan(120 * math.sin(2 * math.pi * phase), 30 * math.sin(2 * math.pi * phase))
    elif progress < 0.6:
        phase = (progress - 0.5) * 10
        state.rotate_mode = "Look"
        state.camera.look(18 * math.sin(2 * math.pi * phase), 6 * math.sin(2 * math.pi * phase))
    else:
        phase = (progress - 0.6) * 10
        modes = ("pred_normal", "sample_count", "pred_gradient_norm", "depth")
        state.display_mode = modes[min(int(phase), len(modes) - 1)]
        state.camera.orbit(120 * math.sin(2 * math.pi * phase), 0)
    state.continuous = True
    state.dirty = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--delay", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-tour", action="store_true", help="Replay Orbit, dolly, pan and Look controls while recording.")
    args, viewer_args = parser.parse_known_args()
    if viewer_args[:1] == ["--"]:
        viewer_args = viewer_args[1:]
    if args.seconds <= 0 or args.delay < 0 or args.fps <= 0:
        parser.error("Duration and FPS must be positive; delay must be nonnegative.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg must be on PATH.")
    if args.output.exists():
        parser.error(f"Refusing to overwrite recording: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from adashell_renderer import viewer_gl

    submit_frame = viewer_gl.GlViewerSession._submit_frame
    run_frame = viewer_gl.GlViewerSession._run_frame
    tour_frame = None
    tour_target = None
    encoder = None
    start = None
    size = None
    previous_pixels = None
    written_frames = 0
    captured_frames = 0
    finished = False

    def animate_frame(session):
        nonlocal tour_frame, tour_target
        if tour_frame is None:
            tour_frame = session.state.camera.to_frame(1.0)
            tour_target = session.state.camera.target.clone()
        if start is not None:
            apply_camera_tour(session.state, tour_frame, tour_target, time.perf_counter() - start, args.seconds)
            session.ui_state.sample_max = min(32, session.sample_limit)
            session.ui_state.gradient_norm_max = 0.25
            session.ui_state.depth_min = session.state.camera.radius * 0.5
            session.ui_state.depth_max = session.state.camera.radius * 1.3
        run_frame(session)

    def capture_frame(session, frame_start):
        nonlocal encoder, start, size, previous_pixels, written_frames, captured_frames, finished
        submit_frame(session, frame_start)
        now = time.perf_counter()
        if start is None:
            start = now + args.delay
        elapsed = now - start
        if elapsed < 0:
            return
        if elapsed >= args.seconds:
            finished = True
            session.glfw.set_window_should_close(session.window, True)
            return
        due_frames = int(elapsed * args.fps) + 1
        if due_frames <= written_frames:
            return
        current_size = session.glfw.get_framebuffer_size(session.window)
        if size is None:
            size = current_size
            encoder = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgba", "-video_size", f"{size[0]}x{size[1]}",
                 "-framerate", str(args.fps), "-i", "pipe:0", "-vf", "vflip",
                 "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)],
                stdin=subprocess.PIPE,
            )
            print(f"Recording {size[0]}x{size[1]} at {args.fps} FPS for {args.seconds:g}s", flush=True)
        if current_size != size:
            raise RuntimeError("Keep the viewer window size fixed during recording.")
        gl = session.gl
        gl.glReadBuffer(gl.GL_FRONT)
        pixels = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        pixels = np.asarray(pixels).tobytes() if not isinstance(pixels, bytes) else pixels
        if len(pixels) != size[0] * size[1] * 4:
            raise RuntimeError("Unexpected framebuffer readback size.")
        # Preserve elapsed time: slow frames hold the previous image, never speed up the recording.
        while written_frames < due_frames - 1:
            encoder.stdin.write(previous_pixels if previous_pixels is not None else pixels)
            written_frames += 1
        encoder.stdin.write(pixels)
        written_frames += 1
        captured_frames += 1
        previous_pixels = pixels

    viewer_gl.GlViewerSession._submit_frame = capture_frame
    if args.camera_tour:
        viewer_gl.GlViewerSession._run_frame = animate_frame
    sys.argv = ["adashell-viewer-gl", *viewer_args]
    try:
        viewer_gl.main()
    finally:
        viewer_gl.GlViewerSession._submit_frame = submit_frame
        viewer_gl.GlViewerSession._run_frame = run_frame
        if encoder is not None:
            if finished:
                while written_frames < round(args.seconds * args.fps):
                    encoder.stdin.write(previous_pixels)
                    written_frames += 1
            encoder.stdin.close()
            returncode = encoder.wait(timeout=60)
            if returncode != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {returncode}.")
            args.output.with_suffix(".json").write_text(json.dumps({
                "source": "live GL front framebuffer after swap; no desktop capture",
                "camera_motion": "controller replay" if args.camera_tour else "interactive",
                "viewer_args": viewer_args, "width": size[0], "height": size[1],
                "fps": args.fps, "written_frames": written_frames,
                "captured_frames": captured_frames, "duration_seconds": written_frames / args.fps,
                "completed_duration": finished,
                "measurement_note": "CPU readback and video encoding add overhead; not a performance benchmark.",
            }, indent=2), encoding="utf-8")
            print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
