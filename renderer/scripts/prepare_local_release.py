"""Build Windows packages from isolated dependency source snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from windows_env import compiler_environment

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PROFILE = {
    "python": "3.10.17",
    "platform": "win_amd64",
    "cuda": "12.8",
    "gpu_architecture": "sm_120",
    "torch": "2.11.0+cu128",
    "numpy": "2.2.6",
    "cupy-cuda12x": "14.0.1",
    "drjit": "1.3.1",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def inventory(root: Path) -> list[dict]:
    return [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True)


def run(command: list[str], cwd: Path, env: dict | None = None) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    logs = DIST / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((str(cwd) + repr(command)).encode()).hexdigest()[:12]
    with (logs / f"{key}.log").open("a", encoding="utf-8") as log:
        log.write(f"cwd={cwd}\ncommand={command!r}\n")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def snapshot(source: Path, name: str, extra: list[str]) -> None:
    target = DIST / "sources" / name
    target.mkdir(parents=True, exist_ok=False)
    names = git(source, "ls-files", "--recurse-submodules", "-z").split("\0")
    for relative in sorted(set(names + extra) - {""}):
        src = source / relative
        if src.is_file():
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            raise FileNotFoundError(f"Tracked source missing: {src}")
    patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD"])
    (DIST / "sources" / f"{name}.patch").write_bytes(patch)
    files = inventory(target)
    write_json(
        DIST / "sources" / f"{name}.manifest.json",
        {
            "source": str(source),
            "head": git(source, "rev-parse", "HEAD").strip(),
            "remotes": git(source, "remote", "-v").splitlines(),
            "status": git(source, "status", "--short").splitlines(),
            "submodules": git(source, "submodule", "status", "--recursive").splitlines(),
            "patch_sha256": digest(DIST / "sources" / f"{name}.patch"),
            "extra_files": extra,
            "files": files,
            "licenses": [
                f
                for f in files
                if any(x in f["path"].lower() for x in ("license", "copying", "notice"))
            ],
            "scope": "Native dependency build sources",
        },
    )
    with zipfile.ZipFile(DIST / "sources" / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            archive.write(target / item["path"], item["path"])


def verify_tree(root: Path, manifest: dict) -> None:
    for item in manifest["files"]:
        path = safe_member(root, item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]
        ):
            raise ValueError(f"Missing or corrupt artifact: {path}")


def safe_member(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Archive path escapes destination: {name}")
    return path


def build(args: argparse.Namespace) -> None:
    for name in ("python-optix", "tiny-cuda-nn"):
        if not (DIST / "sources" / f"{name}.manifest.json").is_file():
            raise RuntimeError("Complete the snapshot command before building")
    work = DIST / "build"
    work.mkdir(parents=True, exist_ok=True)
    env = compiler_environment(args.cuda, args.vcvars)
    nvcc_version = subprocess.check_output(["nvcc", "--version"], env=env, text=True)
    if "release 12.8," not in nvcc_version:
        raise ValueError("This wheel profile requires CUDA Toolkit 12.8")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    env.update(UV_LINK_MODE="copy", MAX_JOBS=str(args.jobs))
    python = work / ".venv/Scripts/python.exe"
    if not python.exists():
        run(["uv", "venv", "--python", PROFILE["python"], str(python.parents[1])], work, env)
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "torch==2.11.0+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        work,
        env,
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "build==1.2.2.post1",
            "setuptools==81.0.0",
            "wheel==0.45.1",
            "Cython==0.29.37",
            "numpy==2.2.6",
            "packaging==25.0",
            "ninja==1.13.0",
            "hatchling==1.27.0",
        ],
        work,
        env,
    )
    sdk = args.optix_sdk.resolve()
    if "#define OPTIX_VERSION 70600" not in (sdk / "include/optix.h").read_text():
        raise ValueError("This build profile requires OptiX 7.6.0")
    env.update(OPTIX_PATH=str(sdk), OPTIX_EMBED_HEADERS="1")
    wheels = DIST / "wheels"
    wheels.mkdir(exist_ok=True)
    notices = DIST / "licenses/optix-sdk"
    notices.mkdir(parents=True, exist_ok=True)
    for name in ("OptiX_EndUserLicense.pdf", "OptiX_ThirdParty_Licenses.txt"):
        shutil.copy2(sdk / "doc" / name, notices / name)
    cuda_notices = DIST / "licenses/cuda"
    cuda_notices.mkdir(parents=True, exist_ok=True)
    for name in ("EULA.txt", "LICENSE", "version.json"):
        shutil.copy2(Path(env["CUDA_PATH"]) / name, cuda_notices / name)
    for name, binding in (("python-optix", ""), ("tiny-cuda-nn", "bindings/torch")):
        source = DIST / "sources" / name
        manifest = json.loads((DIST / "sources" / f"{name}.manifest.json").read_text())
        verify_tree(source, manifest)
        for item in manifest["licenses"]:
            notice = DIST / "licenses" / name / item["path"]
            notice.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item["path"], notice)
        dest = work / name
        if not dest.exists():
            shutil.copytree(source, dest)
        verify_tree(dest, manifest)
        run(
            [str(python), "-m", "build", "--wheel", "--no-isolation", "--outdir", str(wheels)],
            dest / binding,
            env,
        )
    write_json(
        DIST / "dependency-wheels.manifest.json",
        {
            "profile": PROFILE,
            "files": [
                f for f in inventory(wheels) if not f["path"].startswith("adashell_renderer-")
            ],
            "sdk_source": str(sdk),
            "sdk_headers": inventory(sdk / "include"),
            "sdk_notices": inventory(notices),
            "licenses": inventory(DIST / "licenses"),
            "compiler": shutil.which("cl.exe", path=env["PATH"]),
            "nvcc": subprocess.check_output(["nvcc", "--version"], env=env, text=True),
            "sources": [
                json.loads(p.read_text())
                for p in sorted((DIST / "sources").glob("*.manifest.json"))
            ],
            "scope": "Private local wheels; embeds OptiX and CUDA headers; not approved for redistribution",
        },
    )


def scene_bundle(config_path: Path) -> None:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    files = [config_path]
    references = [config["checkpoint"], config["camera_path"], config.get("provenance_config")]
    references.extend(config["mesh_assets"].values())
    for value in references:
        if value:
            path = safe_member(config_path.parent, value)
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    entries = [
        {
            "path": p.relative_to(config_path.parent).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in sorted(set(files))
    ]
    manifest = {
        "scene": config_path.stem,
        "config": config_path.name,
        "files": entries,
        "renderer_head": git(ROOT, "rev-parse", "HEAD").strip(),
        "license": "Scene asset license not specified",
        "provenance_config": config.get("provenance_config"),
        "scope": "Local preparation only",
    }
    output = DIST / "scenes" / f"{config_path.stem}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for item in entries:
            archive.write(config_path.parent / item["path"], item["path"])
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    write_json(output.with_suffix(".manifest.json"), {**manifest, "archive_sha256": digest(output)})
    print(f"Created {output}", flush=True)


def extract(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Use a fresh destination: {destination}")
    sidecar = archive_path.with_suffix(".manifest.json")
    expected = json.loads(sidecar.read_text())
    if digest(archive_path) != expected["archive_sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest != {k: v for k, v in expected.items() if k != "archive_sha256"}:
            raise ValueError("Archive and sidecar manifests disagree")
        names = archive.namelist()
        if len(names) != len({n.casefold() for n in names}) or set(names) != {
            f["path"] for f in manifest["files"]
        } | {"manifest.json"}:
            raise ValueError("Unexpected or duplicate archive members")
        for name in names:
            safe_member(destination, name)
        destination.mkdir(parents=True)
        for name in names:
            path = safe_member(destination, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    verify_tree(destination, manifest)
    print(f"Extracted and verified {destination}")


def renderer_wheel() -> None:
    work = DIST / "build"
    work.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="renderer-", dir=work))
    shutil.copytree(
        ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, target / name)
    for name in ("LICENSE", "LICENSE.Apache-2.0"):
        shutil.copy2(ROOT / name, target / name)
    run(
        [
            str(DIST / "build/.venv/Scripts/python.exe"),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(DIST / "wheels"),
        ],
        target,
    )
    write_json(
        DIST / "renderer-source.manifest.json",
        {
            "head": git(ROOT, "rev-parse", "HEAD").strip(),
            "snapshot": str(target),
            "files": inventory(target),
            "license": "MIT AND Apache-2.0; see LICENSE and LICENSE.Apache-2.0",
        },
    )


def validate(destination: Path | None = None) -> None:
    target = (destination or DIST / "validation").resolve()
    if not target.is_relative_to(DIST.resolve()) or target == DIST.resolve():
        raise ValueError("Validation destinations must be beneath ignored dist/")
    target.mkdir(parents=True, exist_ok=True)
    python = target / ".venv/Scripts/python.exe"
    if python.exists():
        raise FileExistsError("Validation requires a fresh venv")
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    env.update(
        UV_LINK_MODE="copy",
        PYTHONNOUSERSITE="1",
        TORCH_EXTENSIONS_DIR=str(target / "cache/torch"),
        CUPY_CACHE_DIR=str(target / "cache/cupy"),
    )
    run(["uv", "venv", "--python", PROFILE["python"], str(python.parents[1])], target, env)
    requirements = DIST / "runtime-requirements.txt"
    run(
        [
            "uv",
            "export",
            "--frozen",
            "--extra",
            "viewer-gl",
            "--no-dev",
            "--no-emit-project",
            "--output-file",
            str(requirements),
        ],
        ROOT,
        env,
    )
    # Sync the lock's explicit Torch index, but never install the source project.
    env["UV_PROJECT_ENVIRONMENT"] = str(python.parents[1])
    run(
        [
            "uv",
            "sync",
            "--frozen",
            "--extra",
            "viewer-gl",
            "--no-dev",
            "--no-install-project",
            "--no-build",
        ],
        ROOT,
        env,
    )
    renderers = list((DIST / "wheels").glob("adashell_renderer-*.whl"))
    if len(renderers) != 1:
        raise ValueError("Keep exactly one renderer wheel in dist/wheels for validation")
    renderer = renderers[0]
    run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(renderer)], target, env)
    run(["uv", "pip", "check", "--python", str(python)], target, env)
    code = """import importlib, importlib.metadata as m, json, pathlib, sys
root = pathlib.Path(sys.prefix).resolve()
report = {}
for name in ('adashell_renderer', 'torch', 'cupy', 'optix', 'tinycudann', 'drjit', 'glfw', 'OpenGL', 'imgui'):
    mod = importlib.import_module(name)
    path = pathlib.Path(mod.__file__).resolve()
    assert path.is_relative_to(root), (name, str(path))
    report[name] = str(path)
native = pathlib.Path(importlib.import_module('tinycudann').modules._C.__file__).resolve()
assert native.is_relative_to(root), native
report['tinycudann_native'] = str(native)
cu = pathlib.Path(importlib.import_module('adashell_renderer').__file__).with_name('narrow_band.cu')
assert cu.is_file(), cu
assert pathlib.Path(importlib.import_module('optix').__file__).with_name('include').joinpath('optix.h').is_file()
report['versions'] = {d.metadata['Name']: d.version for d in m.distributions()}
print(json.dumps(report, indent=2))
"""
    result = subprocess.check_output(
        [str(python), "-I", "-c", code], cwd=target, env=env, text=True
    )
    (target / "imports.json").write_text(result, encoding="utf-8")
    write_json(
        DIST / "release.manifest.json",
        {
            "profile": PROFILE,
            "wheels": inventory(DIST / "wheels"),
            "licenses": inventory(DIST / "licenses"),
            "dependency_provenance_sha256": digest(DIST / "dependency-wheels.manifest.json"),
            "renderer_provenance_sha256": digest(DIST / "renderer-source.manifest.json"),
            "source_archives": [
                {"path": p.relative_to(DIST).as_posix(), "sha256": digest(p)}
                for p in sorted((DIST / "sources").glob("*.zip"))
            ],
            "scene_manifests": [
                {"path": p.relative_to(DIST).as_posix(), "sha256": digest(p)}
                for p in sorted((DIST / "scenes").glob("*.manifest.json"))
            ],
            "lock_sha256": digest(ROOT / "uv.lock"),
            "validation_report": str(target / "imports.json"),
            "validation_report_sha256": digest(target / "imports.json"),
            "validation": "Imports and dependency checks only; no rendering",
        },
    )
    print(result)


def selftest() -> None:
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DIST) as temp:
        root = Path(temp)
        payload = b"release integrity test"
        manifest = {
            "files": [
                {
                    "path": "assets/test.bin",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
        }
        archive = root / "test.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("assets/test.bin", payload)
            zipped.writestr("manifest.json", json.dumps(manifest))
        write_json(
            archive.with_suffix(".manifest.json"), {**manifest, "archive_sha256": digest(archive)}
        )
        extract(archive, root / "extracted")
        for name in ("../escape", "/absolute", "C:/escape", "a\\b"):
            try:
                safe_member(root, name)
            except ValueError:
                pass
            else:
                raise AssertionError(name)
        (root / "extracted/assets/test.bin").write_bytes(b"tampered")
        try:
            verify_tree(root / "extracted", manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Corruption was not detected")
        with archive.open("ab") as stream:
            stream.write(b"tampered")
        try:
            extract(archive, root / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("Archive corruption was not detected")
    print("PASS: extract roundtrip, traversal rejection, file and archive corruption detection")


def compile_interop(args: argparse.Namespace) -> None:
    target = (args.destination or DIST / "validation").resolve()
    if not target.is_relative_to(DIST.resolve()):
        raise ValueError("Interop build must stay beneath dist/")
    python = target / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise FileNotFoundError("Run validate first")
    env = compiler_environment(args.cuda, args.vcvars)
    env.update(
        TORCH_EXTENSIONS_DIR=str(target / "cache/torch"),
        CUPY_CACHE_DIR=str(target / "cache/cupy"),
        MAX_JOBS=str(args.jobs),
    )
    env["PATH"] = str(python.parent) + os.pathsep + env["PATH"]
    code = """import json, os, pathlib, sys
from adashell_renderer.cuda_gl_interop import load_cuda_gl_extension
import adashell_renderer
assert pathlib.Path(adashell_renderer.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
extension = load_cuda_gl_extension()
path = pathlib.Path(extension.__file__).resolve()
assert path.is_relative_to(pathlib.Path(os.environ['TORCH_EXTENSIONS_DIR']).resolve())
pathlib.Path('interop-build.json').write_text(json.dumps({'extension': str(path), 'rendered': False}, indent=2))
"""
    run([str(python), "-I", "-c", code], target, env)


def gpu_smoke(destination: Path | None) -> None:
    target = (destination or DIST / "validation").resolve()
    if not target.is_relative_to(DIST.resolve()):
        raise ValueError("Smoke destination must stay beneath dist/")
    python = target / ".venv/Scripts/python.exe"
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    env.update(CUPY_CACHE_DIR=str(target / "cache/cupy"), PYTHONNOUSERSITE="1")
    code = """import importlib.metadata as metadata, json, pathlib, sys
import torch, cupy as cp, drjit as dr
import adashell_renderer.viewer_gl as viewer
from adashell_renderer.config_io import load_renderer_config
prefix = pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(viewer.__file__).resolve().is_relative_to(prefix)
assert float((torch.ones(1, device='cuda') + 2).item()) == 3
assert float((cp.ones(1) + 2).get()[0]) == 3
x = dr.cuda.Float(1) + 2
dr.eval(x)
assert float(x[0]) == 3
configs = []
for name in ('khady', 'horse'):
    path = pathlib.Path('scenes') / name / (name + '.json')
    cfg = load_renderer_config(path.resolve())
    for asset in (cfg.assets.checkpoint, cfg.assets.camera_path,
                  cfg.mesh_assets.dilation_mesh, cfg.mesh_assets.erosion_mesh):
        assert asset.is_file(), asset
    configs.append(str(path.resolve()))
origins = {}
for name in ('adashell-renderer', 'python-optix', 'tinycudann'):
    origin = json.loads(metadata.distribution(name).read_text('direct_url.json'))
    assert 'archive_info' in origin and origin['url'].endswith('.whl'), origin
    origins[name] = origin
report = {'scope': 'Scalar CUDA checks only; no model load, rendering, timing, or GUI',
          'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
          'viewer': viewer.__file__, 'configs': configs, 'origins': origins}
pathlib.Path('gpu-smoke.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
"""
    run([str(python), "-I", "-c", code], target, env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--python-optix", type=Path, required=True)
    snap.add_argument("--tiny-cuda-nn", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--optix-sdk", type=Path, required=True)
    b.add_argument("--cuda", type=Path)
    b.add_argument("--vcvars", type=Path)
    b.add_argument("--jobs", type=int, default=4)
    assets = sub.add_parser("scenes")
    assets.add_argument("configs", type=Path, nargs="+")
    ext = sub.add_parser("extract")
    ext.add_argument("archive", type=Path)
    ext.add_argument("destination", type=Path)
    sub.add_parser("renderer-wheel")
    val = sub.add_parser("validate")
    val.add_argument("--destination", type=Path)
    interop = sub.add_parser("compile-interop")
    interop.add_argument("--destination", type=Path)
    interop.add_argument("--cuda", type=Path)
    interop.add_argument("--vcvars", type=Path)
    interop.add_argument("--jobs", type=int, default=4)
    smoke = sub.add_parser("gpu-smoke")
    smoke.add_argument("--destination", type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(args.python_optix.resolve(), "python-optix", [])
        snapshot(args.tiny_cuda_nn.resolve(), "tiny-cuda-nn", ["bindings/torch/pyproject.toml"])
    elif args.action == "build":
        build(args)
    elif args.action == "scenes":
        for config in args.configs:
            scene_bundle(config)
    elif args.action == "extract":
        extract(args.archive.resolve(), args.destination.resolve())
    elif args.action == "renderer-wheel":
        renderer_wheel()
    elif args.action == "validate":
        validate(args.destination)
    elif args.action == "selftest":
        selftest()
    elif args.action == "compile-interop":
        compile_interop(args)
    elif args.action == "gpu-smoke":
        gpu_smoke(args.destination)


if __name__ == "__main__":
    main()
