"""Deploy the API to a Hugging Face Docker Space.

Uploads the Docker build context (the files the Dockerfile copies, as
tracked by git) plus this folder's README.md, whose front matter tells the
Space to build the Dockerfile. The Space rebuilds on every upload.

CI runs this on pushes to main with:

    HF_TOKEN          a Hugging Face token with write access
    HF_SPACE          the Space, as "user/space-name"; created if missing
    FRONTEND_ORIGIN   optional: the web app's origin, e.g.
                      https://robustness-explorer.vercel.app, which becomes
                      the Space's VISION_CORS_ORIGINS

To inspect the build context without uploading anything:

    python deploy/huggingface/deploy.py --stage-only /tmp/space
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_CONTEXT = [
    "Dockerfile",
    "requirements.txt",
    "requirements-cpu.txt",
    "src",
    "runs/multiseed/experiment1",
]


def tracked_files(paths: list[str]) -> list[str]:
    """Files under ``paths`` that git tracks, so a local run uploads exactly
    what CI would, not untracked checkpoints or caches."""
    output = subprocess.run(
        ["git", "ls-files", "-z", "--", *paths],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return [path for path in output.decode().split("\0") if path]


def stage(target: Path) -> list[str]:
    """Copies the build context and the Space README into ``target``."""
    files = tracked_files(BUILD_CONTEXT)
    for path in files:
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / path, destination)
    shutil.copy2(Path(__file__).with_name("README.md"), target / "README.md")
    return files


def deploy() -> None:
    from huggingface_hub import HfApi

    space = os.environ["HF_SPACE"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    origin = os.environ.get("FRONTEND_ORIGIN")
    if origin:
        api.add_space_variable(space, "VISION_CORS_ORIGINS", origin)

    with tempfile.TemporaryDirectory() as directory:
        files = stage(Path(directory))
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=directory,
            commit_message=f"Deploy {os.environ.get('GITHUB_SHA', 'local build')[:7]}",
            # Removes remote files that are no longer part of the context.
            delete_patterns="*",
        )
    print(f"Uploaded {len(files) + 1} files to https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", type=Path, metavar="DIR", help="copy the context to DIR and stop")
    args = parser.parse_args()

    if args.stage_only:
        args.stage_only.mkdir(parents=True, exist_ok=True)
        print("\n".join(stage(args.stage_only)))
    else:
        deploy()
