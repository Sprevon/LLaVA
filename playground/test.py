#!/usr/bin/env python3
# test.py
import sys, subprocess, argparse
import importlib.metadata as importlib_metadata

from packaging.requirements import Requirement

# --------------- 依赖表 ---------------
CORE = [
    "torch>=2.3.0", "torchvision>=0.18.0",
    "transformers==4.37.2", "tokenizers==0.15.1", "sentencepiece==0.1.99", "shortuuid",
    "accelerate>=0.27.0", "peft", "bitsandbytes",
    "pydantic", "markdown2[all]", "numpy", "scikit-learn==1.2.2",
    "gradio==4.16.0", "gradio_client==0.8.1",
    "requests", "httpx==0.24.0", "uvicorn", "fastapi",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
]

EXTRA = {
    "train": ["deepspeed>=0.14.0", "ninja", "wandb"],
    "build": ["build", "twine"],
}

# --------------- 辅助函数 ---------------
def parse_whitelist(req: str):
    """把 'markdown2[all]' 这类 extras 去掉，只留包名"""
    return Requirement(req).name

def check_one(req_str: str) -> bool:
    """返回 True 表示满足要求"""
    req = Requirement(req_str)
    name = parse_whitelist(req_str)
    try:
        dist = importlib_metadata.distribution(name)
    except importlib_metadata.PackageNotFoundError:
        return False
    if req.specifier:
        return req.specifier.contains(dist.version)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", help="逗号分隔的 optional 组, 如 train,build")
    args = ap.parse_args()

    testing = {"core": CORE}
    if args.extra:
        for grp in args.extra.split(","):
            grp = grp.strip()
            if grp in EXTRA:
                testing[grp] = EXTRA[grp]
            else:
                print(f"⚠️  unknown extra group: {grp}")

    missing = []
    for grp, deps in testing.items():
        print(f"========== checking {grp} ==========")
        for dep in deps:
            ok = check_one(dep)
            flag = "✔" if ok else "✘"
            print(f"{flag}  {dep}")
            if not ok:
                missing.append(dep)
        print()

    if not missing:
        print("🎉  All required packages are satisfied.")
        sys.exit(0)

    print("❌  Missing / version-mismatch packages:")
    for m in missing:
        print("  -", m)
    print("\n💡  You can install them via:\n"
          "  pip install " + " ".join(f'"{p}"' for p in missing))
    sys.exit(1)

if __name__ == "__main__":
    main()