"""兼容旧版 pip 的安装入口。"""

from pathlib import Path

from setuptools import find_packages, setup


root = Path(__file__).parent
setup(
    name="ttt-e2e-research",
    version="0.1.0",
    description="TTT-E2E 与混合注意力研究实现",
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["ttt", "ttt.*"]),
    python_requires=">=3.10",
    install_requires=["jax>=0.4.38,<0.5", "optax>=0.2.4,<0.2.5", "numpy>=1.26,<3"],
    extras_require={
        "dev": ["pytest>=8,<9", "ruff>=0.9", "httpx>=0.27,<1"],
        "tokenizer": ["transformers>=4.48,<5"],
        "checkpoint": ["orbax-checkpoint==0.10.3"],
        "cuda": ["jax[cuda12]==0.4.38"],
        "serve": ["fastapi>=0.115,<1", "uvicorn>=0.30,<1"],
        "analysis": ["scipy>=1.11,<2"],
    },
    entry_points={
        "console_scripts": [
            "ttt-train=ttt.train:main",
            "ttt-generate=ttt.generate:main",
            "ttt-prepare=ttt.data:main",
            "ttt-eval=ttt.eval_retrieval:main",
            "ttt-profile=ttt.profile:main",
            "ttt-serve=ttt.serve:main",
        ]
    },
    include_package_data=True,
)
