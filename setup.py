from setuptools import find_packages, setup

setup(
    name="finetune_eval_harness",
    version="0.6.13",
    description="This project is a unified framework for evaluation of various language models on a large number of different evaluation tasks",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning, language models, evaluation",
    license="MIT",
    author="German Research Center for Artificial Intelligence (DFKI GmbH)",
    author_email="info@dfki.de",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={"console_scripts": ["finetune-eval-harness=finetune_eval_harness.main:main"]},
    python_requires=">=3.8.0",
    install_requires=[
        "pyarrow==6.0.1",
        "wandb",
        "jupyter",
        "ipywidgets>=8.0.2",
        "seqeval",
        "pandas==1.5.3",
        "coverage",
        "transformers",
        "accelerate",
        "evaluate",
        "datasets==2.8.0",
        "loralib",
        "flake8",
        "pytest",
        "pytest-cov",
    ],
    dependency_links=[
        "git+git://github.abc.com/abc/huggingface/peft.git#egg=huggingface/peft",
        #'git+https://github.com/huggingface/peft'
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
