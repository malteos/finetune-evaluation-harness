from setuptools import find_packages, setup

setup(
    name = "finetune_eval_harness",
    version="0.5.0.dev0",
    description="Finetune_Eval_Harness",
    #long_description=open("README.md", "r", encoding="utf-8").read(),
    keywords="deep learning",
    author="DFKI Berlin",
    author_email="akga01@dfki.de",
    #url="https://github.com/malteos/finetune-evaluation-harness",
    #package_dir={"":""},
    packages=[
            'hf_scripts', 
            'tasks',
            'templates',
            'tests',
    ],
    scripts = [
        'process_args.py',
        'main.py'
    ],
    python_requires=">=3.7.0",
    install_requires=[
        'pyarrow==6.0.1',
        'wandb',
        'jupyter',
        'ipywidgets>=8.0.2',
        'seqeval',
        'pandas==1.5.3',
        'coverage',
        'transformers', 
        'accelerate', 
        'evaluate', 
        'datasets==2.8.0',
        'loralib', 
        'flake8',
        'pytest',
        'pytest-cov',
    ],
    dependency_links = [
        'git+git://github.abc.com/abc/huggingface/peft.git#egg=huggingface/peft',
        #'git+https://github.com/huggingface/peft'
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

)