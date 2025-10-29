from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="eagle-vql",
    version="1.0.0",
    author="yfcao",
    author_email="yfcao@mail.dlut.edu.cn",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyfedu-dlut/EAGLE_VQL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eagle-train=tools.train_vq2d:main",
            "eagle-eval=tools.eval_vq2d:main",
            "eagle-inference=tools.inference:main",
        ],
    },
)