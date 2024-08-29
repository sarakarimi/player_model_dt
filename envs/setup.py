from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent


# This call to setup() does all the work
setup(
    name="metalgridsolid",  # This is the name of your package
    version="0.1.0",  # Version number
    description="A description of your project",  # Short description of your package
    long_description_content_type="text/markdown",  # This indicates your long description is in Markdown
    url="https://github.com/miguelsvasco/metalgridsolid",  # URL of your project
    author="Miguel Vasco",  # Your name
    author_email="miguelsv@kth.se",  # Your email
    license="MIT",  # License for your project
    classifiers=[  # Classifiers help users find your project by category
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),  # Automatically find your packages
    include_package_data=True,  # Include package data specified in MANIFEST.in
    install_requires=[  # List your project's dependencies here
        # Example: "requests", "numpy",
    ],
    entry_points={  # If your package has executable scripts, indicate them here
        "console_scripts": [
            # Example: "mycli=metalgridsolid.cli:main",
        ]
    },
    python_requires=">=3.6",  # Minimum Python version required
)