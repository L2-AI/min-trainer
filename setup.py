from setuptools import setup, find_packages

setup(
    name='mintrainer',
    version='0.1.0',
    description='Lightweight ModernBERT fine-tuning for classification',
    author='Andrew Bell',
    author_email='andrew@l2labs.ai',
    packages=find_packages(),  # Automatically finds `mintrainer`
    install_requires=[
        'scikit-learn',
        'pandas',
        'datasets',
        'transformers',
        'torch',
    ],       # Add dependencies here, e.g., ['numpy', 'torch']
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)