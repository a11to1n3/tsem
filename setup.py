from setuptools import setup, find_packages

setup(
  name = 'tsem',
  packages = find_packages(exclude=['examples']),
  version = '1.0.0',
  license='MIT',
  description = 'TSEM',
  long_description_content_type = 'text/markdown',
  author = 'Anh-Duy Pham',
  author_email = 'duyanhpham@outlook.com',
  url = 'https://github.com/a11to1n3/tsem',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'XAI',
    'multivariate timeseries'
  ],
  install_requires=[
    'torch>=1.10',
    'scipy',
    'pandas',
    'matplotlib',
    'scikit-learn'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==1.12.1',
    'scipy',
    'pandas',
    'matplotlib',
    'scikit-learn'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)