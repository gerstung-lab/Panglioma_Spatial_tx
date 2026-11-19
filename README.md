# The cellular neighbourhoods of adult gliomas and their clinical associations

Code related to manuscript "The cellular neighbourhoods of adult gliomas and their clinical associations" Mathioudaki et al 2026 Medrxiv 

Code was restructured and cleaned up with the help of Claude Opus 5.5 (Anthropic); all code and outputs were checked by the authors.

Data will be made available in Zenodo upon publication. 

dependencies:
  - python=3.10.19
  - pip=26.0.1
  
  - jupyterlab=4.5.4
  - ipykernel=7.2.0
  - numpy=2.2.6
  - scipy=1.14.1
  - pandas=2.3.3
  - h5py=3.15.1
  - numba=0.63.1
  - statsmodels=0.14.6
  - matplotlib-base=3.10.8
  - seaborn=0.13.2

  ###  Single-cell and spatial analysis
  - scanpy=1.11.5
  - umap-learn=0.5.11
  - pynndescent=0.5.13
  - anndata==0.11.4
  - mudata==0.3.3
  - scvi-tools==1.3.3
  - decoupler==2.1.4
  - igraph==1.0.0
  - leidenalg==0.11.0
  - shapely==2.1.2
    
  ### Machine learning
  - pytorch=2.4.0
  - torchvision=0.19.0
  - cpuonly=2.0
  - scikit-learn=1.7.2
