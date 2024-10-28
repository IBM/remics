# Remics
A framework for Redescription-based multi-omics analysis 

[![Remics Diagram][remics]](#)
<!-- <a>![badge-alt-text](images/remics_final.jpg)</a> -->

## 📂 Project File Structure

The project structure is as follows:

``` 
Remics/
│
├── 📂 src/  # main source code
    ├── 📂 tcga/
    │   ├── 📄 compute_cuna_tcga.py # Reads the multi-omics files, performs feature selection, and compute cumulants
    │   ├── 📄 compute_cures_tcga.py # Reads the cumulants, associated vectors, integrated multi-omics files and computes CuRES, while also performing the classification task  
    │   ├── 📄 cuna_networks_tcga.py # Reads the cumulants file and compute the networks, its communities, and ranking of nodes according to importance in the network
│   ├── 📂 remics/
│       ├── 📄  cumulants.py # computes cumulants 
│       ├── 📄  cures.py # computes CuRES
│       ├── 📄  cuna.py # computes CuNA
│       ├── 📄  cumulants.jl # Julia subroutine performing cumulants

