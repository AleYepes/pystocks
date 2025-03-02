# pystocks
Multi-factor modeling of IBKR's ETPs (exchange traded products)

# Requirements
Use Python 3.12.8

## Install nbstripout to Remove Unnecessary Metadata
```
pip install nbstripout
nbstripout --install
```


## 2. Use nbdime for Better Diffs and Merging
```
pip install nbdime
nbdime config-git --enable
```

To manually compare two notebook versions:
```
nbdiff notebook_1.ipynb notebook_2.ipynb
```

To resolve conflicts interactively:

```
nbmerge notebook.ipynb
```