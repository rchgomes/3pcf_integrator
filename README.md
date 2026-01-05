# 3PCF integrator

# Contents
- python: cosmoSIS modules and auxilliary python codes.
- config: cosmoSIS configuration files
- data: source redshift files, covariances, etc
- scripts: other python scripts

# Installation
`fastnc` is available at [github](https://github.com/git-sunao/fastnc). For installation, use
```
pip install fastnc
```
**Note**: fastnc creates cache of the mode-coupling function to avoid recomputation. This cache is for now created under package dir. The typical file size is <4MB if you use the multipoles only up to L=100 and M=100.

# Tips
Run the following command before you use the cosmosis modules. 
This sources the environmental paths necessary for cosmosis configs.
```
source setup-3pcf-path.sh
```

