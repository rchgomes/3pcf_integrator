# 3PCF integrator

# Contents
- python: python codes including cosmosis modules.
- data: source redshift, signal, etc
- config: config of cosmosis
- scripts: other python scripts

# Installation
`fastnc` is available at [github](https://github.com/git-sunao/fastnc). For installation, use
```
pip install fastnc
```
**Note**: fastnc create cache of mode-coupling function to avoid recomputation. This cache is for now created under package dir, but I eventually want to allow user to specify the dir of cache. The typical file size is <4MB if you use the multipoles only up to L=100 and M=100.

# Tips
Run the following command before you use the cosmosis modules. 
This sources the environmental pathes necessary for cosmosis configs.
```
source setup-3pcf-path.sh
```

