# config directories

## Contents
### [test-first-run](test-first-run)
This is very old run. We did not include even NLA. and pipeline no longer work with this.

### [test-second-run](test-second-run)
Includes NLA

### [test-baryon-run](test-baryon-run)
Stress test for baryon systematics. Simulated the baryon effect from OWLS-AGN. 
For map3, we simulated it by rescaling the baryon factor estimated for TNG-300 in bihalofit paper, 
where rescaling factor is estimated from 2pt.

### [test-wcdm-run](test-wcdm-run)
Assessment of constraining power on wCDM.

### [test-desy3c-run](test-desy3c-run)
Now we use DES-Y3 cosmological parameter as input cosmological parameters, 
so 'desy3c' indicates DES-Y3 cosmology. This is more reasonable test, because 
we would get contour around the DES Y3 2pt cosmic shear best-fit params.

### [test-desy3c-baryon-run](test-desy3c-baryon-run)
Same as [test-baryon-run](test-baryon-run) but using DES-Y3 cosmology for input.

### [test-desy3c-emu-run](test-desy3c-emu-run)
Same as [test-desy3c-run](test-desy3c-run), but using the emulator.
The result of this run is supposed to agree with that of [test-desy3c-run](test-desy3c-run).
We use this for validation of emulator and pipeline.

### [test-desy3c-wcdm-run](test-desy3c-wcdm-run)
Same as [test-desy3c-run](test-desy3c-run), but sampling wCDM.

### [emulator-train](emulator-train)
configs to generate the training dataset for emulator.

### [try-des-y3-shear](try-des-y3-shear)
Some configs used to setup baseline pipeline in cosmosis.

### [dev](dev)
configs for development.