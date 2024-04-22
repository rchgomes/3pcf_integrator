# test-baryon-run

This is to validate the scale cuts of map3 to avoid the baryonic effect.

Strategy of validation:
Use the fitting formula for suppression factor measured in TNG-300, 
developed in [the bihalofit paper](https://arxiv.org/abs/1911.07886).

The fiducial baryonic effect used for validation of scale cut in 2pt 
analysis in DES-Y3 is from OWLS, which have stronger baryonic effect
than TNG-300. To obtain the OWLS-like suppression factor, we enlarge
the suppression factor of TNG-300 obtained from the fitting formula.
The rescaling factor would be similar for bispectrum as power spectrum.
See Fig. 13 of [this paper](https://arxiv.org/pdf/1910.03597.pdf), 
showing suppression factor on power and bispectrum, and shows similar 
amplitudes in the same baryonic effect.

More quantitatively, we multiply the following suppression factor to
the matter bispectrum
$$
R_b(k_1,k_2,k_3; f_b) = 1 + f_b [ R_b(k_1, k_2, k_3; {\rm TNG-300}) - 1]
$$
where the $R_b({\rm TNG-300})$ is the suppression factor of TNG-300.
With this parametrization, $f_b=0$ corresponds to no-baryon, and $f_b=1$
corresponds to TNG-300 baryonic effect. OWLS has stonger baryonic effect
and hence $f_b=1.5$ because the largest suppression factor in TNG-300 is
20% and that in OWLS is 30%.
