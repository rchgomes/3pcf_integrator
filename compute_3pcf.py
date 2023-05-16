import numpy as np
from bihalofit import bihalofit
from funcs import f_h3, f_psi3, f_psi1, f_psi2, transform_gamma
import matplotlib.pyplot as plt
from halo_model import halo_model_bispectrum
from tree_level import tree_level_bispectrum
from gil_marin_bispectrum import gil_marin
import time

def compute_3pcf(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=100000, rmax=50, model='bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        model = halo_model_bispectrum(cosmo_parameters, kless, zless, "Halo_bispectrum_with_baryons_and_NO_DM_PROFILE_apr26.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)

    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re.mean + 1j*test_integration_0_im.mean

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("Gamma0_imag_" + output_suffix, np.imag(result_array_0))

    bb_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_1_re = model.gamma1(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_1_im = model.gamma1(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        bb_vals[i] = test_integration_1_re.mean + 1j*test_integration_1_im.mean

    result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)
    np.save("Gamma1_real_" + output_suffix, np.real(result_array_1))
    np.save("Gamma1_imag_" + output_suffix, np.imag(result_array_1))

    cc_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_2_re = model.gamma2(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_2_im = model.gamma2(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        cc_vals[i] = test_integration_2_re.mean + 1j*test_integration_2_im.mean

    result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)
    np.save("Gamma2_real_" + output_suffix, np.real(result_array_2))
    np.save("Gamma2_imag_" + output_suffix, np.imag(result_array_2))

    dd_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_3_re = model.gamma3(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_3_im = model.gamma3(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        dd_vals[i] = test_integration_3_re.mean + 1j*test_integration_3_im.mean

    result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)
    np.save("Gamma3_real_" + output_suffix, np.real(result_array_3))
    np.save("Gamma3_imag_" + output_suffix, np.imag(result_array_3))

    np.save("Gamma_ttt_" + output_suffix, 1/4*(np.real(result_array_0+result_array_1+result_array_2+result_array_3)))

def plot_3pcf(files, labels, xvals, gamma_func, no_log = False):

    for i in range(len(files)):
        a = np.load(files[i])
        plt.plot(xvals, a * (-1) * np.pi ** 2,
             label=labels[i])
    plt.title('Halo model' + gamma_func + '(isosceles)')
    plt.xlabel("phi")
    if no_log == False:
        plt.yscale("log")
    plt.ylabel(gamma_func)
    plt.legend(fontsize=7)
    plt.grid()
    plt.show()

    first = np.load(files[0])
    second = np.load(files[2])
    plt.title('Ratio between 3pcfs')
    plt.plot(xvals, first/second)
    plt.xlabel("phi")
    if no_log == False:
        plt.yscale("log")
    plt.ylabel(gamma_func)
    plt.legend(fontsize=7)
    plt.grid()
    plt.show()

def compute_3pcf_ro(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=100000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)
    print("time to kernel:", time.time()-timelens)

    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re + 1j*test_integration_0_im

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("Gamma0_imag_" + output_suffix, np.imag(result_array_0))

    bb_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_1_re = model.gamma1_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_1_im = model.gamma1_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        bb_vals[i] = test_integration_1_re + 1j*test_integration_1_im

    result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)
    np.save("Gamma1_real_" + output_suffix, np.real(result_array_1))
    np.save("Gamma1_imag_" + output_suffix, np.imag(result_array_1))

    cc_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_2_re = model.gamma2_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_2_im = model.gamma2_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        cc_vals[i] = test_integration_2_re + 1j*test_integration_2_im

    result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)
    np.save("Gamma2_real_" + output_suffix, np.real(result_array_2))
    np.save("Gamma2_imag_" + output_suffix, np.imag(result_array_2))

    dd_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_3_re = model.gamma3_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_3_im = model.gamma3_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        dd_vals[i] = test_integration_3_re + 1j*test_integration_3_im

    result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)
    np.save("Gamma3_real_" + output_suffix, np.real(result_array_3))
    np.save("Gamma3_imag_" + output_suffix, np.imag(result_array_3))

    np.save("Gamma_ttt_" + output_suffix, 1/4*(np.real(result_array_0+result_array_1+result_array_2+result_array_3)))

def compute_3pcf_ro_gamma0only(cosmo_parameters, dndz_file, d2_vals, u_vals, v_vals, output_suffix,
                 kmin=10**(-3), kmax=50, n_kbins=10000, chimin=1, chimax=4000,
                 n_chibins=10, niter=5, neval=100000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    my_z_new = np.linspace(0,3,1000)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    model.compute_lensing_kernel(chimin, chimax, 10000, dndz_file)
    print("time to kernel:", time.time()-timelens)

    aa_vals = np.ndarray(shape=len(d2_vals), dtype=complex)
    for i in range(len(d2_vals)):
        test_integration_0_re = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=False)
        test_integration_0_im = model.gamma0_loop(limits, d2_vals[i], u_vals[i], v_vals[i], chimin,
                                             chimax, n_chibins, niter, neval, baryons, imag=True)
        aa_vals[i] = test_integration_0_re + 1j*test_integration_0_im

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)
    np.save("training_set/Gamma0_real_" + output_suffix, np.real(result_array_0))
    np.save("training_set/Gamma0_imag_" + output_suffix, np.imag(result_array_0))

def compute_3pcf_for_emulator(cosmo_parameters, d2_vals, u_vals, v_vals, z,
                 kmin=10**(-3), kmax=50, n_kbins=10000, niter=5, neval=100000, rmax=50, model = 'bihalofit', baryons = False):

    my_k = np.logspace(np.log10(kmin), np.log10(kmax), num=n_kbins)  # h/Mpc^-1
    #my_z_new = np.linspace(0,3,1000)
    my_z_new = np.linspace(0, 3, 100)
    if model == 'bihalofit':
        model = bihalofit(cosmo_parameters, my_k, my_z_new)
    if model == 'halo model':
        kless = np.logspace(-2, np.log10(20), 20)
        zless = np.linspace(0, 2.7, 10)
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless, "new_halo_model_bispectrum_with_baryons_apr14.npy")
        model = halo_model_bispectrum(cosmo_parameters, kless, zless,
                                      "new_halo_model_bispectrum_with_baryons_apr21_Mc038.npy")
        #model = halo_model_bispectrum(cosmo_parameters, kless, zless,
        #                              "new_halo_model_bispectrum_with_baryons_apr14_dmprofnorelax.npy")
    if model == 'tree level':
        model = tree_level_bispectrum(cosmo_parameters, my_k, my_z_new)
    if model == 'gil marin':
        model = gil_marin(cosmo_parameters, my_k, my_z_new)
    limits = [[0, 2*np.pi],[0, np.pi/2],[0,rmax]]
    timelens = time.time()
    print("time to kernel:", time.time()-timelens)

    chi = model.r_from_z_func(z)

    constant = 27 * (100 / 299792) ** 6 * model.omegam ** 3 / 8
    output = np.ndarray(shape=(8))
    test_integration_0_re = model.gamma0_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
    test_integration_0_im = model.gamma0_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
    print(test_integration_0_re, test_integration_0_im)
    aa_vals = test_integration_0_re.mean + 1j*test_integration_0_im.mean

    result_array_0 = transform_gamma(aa_vals, 0, d2_vals, u_vals, v_vals)

    output[0] = np.real(result_array_0)
    output[1] = np.imag(result_array_0)

    test_integration_1_re = model.gamma1_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
    test_integration_1_im = model.gamma1_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
    bb_vals = test_integration_1_re.mean + 1j*test_integration_1_im.mean

    result_array_1 = transform_gamma(bb_vals, 1, d2_vals, u_vals, v_vals)

    output[2] = np.real(result_array_1)
    output[3] = np.imag(result_array_1)

    test_integration_2_re = model.gamma2_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
    test_integration_2_im = model.gamma2_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
    cc_vals = test_integration_2_re.mean + 1j*test_integration_2_im.mean

    result_array_2 = transform_gamma(cc_vals, 2, d2_vals, u_vals, v_vals)

    output[4] = np.real(result_array_2)
    output[5] = np.imag(result_array_2)

    test_integration_3_re = model.gamma3_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=False)
    test_integration_3_im = model.gamma3_ro(limits, d2_vals, u_vals, v_vals, chi, niter, neval, baryons, imag=True)
    dd_vals = test_integration_3_re.mean + 1j*test_integration_3_im.mean

    result_array_3 = transform_gamma(dd_vals, 3, d2_vals, u_vals, v_vals)

    output[6] = np.real(result_array_3)
    output[7] = np.imag(result_array_3)

    return(constant*output)

