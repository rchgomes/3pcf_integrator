import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class
from vegas import Integrator
import functools
from inspect import signature

#Verify dependance of k_nl and n_eff by cosmology
class bispectrum:

    def __init__(self, cosmo_params, k, z):

        self.cosmo_params = cosmo_params
        self.my_cosmo = Class()
        self.my_cosmo.set(self.cosmo_params)
        self.my_cosmo.compute()

        self.k = k
        self.z = z

        self.omegam = self.cosmo_params['Omega_b'] + self.cosmo_params['Omega_cdm']
        h = self.cosmo_params['h']
        knl = self.my_cosmo.nonlinear_scale(z, len(z))

        self.knl_interp = interp1d(z, knl, bounds_error = False, fill_value = knl[-1])

        self.Pk_linear = np.ndarray(shape=(len(k),len(z)))
        self.Pk_nonlinear = np.ndarray(shape=(len(k), len(z)))
        for i in range(len(self.Pk_linear)):
            for j in range(len(self.Pk_linear[0])):
                self.Pk_linear[i][j] = self.my_cosmo.pk_lin(k[i]*h,z[j])
                self.Pk_nonlinear[i][j] = self.my_cosmo.pk(k[i]*h, z[j])

        self.Pk_linear *= h ** 3
        self.Pk_nonlinear *= h ** 3

        newpkinterp = []
        for kii in range(len(self.k)):
            newpkinterp.append([])
            for zi in range(len(self.z)):
                newpkinterp[kii].append(self.Pk_linear[kii][zi])
        print("will do")
        t1 = time.time()
        self.PL = RegularGridInterpolator((self.k, self.z), newpkinterp, bounds_error = False, fill_value = 0)
        t2 = time.time()
        print("time first", t2-t1)

        self.r_from_z, self.dzdr_of_z = self.my_cosmo.z_of_r(z)

        self.r_from_z *= h
        self.dzdr_of_z *= h

        self.dzdr_from_r_func = interp1d(self.r_from_z, self.dzdr_of_z,  bounds_error = False, fill_value = 0)
        self.z_from_r_func = interp1d(self.r_from_z, z, bounds_error = False, fill_value = 0)

    def compute_kernel(self, k1, k2, k3):
        dot = (-k3 ** 2 + k1 ** 2 + k2 ** 2) / 2
        f2 = 5 / 7 + 2 * dot ** 2 / (7 * k1 ** 2 * k2 ** 2) - dot * (1 / k1 ** 2 + 1 / k2 ** 2) / 2  # from Laila
        return (f2)

    def b_of_l(self, l1, l2, l3, chi):

        z_actual = self.z_from_r_func(chi)
        b_of_l_vals = self.matter_bispectrum(z_actual, l1/chi, l2/chi, l3/chi)
        #print("b of l:", b_of_l_vals)

        return (b_of_l_vals)

    def compute_lensing_kernel(self, chimin, chimax, npoints, dndz):

        print("into lensing kernel")
        chivals = np.linspace(chimin, chimax, npoints)
        zvals = self.z_from_r_func(chivals)
        nz = interp1d(dndz[:, 0], dndz[:, 1], bounds_error = False, fill_value = 0)

        def_z = np.trapz(dndz[:,1], dndz[:,0])
        print(def_z)

        try:
            nvals = nz(zvals)*self.dzdr_from_r_func(chivals)
            #nvals = nz(zvals)
        except ValueError:
            raise ValueError(f"There was an interpolation problem. Your requested redshifts span from z={zvals[0]} to z={zvals[-1]}, "
                             f"while your redshift distribution function spans from z={dndz[0][0]} to z={dndz[-1][0]}")

        print(chimin, chimax)
        def_int = np.trapz(nvals, chivals)
        nvals /= def_int
        plt.plot(chivals, nvals)
        plt.show()
        lensing_kernel = np.zeros_like(chivals)
        for chii in range(len(chivals)-1):
            integrand = nvals[chii:]*(chivals[chii:]-chivals[chii]*np.ones_like(chivals[chii:]))/chivals[chii:]
            lensing_kernel[chii] = np.trapz(integrand, chivals[chii:])

        print("did lensing kernel")
        self.lensing_kernel  = interp1d(chivals, lensing_kernel)
        plt.plot(chivals, lensing_kernel)
        plt.yscale("log")
        plt.show()
        print("interpolated lensing kernel")

    def compute_kappa_bispectrum(self, l1, l2, l3, chimax, npoints):

        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(1, chimax, npoints)
        if type(l1) == np.float64:
            integrand = (self.lensing_kernel(chivals_simple) * (
                        1 + self.z_from_r_func(chivals_simple))) ** 3 / chivals_simple * self.b_of_l(
                l1, l2, l3, chivals_simple)
            integral = np.trapz(integrand, chivals_simple)
        else:
            integral = np.ndarray(shape=(len(l1)))
            for l1i in range(len(l1)):
                integrand = (self.lensing_kernel(chivals_simple) * (1 + self.z_from_r_func(chivals_simple))) ** 3 / chivals_simple * self.b_of_l(
                    l1[l1i], l2[l1i], l3[l1i], chivals_simple)
                integral[l1i] = np.trapz(integrand, chivals_simple)

        return(constant*integral)

    def compute_kappa_bispectrum_equilateral(self, l0, chimax, npoints):

        print("into kappa bispectrum")
        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(350, chimax, npoints)
        chivals = np.ndarray(shape=(len(l0), len(chivals_simple)))
        for l0i in range(len(l0)):
            chivals[l0i] = chivals_simple

        print("will compute integrand")
        aa0 = time.time()
        integrand = (self.lensing_kernel(chivals)*(1+self.z_from_r_func(chivals)))**3/chivals*self.b_of_l_equilateral(l0,chivals_simple)

        aa1 = time.time()
        print("computed integrand")
        integral = np.trapz(integrand, chivals)
        aa2 = time.time()
        print("computed integal. Times:", aa1-aa0, aa2-aa1)

        return(constant*integral)

    def create_interpolated_kappa_bispectrum(self, lvec):

        before = time.time()
        print(np.shape(self.kappa_bispectrum))
        print(np.shape(lvec))
        self.interpolated_kappa_bispectrum = RegularGridInterpolator((lvec, lvec, lvec), self.kappa_bispectrum, bounds_error = False, fill_value = 0)
        after = time.time()
        print("time to interpolate kappa bispectrum:", after - before)
        x = signature(self.interpolated_kappa_bispectrum)
        print(x)

    def gamma0_integrand_adapt(self, r, u, v, phi, psi, R):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * (np.exp(1j*beta_bar_times2)) * R ** 3 * jv(6, R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))

        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.sin((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1 = np.exp(1j*(phi2-phi3-6*alpha1))

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.sin((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2 = np.exp(1j*(phi3 - phi1 - 6*alpha2))

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.sin((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3 = np.exp(1j*(phi1 - phi2 - 6*alpha3))

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        first_term = E1 / A1_prime ** 4 * self.interpolated_kappa_bispectrum((l1_1,l2_1, l3_1))
        second_term = E2 / A2_prime ** 4 * self.interpolated_kappa_bispectrum((l1_2, l2_2, l3_2))
        third_term = E3 / A3_prime ** 4 * self.interpolated_kappa_bispectrum((l1_3, l2_3, l3_3))

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

    def gamma0_integrand(self, r, u, v, y):

        measure0 = time.time()
        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[0]
        psi = y[1]
        R = y[2]

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * (np.exp(1j*beta_bar_times2)) * R ** 3 * jv(6, R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))

        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.sin((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1 = np.exp(1j*(phi2-phi3-6*alpha1))

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.sin((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2 = np.exp(1j*(phi3 - phi1 - 6*alpha2))

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.sin((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3 = np.exp(1j*(phi1 - phi2 - 6*alpha3))

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        first_term = E1 / A1_prime ** 4 * self.compute_kappa_bispectrum(l1_1,l2_1, l3_1, 4000, 500)
        second_term = E2 / A2_prime ** 4 * self.compute_kappa_bispectrum(l1_2, l2_2, l3_2, 4000, 500)
        third_term = E3 / A3_prime ** 4 * self.compute_kappa_bispectrum(l1_3, l2_3, l3_3, 4000, 500)

        complete_integrand = outside_term*(first_term+second_term+third_term)
        #if np.abs(complete_integrand[6]) > 10**(-8):
        print("l values, integral", l1_1, l2_1, l3_1)
        #print("one loop of integrand:", time.time()-measure0)

        return(complete_integrand)

    def gamma0(self, integ_limits, r, u, v):

        timenow = time.time()
        int_obj = Integrator(integ_limits)
        train = int_obj(functools.partial(self.gamma0_integrand, r, u, v), nitn=10, neval=1000)
        result = int_obj(functools.partial(self.gamma0_integrand, r, u, v), nitn=10, neval=1000)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma1_integrand(self, r, u, v, y):

        measure0 = time.time()
        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        phi = y[0]
        psi = y[1]
        R = y[2]

        outside_term = 1 / (6 * 32 * np.pi ** 5) * np.sin(2 * psi) * R ** 3 * jv(2, R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        sinn = (2 * np.cos(psi) ** 2 - 1) * np.sin(phi)
        coss = np.cos(phi)+np.sin(2*psi)

        beta_bar_times2 = np.arctan2(sinn, coss)

        """inside terms:"""

        A1_prime = np.sqrt((x3*np.cos(psi))**2+(x2*np.sin(psi))**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))
        A1p_sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)
        A1p_cos_alpha1 = (x3 * np.cos(psi) + x2 * np.sin(psi)) * np.sin((phi + phi1) / 2)
        alpha1 = np.arctan2(A1p_sin_alpha1, A1p_cos_alpha1)
        E1 = np.exp(1j*(phi3-phi2-2*alpha1-beta_bar_times2))

        A2_prime = np.sqrt((x1*np.cos(psi))**2+(x3*np.sin(psi))**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        A2p_sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)
        A2p_cos_alpha2 = (x1 * np.cos(psi) + x3 * np.sin(psi)) * np.sin((phi + phi2) / 2)
        alpha2 = np.arctan2(A2p_sin_alpha2, A2p_cos_alpha2)
        E2 = np.exp(1j*(2*phi-2*alpha2+beta_bar_times2+phi3-phi2-2*phi2))

        A3_prime = np.sqrt((x2*np.cos(psi))**2+(x1*np.sin(psi))**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        A3p_sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)
        A3p_cos_alpha3 = (x2 * np.cos(psi) + x1 * np.sin(psi)) * np.sin((phi + phi3) / 2)
        alpha3 = np.arctan2(A3p_sin_alpha3, A3p_cos_alpha3)
        E3 = np.exp(1j*(-2*phi-2*alpha3+beta_bar_times2+phi1-phi2+2*phi3))

        l1_1 = R*np.cos(psi)/A1_prime
        l1_2 = R*np.cos(psi)/A2_prime
        l1_3 = R*np.cos(psi)/A3_prime

        l2_1 = R*np.sin(psi)/A1_prime
        l2_2 = R*np.sin(psi)/A2_prime
        l2_3 = R*np.sin(psi)/A3_prime

        l3_1 = np.sqrt((R/A1_prime)**2*(1 + 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 + 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        first_term = E1 / A1_prime ** 4 * self.compute_kappa_bispectrum(l1_1,l2_1, l3_1, 4000, 500)
        second_term = E2 / A2_prime ** 4 * self.compute_kappa_bispectrum(l1_2, l2_2, l3_2, 4000, 500)
        third_term = E3 / A3_prime ** 4 * self.compute_kappa_bispectrum(l1_3, l2_3, l3_3, 4000, 500)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        if np.abs(complete_integrand[6]) > 10**(-11):
            print("l values, integral", l1_1[6], l2_1[6], l3_1[6], complete_integrand[6])
            print("one loop of integrand:", time.time()-measure0)

        return(complete_integrand)

    def gamma1(self, integ_limits, r, u, v):

        timenow = time.time()
        int_obj = Integrator(integ_limits)
        train = int_obj(functools.partial(self.gamma1_integrand, r, u, v), nitn=10, neval=10000)
        result = int_obj(functools.partial(self.gamma1_integrand, r, u, v), nitn=10, neval=40000)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma2(self, integ_limits, r, u, v):

        x2 = r*np.pi/(60*180)
        x3 = u*x2
        x1 = v*x3+x2

        new_r = x3*(60*180)/np.pi
        new_u = x1/x3
        new_v = (x2-x3)/x1

        timenow = time.time()
        int_obj = Integrator(integ_limits)
        train = int_obj(functools.partial(self.gamma1_integrand, new_r, new_u, new_v), nitn=10, neval=10000)
        result = int_obj(functools.partial(self.gamma1_integrand, new_r, new_u, new_v), nitn=10, neval=40000)
        timeafter = time.time()
        print("the time to integrate is", timeafter-timenow)

        return(result)

    def gamma3(self, integ_limits, r, u, v):

        x2 = r * np.pi / (60 * 180)
        x3 = u * x2
        x1 = v * x3 + x2

        new_r = x1 * (60 * 180) / np.pi
        new_u = x2 / x1
        new_v = (x3 - x1) / x2

        timenow = time.time()
        int_obj = Integrator(integ_limits)
        train = int_obj(functools.partial(self.gamma1_integrand, new_r, new_u, new_v), nitn=10, neval=10000)
        result = int_obj(functools.partial(self.gamma1_integrand, new_r, new_u, new_v), nitn=10, neval=40000)
        timeafter = time.time()
        print("the time to integrate is", timeafter - timenow)

        return (result)