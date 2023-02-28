import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class

#for lambda CDM (knl and neff as in Smith et. al. 2003, which used different parameters)
class bispectrum:

    def __init__(self, cosmo_params, k, kgrid, z):

        self.cosmo_params = cosmo_params
        self.my_cosmo = Class()
        self.my_cosmo.set(self.cosmo_params)
        self.my_cosmo.compute()

        self.k = k
        self.z = z

        self.omegam = self.cosmo_params['Omega_b'] + self.cosmo_params['Omega_cdm']

        knl = self.my_cosmo.nonlinear_scale(z, len(z))
        self.knl = np.ndarray(shape=(len(k), len(z)))
        self.knl_grid = np.ndarray(shape=(len(z), len(kgrid)))

        for i in range(len(k)):
            self.knl[i] = knl

        for i in range(len(kgrid)):
            self.knl_grid[:,i] = knl

        self.Pk_linear = np.ndarray(shape=(len(k),len(z)))
        self.Pk_nonlinear = np.ndarray(shape=(len(k), len(z)))
        for i in range(len(self.Pk_linear)):
            for j in range(len(self.Pk_linear[0])):
                self.Pk_linear[i][j] = self.my_cosmo.pk_lin(k[i]*self.cosmo_params['h'],z[j])
                self.Pk_nonlinear[i][j] = self.my_cosmo.pk(k[i]*self.cosmo_params['h'], z[j])

        self.Pk_linear *= self.cosmo_params['h'] ** 3
        self.Pk_nonlinear *= self.cosmo_params['h'] ** 3
        #gridpoints = []
        #Pk_interp = []
        newpkinterp = []
        for kii in range(len(self.k)):
            newpkinterp.append([])
            for zi in range(len(self.z)):
                #gridpoints.append((self.k[kii],self.z[zi]))
                #Pk_interp.append(self.Pk_linear[kii][zi])
                newpkinterp[kii].append(self.Pk_linear[kii][zi])
        ## print(len(gridpoints))
        t1 = time.time()
        self.PL = RegularGridInterpolator((self.k, self.z), newpkinterp)
        #self.PL = LinearNDInterpolator(gridpoints, Pk_interp)
        t2 = time.time()
        # print("time first", t2-t1)

        self.r_from_z, self.dzdr_of_z = self.my_cosmo.z_of_r(z)

        self.r_from_z *= self.cosmo_params['h']
        self.dzdr_of_z *= self.cosmo_params['h']

        self.dzdr_from_r_func = interp1d(self.r_from_z, self.dzdr_of_z)
        self.z_from_r_func = interp1d(self.r_from_z, z)

    def compute_kernel(self, k1, k2, k3):
        dot = (-k3 ** 2 + k1 ** 2 + k2 ** 2) / 2
        f2 = 5 / 7 + 2 * dot ** 2 / (7 * k1 ** 2 * k2 ** 2) - dot * (1 / k1 ** 2 + 1 / k2 ** 2) / 2  # from Laila
        return (f2)

    def compute_kernel_grid(self, kgrid):
        dot = (-kgrid[:,:,2] ** 2 + kgrid[:,:,0] ** 2 + kgrid[:,:,1] ** 2) / 2
        f2 = 5 / 7 + 2 * dot ** 2 / (7 * kgrid[:,:,0] ** 2 * kgrid[:,:,1] ** 2) - dot * (1 / kgrid[:,:,0] ** 2 + 1 / kgrid[:,:,1] ** 2) / 2  # from Laila
        return (f2) #shape=(len(self.z), len(kgrid))

    def create_interpolated_bispectrum(self, kgrid, kvec):
        # print("tried")
        if self.model == 'bihalofit':
            # print("got in")
            gridpoints_bispec = []

            klen = len(kvec)
            # print(klen)
            # print(len(self.z))
            kzpoints = np.reshape(self.all_halo, newshape=(len(self.z), klen, klen, klen))
            # print(np.shape(kzpoints))

            t1 = time.time()
            self.interpolated_bispectrum = RegularGridInterpolator((self.z, kvec, kvec, kvec), kzpoints)
            t2 = time.time()
            # print("time", t2-t1)
            # print("interpotaled the full grid")


    def b_of_l(self, l1, l2, l3, chi):
        augment = np.ndarray(shape=(len(l1), len(l2), len(l3), len(chi)))
        for chii in range(len(chi)):
            z_actual = self.z_from_r_func(chi[chii])
            for l1i in range(len(l1)):
                for l2i in range(len(l2)):
                    for l3i in range(len(l3)):
                        ## print(self.r_from_z[0], self.r_from_z[-1], self.z[0], self.z[-1], chi[0], chi[-1], l1[0],
                              #l1[-1])
                        ## print(they)
                        ## print(l1[l1i], l2[l2i], l3[l3i], chi[chii])
                        augment[l1i][l2i][l3i][chii] = self.interpolated_bispectrum((z_actual, l1[l1i]/chi[chii], l2[l2i]/chi[chii], l3[l3i]/chi[chii]))

        #return(self.interpolated_bispectrum((self.z_from_r_func(chi), l1/chi, l2/chi, l3/chi)))
        return (augment)
    #def limber_projection(self):
    def b_of_l_equilateral(self, l0, chi):
        augment = np.ndarray(shape=(len(l0), len(chi)))
        for chii in range(len(chi)):
            z_actual = self.z_from_r_func(chi[chii])
            for l0i in range(len(l0)):
                augment[l0i][chii] = self.interpolated_bispectrum((z_actual, l0[l0i]/chi[chii], l0[l0i]/chi[chii], l0[l0i]/chi[chii]))

        #return(self.interpolated_bispectrum((self.z_from_r_func(chi), l1/chi, l2/chi, l3/chi)))
        return (augment)

    def compute_lensing_kernel(self, chimin, chimax, npoints, dndz):

        # print("into lensing kernel")
        chivals = np.linspace(chimin, chimax, npoints)
        zvals = self.z_from_r_func(chivals)
        nz = interp1d(dndz[:, 0], dndz[:, 1])

        def_z = np.trapz(dndz[:,1], dndz[:,0])
        # print(def_z)

        try:
            nvals = nz(zvals)*self.dzdr_from_r_func(chivals)
            #nvals = nz(zvals)
        except ValueError:
            raise ValueError(f"There was an interpolation problem. Your requested redshifts span from z={zvals[0]} to z={zvals[-1]}, "
                             f"while your redshift distribution function spans from z={dndz[0][0]} to z={dndz[-1][0]}")

        # print(chimin, chimax)
        def_int = np.trapz(nvals, chivals)
        nvals /= def_int
        lensing_kernel = np.zeros_like(chivals)
        for chii in range(len(chivals)-1):
            integrand = nvals[chii:]*(chivals[chii:]-chivals[chii]*np.ones_like(chivals[chii:]))/chivals[chii:]
            lensing_kernel[chii] = np.trapz(integrand, chivals[chii:])

        # print("did lensing kernel")
        self.lensing_kernel  = interp1d(chivals, lensing_kernel)
        # print("interpolated lensing kernel")

    def compute_kappa_bispectrum(self, l1, l2, l3, chimax, npoints):

        # print("into kappa bispectrum")
        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(350, chimax, npoints)
        chivals = np.ndarray(shape=(len(l1), len(l2), len(l3), len(chivals_simple)))
        for l1i in range(len(l1)):
            for l2i in range(len(l2)):
                for l3i in range(len(l3)):
                    chivals[l1i][l2i][l3i] = chivals_simple

        # print("will compute integrand")
        aa0 = time.time()
        integrand = (self.lensing_kernel(chivals)*(1+self.z_from_r_func(chivals)))**3/chivals*self.b_of_l(l1,l2,l3,chivals_simple)
        aa1 = time.time()
        # print("computed integrand")
        integral = np.trapz(integrand, chivals)
        aa2 = time.time()
        # print("computed integal. Times:", aa1-aa0, aa2-aa1)

        self.kappa_bispectrum = constant*integral

    def compute_kappa_bispectrum_equilateral(self, l0, chimax, npoints):

        # print("into kappa bispectrum")
        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(350, chimax, npoints)
        chivals = np.ndarray(shape=(len(l0), len(chivals_simple)))
        for l0i in range(len(l0)):
            chivals[l0i] = chivals_simple

        #test_constant = self.interpolated_bispectrum((0.55, 1, 1, 1))
        # print("will compute integrand")
        aa0 = time.time()
        integrand = (self.lensing_kernel(chivals)*(1+self.z_from_r_func(chivals)))**3/chivals*self.b_of_l_equilateral(l0,chivals_simple)
        #integrand = self.lensing_kernel(chivals) * (
        #            1 + self.z_from_r_func(chivals)) ** 3 / chivals * test_constant
        aa1 = time.time()
        # print("computed integrand")
        integral = np.trapz(integrand, chivals)
        aa2 = time.time()
        # print("computed integal. Times:", aa1-aa0, aa2-aa1)

        self.kappa_bispectrum = constant*integral

    def create_interpolated_kappa_bispectrum(self, lvec):

        before = time.time()
        self.interpolated_kappa_bispectrum = RegularGridInterpolator((lvec, lvec, lvec), self.kappa_bispectrum)
        after = time.time()
        # print("time to interpolate kappa bispectrum:", after - before)

    def gamma0_real_part_integrand(self, x1, x2, x3, phi, psi, R):

        outside_term =  1/(6*32*np.pi**5)*np.sin(2*psi)*(np.cos(phi)+2*np.cos(psi)*np.sin(psi))*R**3*jv(6,R)

        '''internal angles of the triangle'''
        phi1 = np.arccos((x1**2-x2**2-x3**2)/(-2*x2*x3))
        phi2 = np.arccos((x2**2-x3**2-x1**2)/(-2*x3*x1))
        phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))

        """inside terms:"""
        A1_prime = np.sqrt(x3*np.cos(psi)**2+x2*np.sin(psi)**2+x2*x3*np.sin(2*psi)*np.cos(phi+phi1))
        sin_alpha1 = (x3*np.cos(psi)-x2*np.sin(psi))*np.sin((phi+phi1)/2)/A1_prime
        cos_alpha1 = (x3*np.cos(psi)+x2*np.sin(psi))*np.sin((phi+phi1)/2)/A1_prime
        sin_6alpha1 = 2*sin_alpha1*cos_alpha1*(4*cos_alpha1**2-1)*(4*cos_alpha1**2-3)
        cos_6alpha1 = 1 - 2*sin_alpha1**2*(4*cos_alpha1**2-1)**2
        E1 = np.cos(phi2-phi3)*cos_6alpha1+np.sin(phi2-phi3)*sin_6alpha1

        A2_prime = np.sqrt(x1*np.cos(psi)**2+x3*np.sin(psi)**2+x3*x1*np.sin(2*psi)*np.cos(phi+phi2))
        sin_alpha2 = (x1*np.cos(psi)-x3*np.sin(psi))*np.sin((phi+phi2)/2)/A2_prime
        cos_alpha2 = (x1*np.cos(psi)+x3*np.sin(psi))*np.sin((phi+phi2)/2)/A2_prime
        sin_6alpha2 = 2*sin_alpha2*cos_alpha2*(4*cos_alpha2**2-1)*(4*cos_alpha2**2-3)
        cos_6alpha2 = 1 - 2*sin_alpha2**2*(4*cos_alpha2**2-1)**2
        E2 = np.cos(phi3-phi1)*cos_6alpha2+np.sin(phi3-phi1)*sin_6alpha2

        A3_prime = np.sqrt(x2*np.cos(psi)**2+x1*np.sin(psi)**2+x1*x2*np.sin(2*psi)*np.cos(phi+phi3))
        sin_alpha3 = (x2*np.cos(psi)-x1*np.sin(psi))*np.sin((phi+phi3)/2)/A3_prime
        cos_alpha3 = (x2*np.cos(psi)+x1*np.sin(psi))*np.sin((phi+phi3)/2)/A3_prime
        sin_6alpha3 = 2*sin_alpha3*cos_alpha3*(4*cos_alpha3**2-1)*(4*cos_alpha3**2-3)
        cos_6alpha3 = 1 - 2*sin_alpha3**2*(4*cos_alpha3**2-1)**2
        E3 = np.cos(phi1-phi2)*cos_6alpha3+np.sin(phi1-phi2)*sin_6alpha3

        l3_1 = np.sqrt((R/A1_prime)**2*(1 - 2*np.cos(psi)*np.sin(psi)*np.cos(phi)))
        l3_2 = np.sqrt((R / A2_prime) ** 2 * (1 - 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))
        l3_3 = np.sqrt((R / A3_prime) ** 2 * (1 - 2 * np.cos(psi) * np.sin(psi) * np.cos(phi)))

        first_term = E1 / A1_prime ** 4 * self.interpolated_kappa_bispectrum(R * np.cos(psi) / A1_prime,
                                                                        R * np.sin(psi) / A1_prime, l3_1)
        second_term = E2 / A2_prime ** 4 * self.interpolated_kappa_bispectrum(R * np.cos(psi) / A2_prime,
                                                                        R * np.sin(psi) / A2_prime, l3_2)
        third_term = E3 / A3_prime ** 4 * self.interpolated_kappa_bispectrum(R * np.cos(psi) / A3_prime,
                                                                        R * np.sin(psi) / A3_prime, l3_3)

        complete_integrand = outside_term*(first_term+second_term+third_term)

        return(complete_integrand)

