import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class

#linear p(k)
#Start by specifying the cosmology
Omega_b = 0.05
Omega_m = 0.308
Omega_cdm = Omega_m - Omega_b
h = 0.678
A_s = 2.1e-9
n_s = 0.968

k_max = 30 #UNITS: h/Mpc

params = {
             'output':'mPk',
             'non linear':'halofit',
             'Omega_b':Omega_b,
             'Omega_cdm':Omega_cdm,
             'h':h,
             'A_s':A_s,
             'n_s':n_s,
             'P_k_max_1/Mpc':k_max*h,
             'z_max_pk':10.
}

#cosmo = Class()
#cosmo.set(params)
#cosmo.compute()

my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
my_k_reduced = np.logspace(-2, np.log10(k_max), num=40) #h/Mpc^-1
my_z = np.linspace(0,5,1000)
my_z_simple = np.array([0,0.55])
my_z_new = np.linspace(0,2.5,100)
#z = 0.55

#plin = np.array([cosmo.pk_lin(ki, z) for ki in k])
#k /= h
#plin *= h**3

#PL = interp1d(k,plin)

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
                self.Pk_linear[i][j] = self.my_cosmo.pk_lin(k[i]*h,z[j])
                self.Pk_nonlinear[i][j] = self.my_cosmo.pk(k[i]*h, z[j])

        self.Pk_linear *= h ** 3
        self.Pk_nonlinear *= h ** 3
        #gridpoints = []
        #Pk_interp = []
        newpkinterp = []
        for kii in range(len(self.k)):
            newpkinterp.append([])
            for zi in range(len(self.z)):
                #gridpoints.append((self.k[kii],self.z[zi]))
                #Pk_interp.append(self.Pk_linear[kii][zi])
                newpkinterp[kii].append(self.Pk_linear[kii][zi])
        #print(len(gridpoints))
        t1 = time.time()
        self.PL = RegularGridInterpolator((self.k, self.z), newpkinterp)
        #self.PL = LinearNDInterpolator(gridpoints, Pk_interp)
        t2 = time.time()
        print("time first", t2-t1)

        self.r_from_z, self.dzdr_of_z = self.my_cosmo.z_of_r(z)

        self.r_from_z *= h
        self.dzdr_of_z *= h

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
        print("tried")
        if self.model == 'bihalofit':
            print("got in")
            gridpoints_bispec = []

            klen = len(kvec)
            print(klen)
            print(len(self.z))
            kzpoints = np.reshape(self.all_halo, newshape=(len(self.z), klen, klen, klen))
            print(np.shape(kzpoints))

            t1 = time.time()
            self.interpolated_bispectrum = RegularGridInterpolator((self.z, kvec, kvec, kvec), kzpoints)
            t2 = time.time()
            print("time", t2-t1)
            print("interpotaled the full grid")


    def b_of_l(self, l1, l2, l3, chi):
        augment = np.ndarray(shape=(len(l1), len(l2), len(l3), len(chi)))
        for chii in range(len(chi)):
            z_actual = self.z_from_r_func(chi[chii])
            for l1i in range(len(l1)):
                for l2i in range(len(l2)):
                    for l3i in range(len(l3)):
                        #print(self.r_from_z[0], self.r_from_z[-1], self.z[0], self.z[-1], chi[0], chi[-1], l1[0],
                              #l1[-1])
                        #print(they)
                        #print(l1[l1i], l2[l2i], l3[l3i], chi[chii])
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

        print("into lensing kernel")
        chivals = np.linspace(chimin, chimax, npoints)
        zvals = self.z_from_r_func(chivals)
        nz = interp1d(dndz[:, 0], dndz[:, 1])

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
        plt.show()
        print("interpolated lensing kernel")

    def compute_kappa_bispectrum(self, l1, l2, l3, chimax, npoints):

        print("into kappa bispectrum")
        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(350, chimax, npoints)
        chivals = np.ndarray(shape=(len(l1), len(l2), len(l3), len(chivals_simple)))
        for l1i in range(len(l1)):
            for l2i in range(len(l2)):
                for l3i in range(len(l3)):
                    chivals[l1i][l2i][l3i] = chivals_simple

        print("will compute integrand")
        aa0 = time.time()
        integrand = (self.lensing_kernel(chivals)*(1+self.z_from_r_func(chivals)))**3/chivals*self.b_of_l(l1,l2,l3,chivals_simple)
        aa1 = time.time()
        print("computed integrand")
        integral = np.trapz(integrand, chivals)
        aa2 = time.time()
        print("computed integal. Times:", aa1-aa0, aa2-aa1)

        self.kappa_bispectrum = constant*integral

    def compute_kappa_bispectrum_equilateral(self, l0, chimax, npoints):

        print("into kappa bispectrum")
        constant = 27*(100/299792)**6*self.omegam**3/8
        chivals_simple = np.linspace(350, chimax, npoints)
        chivals = np.ndarray(shape=(len(l0), len(chivals_simple)))
        for l0i in range(len(l0)):
            chivals[l0i] = chivals_simple

        #test_constant = self.interpolated_bispectrum((0.55, 1, 1, 1))
        print("will compute integrand")
        aa0 = time.time()
        integrand = (self.lensing_kernel(chivals)*(1+self.z_from_r_func(chivals)))**3/chivals*self.b_of_l_equilateral(l0,chivals_simple)
        #integrand = self.lensing_kernel(chivals) * (
        #            1 + self.z_from_r_func(chivals)) ** 3 / chivals * test_constant
        aa1 = time.time()
        print("computed integrand")
        integral = np.trapz(integrand, chivals)
        aa2 = time.time()
        print("computed integal. Times:", aa1-aa0, aa2-aa1)

        self.kappa_bispectrum = constant*integral

    def create_interpolated_kappa_bispectrum(self, lvec):

        before = time.time()
        self.interpolated_kappa_bispectrum = RegularGridInterpolator((lvec, lvec, lvec), self.kappa_bispectrum)
        after = time.time()
        print("time to interpolate kappa bispectrum:", after - before)

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


class bihalofit(bispectrum):

    def __init__(self, cosmo_params, k, kgrid, z):

        bispectrum.__init__(self, cosmo_params, k, kgrid, z)

        self.model = 'bihalofit'
        delta_r = 0.001
        sigma_high = np.array([self.my_cosmo.sigma(1/(knl_i)+delta_r/2, zi) for knl_i, zi in zip(self.knl[0], z)])
        sigma = np.array([self.my_cosmo.sigma(1/knl_i, zi) for knl_i, zi in zip(self.knl[0], z)])
        sigma_low = np.array([self.my_cosmo.sigma(1/(knl_i)-delta_r/2, zi) for knl_i, zi in zip(self.knl[0], z)])
        dsigmadr = (sigma_high-sigma_low)/delta_r

        ns = self.cosmo_params['n_s']
        omegam = self.omegam
        #self.neff = -3 - 2/(self.knl*sigma)*dsigmadr
        #self.neff = -1.55*np.ones(shape=(len(self.k), len(self.z)))
        self.neff = -1.55
        self.paramns = np.log10(1 - 2 * ns / 3)
        self.logsigma8 = np.log10(self.my_cosmo.sigma(8, 0, h_units = True))

        # bihalofit global params

        # one halo
        self.bn = 10 ** (-3.428 - 2.681 * self.logsigma8 + 1.624 * self.logsigma8 ** 2 - 0.095 * self.logsigma8 ** 3)
        self.cn = 10 ** (0.159 - 1.107 * self.neff)
        self.gamman = 10 ** (0.182 + 0.57 * self.neff)

        # three halo
        self.fn = 10 ** (-10.533 - 16.838 * self.neff - 9.3048 * self.neff ** 2 - 1.8263 * self.neff ** 3)
        self.gn = 10 ** (2.787 + 2.405 * self.neff + 0.4577 * self.neff ** 2)
        self.hn = 10 ** (-1.118 - 0.394 * self.neff)
        self.mn = 10 ** (-2.605 - 2.434 * self.logsigma8 + 5.710 * self.logsigma8 ** 2)
        self.nn = 10 ** (-4.468 - 3.080 * self.logsigma8 + 1.035 * self.logsigma8 ** 2)
        self.mun = 10 ** (15.312 + 22.977 * self.neff + 10.9579 * self.neff ** 2 + 1.6586 * self.neff ** 3)
        self.nun = 10 ** (1.347 + 1.246 * self.neff + 0.4525 * self.neff ** 2)
        self.pn = 10 ** (0.071 - 0.433 * self.neff)
        self.dn = 10 ** (-0.483 + 0.892 * self.logsigma8 - 0.086 * omegam)
        self.en = 10 ** (-0.632 + 0.646 * self.neff)

    def compute_dependent_params_old(self, k1, k2, k3):

        tosort = [k1, k2, k3]
        sortedlist = np.sort(tosort)
        kmin = sortedlist[0]
        kmid = sortedlist[1]
        kmax = sortedlist[2]
        r1 = kmin / kmax
        r2 = (kmid + kmin - kmax) / kmax

        r1_full = np.ndarray(shape=(len(self.k), len(self.z)))
        r2_full = np.ndarray(shape=(len(self.k), len(self.z)))

        for i in range(len(self.z)):
            r1_full[:,i] = r1
            r2_full[:, i] = r2

        an = 10 ** (-2.167 - 2.944 * self.logsigma8 - 1.106 * self.logsigma8 ** 2 - 2.865 * self.logsigma8 ** 3 - 0.310 * r1_full ** self.gamman)
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff - 0.5745 * self.neff ** 2 + 10 ** (-0.9 + 0.2 * self.neff) * r2_full ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff - 1.4995 * self.neff ** 2 - 0.2811 * self.neff ** 3 + 0.007 * r2_full)
        return (an, alphan, betan)

    def compute_dependent_params(self, kgrid):

        sortedlist = np.sort(kgrid, axis=1)
        kmin = sortedlist[:,0]
        kmid = sortedlist[:,1]
        kmax = sortedlist[:,2]
        r1 = kmin / kmax
        r2 = (kmid + kmin - kmax) / kmax

        an = 10 ** (-2.167 - 2.944 * self.logsigma8 - 1.106 * self.logsigma8 ** 2 - 2.865 * self.logsigma8 ** 3 - 0.310 * r1 ** self.gamman)
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff - 0.5745 * self.neff ** 2 + 10 ** (-0.9 + 0.2 * self.neff) * r2 ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff - 1.4995 * self.neff ** 2 - 0.2811 * self.neff ** 3 + 0.007 * r2)
        return (an, alphan, betan)

    def compute_one_halo(self, kgrid):

        k1_full = np.ndarray(shape=(len(self.z), len(kgrid)))
        k2_full = np.ndarray(shape=(len(self.z), len(kgrid)))
        k3_full = np.ndarray(shape=(len(self.z), len(kgrid)))

        for i in range(len(self.z)):
            k1_full[i] = kgrid[:,0]
            k2_full[i] = kgrid[:, 1]
            k3_full[i] = kgrid[:, 2]

        qvec = [k1_full / self.knl_grid, k2_full / self.knl_grid, k3_full / self.knl_grid] #shape = (3, len(z), len(kgrid))
        an, alphan, betan = self.compute_dependent_params(kgrid)

        valuetot = 1
        for q in qvec:
            value = (1 / (an * q ** alphan + self.bn * q ** betan)) * (1 / (1 + (self.cn * q) ** (-1)))
            valuetot = valuetot * value

        self.one_halo = valuetot #shape = (len(self.z), len(kgrid))
        print("one halo done")
        return (self.one_halo)

    def I_func(self, ki):
        return (1 / (1 + self.en * ki / self.knl_grid))

    def P_enhanced(self, ki):

        #tenh = time.time()
        first = (1 + self.fn * (ki / self.knl_grid) ** 2) / (1 + self.gn * (ki / self.knl_grid) + self.hn * (ki / self.knl_grid) ** 2)
        second = 1 / (self.mn * (ki / self.knl_grid) ** self.mun + self.nn * (ki / self.knl_grid) ** self.nun)
        third = 1 / (1 + (self.pn * ki / self.knl_grid) ** (-3))
        #tenh2 = time.time()

        self.values = first[:,1], second[:,1], third[:,1]
        #tenh3 = time.time()
        print("interpolei")
        tmp  = first * self.PL((ki, 0.55)) + second * third
        #tenh4 = time.time()
        #print("time breakdown penh:", tenh2-tenh, tenh3-tenh2, tenh4-tenh3)

        return (tmp)

    def compute_three_halo_part(self, kgrid):

        time_into = time.time()
        kgrid_full = np.ndarray(shape=(len(self.z), len(kgrid), 3))
        k1_full = np.ndarray(shape=(len(self.z), len(kgrid)))
        k2_full = np.ndarray(shape=(len(self.z), len(kgrid)))
        k3_full = np.ndarray(shape=(len(self.z), len(kgrid)))
        for i in range(len(self.z)):
            k1_full[i] = kgrid[:,0]
            k2_full[i] = kgrid[:,1]
            k3_full[i] = kgrid[:,2]
            kgrid_full[i] = kgrid

        time0 = time.time()
        first_term = self.compute_kernel_grid(kgrid_full) + self.dn * k3_full / self.knl_grid
        time1 = time.time()
        second_term = self.I_func(k1_full) * self.I_func(k2_full) * self.I_func(k3_full)
        time2 = time.time()
        third_term = self.P_enhanced(k1_full) * self.P_enhanced(k2_full)
        time3 = time.time()
        self.PE = self.P_enhanced(k1_full)
        time4 = time.time()
        return (2 * first_term * second_term * third_term) #shape = (len(self.z), len(kgrid))

    def compute_three_halo(self, kgrid):
        kgrid_perm1 = np.ndarray(shape=np.shape(kgrid))
        kgrid_perm2 = np.ndarray(shape=np.shape(kgrid))
        kgrid_perm1[:,0] = kgrid[:,1]
        kgrid_perm1[:,1] = kgrid[:,2]
        kgrid_perm1[:,2] = kgrid[:,0]
        kgrid_perm2[:,0] = kgrid[:,2]
        kgrid_perm2[:,1] = kgrid[:,0]
        kgrid_perm2[:,2] = kgrid[:,1]
        print(np.shape(kgrid))
        print(np.shape(kgrid_perm1))
        print(np.shape(kgrid_perm2))
        self.three_halo = self.compute_three_halo_part(kgrid)+self.compute_three_halo_part(kgrid_perm1)+ self.compute_three_halo_part(kgrid_perm2)
        return(self.three_halo)

    def compute_all_halo(self, kgrid):

        try:
            self.all_halo = self.one_halo + self.three_halo
        except:
            self.all_halo = self.compute_one_halo(kgrid)+self.compute_three_halo(kgrid)

        return(self.all_halo)

class tree_level_bispectrum(bispectrum):

    def __init__(self, cosmo_params, k, kgrid, z):

        bispectrum.__init__(self, cosmo_params, k, kgrid, z)

    def compute_tree_level(self, k1,k2,k3):
        return(2*(self.compute_kernel(k1,k2,k3)*self.PL((k1,0.55))*self.PL((k2, 0.55)) +
                  self.compute_kernel(k2,k3,k1)*self.PL((k2, 0.55))*self.PL((k3, 0.55)) +
                  self.compute_kernel(k3,k1,k2)*self.PL((k3, 0.55))*self.PL((k1, 0.55))))

count = 0
diag = []
my_kgrid = []
for i in range(len(my_k_reduced)):
    for j in range(len(my_k_reduced)):
        for k in range(len(my_k_reduced)):
            my_kgrid.append([my_k_reduced[i], my_k_reduced[j], my_k_reduced[k]])
            if i==j and i==k:
                diag.append(count)
            count += 1
my_kgrid = np.array(my_kgrid)
print("my shape is", np.shape(my_kgrid))

print("diag=", diag)

bitree = tree_level_bispectrum(params, my_k, my_kgrid, my_z_simple)
model = bihalofit(params, my_k, my_kgrid, my_z_new)

#neee = bi.neff[:,1]
#print(bi.logsigma8)
#x1 = bi.compute_one_halo(my_k,my_k,my_k)
#x2 = bi.compute_three_halo(my_k,my_k,my_k)
#x3 = bi.compute_all_halo(my_k,my_k,my_k)
x4 = bitree.compute_tree_level(my_k,my_k,my_k)
print(x4)

mo1 = model.compute_one_halo(my_kgrid)
mo3 = model.compute_three_halo(my_kgrid)
mo4 = model.compute_all_halo(my_kgrid)

l = np.logspace(2,np.log10(9000), 10)

maximum_distance = 3500
mynz = np.loadtxt("bin_04_desy3_source_nz.dat")

plt.plot(mynz[:,0], mynz[:,1])
plt.show()
#aaa = time.time()
#model.create_interpolated_bispectrum(my_kgrid, my_k_reduced)
#bbb = time.time()
#model.compute_lensing_kernel(90, maximum_distance, 10000, mynz)
#ccc = time.time()
#model.compute_kappa_bispectrum(l,l,l,maximum_distance,500)
#ddd = time.time()
#print("the times are", bbb-aaa, ccc-bbb, ddd-ccc)
#model.create_interpolated_kappa_bispectrum()

aaa = time.time()
model.create_interpolated_bispectrum(my_kgrid, my_k_reduced)
bbb = time.time()
model.compute_lensing_kernel(90, maximum_distance, 10000, mynz)
ccc = time.time()
model.compute_kappa_bispectrum_equilateral(l,maximum_distance,1000)
ddd = time.time()
print("the times are", bbb-aaa, ccc-bbb, ddd-ccc)
kbsp_equi = model.kappa_bispectrum
#np.save("test_kappa_bispectum_to9000_morez", model.kappa_bispectrum)

#kbsp = np.load("test_kappa_bispectum.npy")
#kbsp = np.load("test_kappa_bispectum_to2000_morez.npy")
#np.save("bihalofit_test_on_grid_1halo", mo1)
#np.save("bihalofit_test_on_grid_3halo", mo3)
#np.save("bihalofit_test_on_grid", mo4)

#kbspl = []
#print(np.shape(kbsp))
#for i in range(len(l)):
#    for j in range(len(l)):
#        for k in range(len(l)):
#            if i==j and i==k:
#                kbspl.append(kbsp[i][j][k])
#kbspl = np.array(kbspl)
#print(len(kbspl))


plt.title("Convergence bispectrum for equilateral triangles")
plt.xlabel("l")
plt.ylabel("B(l,l,l)*l**3")
#plt.plot(l,kbspl*l**3, color='m', ls = '--', label="Bihalofit (kappa)")
plt.plot(l,kbsp_equi*l**3, color='m', ls = '--', label="Bihalofit (kappa)")
#plt.plot(my_k_reduced,equilat*my_k_reduced**3, color='g', ls = '--', label="Bihalofit (full) - in grid")
plt.grid()
#plt.xlim(10**(-2),20)
#plt.ylim(1, 3*10**7)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()

#mo1 = np.load("bihalofit_test_on_grid_1halo.npy")
#mo3 = np.load("bihalofit_test_on_grid_3halo.npy")
#mo4 = np.load("bihalofit_test_on_grid.npy")
equilat = mo4[17][diag]
x3 = np.load("bihalofit_newz_newk_test_allhalo.npy")

plt.title("Matter bispectrum for equilateral triangles")
plt.xlabel("k (h/Mpc)")
plt.ylabel("B(k,k,k)*k**3")
plt.plot(my_k,x3[:,1]*my_k**3, color='m', ls = '--', label="Bihalofit (full) - no grid")
plt.plot(my_k_reduced,equilat*my_k_reduced**3, color='g', ls = '--', label="Bihalofit (full) - in grid")
plt.grid()
plt.xlim(10**(-2),20)
plt.ylim(1, 3*10**7)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()

#x = bi.compute_all_halo(my_k,my_k,my_k)
#np.save("bihalofit_test", x)
x1 = np.load("bihalofit_newz_newk_test_onehalo.npy")
x2 = np.load("bihalofit_newz_newk_test_threehalo.npy")
x3 = np.load("bihalofit_newz_newk_test_allhalo.npy")
#print(np.shape(x))
y = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_actual_newk_3")
ydiv = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_actual_newk_3_CDIV2")
ytimes = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_actual_newk_3_CTIMES2")
yy = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_m99")
yydiv = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_m99_CDIV2")
yytimes = np.loadtxt("/Users/gchgomes/Documents/bispectrum_new_modeling/one_halo_bispectrum_full_m99_CTIMES2")

plt.title("Matter bispectrum for equilateral triangles")
plt.xlabel("k (h/Mpc)")
plt.ylabel("B(k,k,k)*k**3")
plt.plot(my_k,x3[:,1]*my_k**3, color='g', ls = '-', label="Bihalofit (full)")
plt.plot(my_k, x1[:,1]*my_k**3, color = 'g', ls=':', label="Bihalofit (one halo term)")
plt.plot(my_k, x2[:,1]*my_k**3, color='g', ls='--', label="Bihalofit (three halo term)")
plt.plot(my_k, x4*my_k**3, color='darkorange', label = "Tree level bispectrum")
plt.plot(yy[0],yy[1]*2*np.pi**2, color='c', ls = '-', label="One halo term (m99 profile)")
plt.plot(yydiv[0],yydiv[1]*2*np.pi**2, color='c', ls = '--', label="One halo term (m99 profile, c'=c/2)")
plt.plot(yytimes[0],yytimes[1]*2*np.pi**2, color='c', ls = ':', label="One halo term (m99 profile, c'=2c)")
plt.plot(y[0],y[1]*2*np.pi**2, color='m', ls = '-', label="One halo term (NFW profile)")
plt.plot(ydiv[0],ydiv[1]*2*np.pi**2, color='m',  ls = '--',label="One halo term (NFW profile, c'=c/2)")
plt.plot(ytimes[0],ytimes[1]*2*np.pi**2, color='m', ls = ':', label="One halo term (NFW profile, c'=2c)")
plt.grid()
plt.xlim(10**(-2),20)
plt.ylim(1, 3*10**7)
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
#plt.savefig("Matter_bispectrum_many_models_2.pdf", dpi=500)

plt.plot(my_z, bi.neff)
plt.show()

plt.plot(my_z, np.exp(bi.logsigma8))
plt.show()
print("done")


knl = 0.306
neff = -1.55
sigma8 = 0.83
omegam = Omega_m
ns = n_s #tentative value
paramns = np.log10(1-2*ns/3)