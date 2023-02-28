import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.special import jv
from classy import Class

from bispectrum import bispectrum

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
        # print(np.shape(kgrid))
        # print(np.shape(kgrid_perm1))
        # print(np.shape(kgrid_perm2))
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

