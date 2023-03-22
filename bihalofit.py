import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum

class bihalofit(bispectrum):

    def __init__(self, cosmo_params, k, z):

        bispectrum.__init__(self, cosmo_params, k, z)

        self.model = 'bihalofit'

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

    def compute_dependent_params(self, k1, k2, k3):

        if type(k1) == float:
            kvals = np.array([k1,k2,k3])
        else:
            kvals = np.ndarray(shape=(3, len(k1)))
            kvals[0] = k1
            kvals[1] = k2
            kvals[2] = k3

        sortedlist = np.sort(kvals, axis=0)
        kmin = sortedlist[0]
        kmid = sortedlist[1]
        kmax = sortedlist[2]
        r1 = kmin / kmax

        r2 = (kmid + kmin - kmax) / kmax

        an = 10 ** (-2.167 - 2.944 * self.logsigma8 - 1.106 * self.logsigma8 ** 2 - 2.865 * self.logsigma8 ** 3 - 0.310 * r1 ** self.gamman)
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff - 0.5745 * self.neff ** 2 + 10 ** (-0.9 + 0.2 * self.neff) * r2 ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff - 1.4995 * self.neff ** 2 - 0.2811 * self.neff ** 3 + 0.007 * r2)
        return (an, alphan, betan)

    def compute_one_halo(self, z, k1, k2, k3):

        knl = self.knl_interp(z)
        if type(k1) == float and type(z) == float:
            qvec = np.array(k1/knl,k2/knl,k3/knl)
        elif type(k1) == float:
            qvec = np.ndarray(shape=(3,len(z)))
            qvec[0] = k1 / knl
            qvec[1] = k2 / knl
            qvec[2] = k3 / knl
        else:
            qvec = np.ndarray(shape=(3,len(k1)))
            qvec[0] = k1 / knl
            qvec[1] = k2 / knl
            qvec[2] = k3 / knl
        an, alphan, betan = self.compute_dependent_params(k1, k2, k3)

        valuetot = 1
        for q in qvec:
            value = (1 / (an * q ** alphan + self.bn * q ** betan)) * (1 / (1 + (self.cn * q) ** (-1)))
            valuetot = valuetot * value

        return (valuetot)

    def I_func(self, ki, z):

        knl = self.knl_interp(z)
        return (1 / (1 + self.en * ki / knl))

    def P_enhanced(self, ki, z):

        knl = self.knl_interp(z)
        first = (1 + self.fn * (ki / knl) ** 2) / (1 + self.gn * (ki / knl) + self.hn * (ki / knl) ** 2)
        second = 1 / (self.mn * (ki / knl) ** self.mun + self.nn * (ki / knl) ** self.nun)
        third = 1 / (1 + (self.pn * ki / knl) ** (-3))

        tmp = first * self.PL((ki, z)) + second * third
        return (tmp)

    def compute_three_halo_part(self, z, k1, k2, k3):

        first_term = self.compute_kernel(k1,k2,k3) + self.dn * k3 / self.knl_interp(z)
        second_term = self.I_func(k1, z) * self.I_func(k2, z) * self.I_func(k3, z)
        third_term = self.P_enhanced(k1, z) * self.P_enhanced(k2, z)

        return (2 * first_term * second_term * third_term)

    def compute_three_halo(self, z, k1, k2, k3):
        self.three_halo = self.compute_three_halo_part(z, k1, k2, k3)+self.compute_three_halo_part(z, k2, k3, k1)+ self.compute_three_halo_part(z, k3, k1, k2)
        return(self.three_halo)

    def matter_bispectrum(self, z, k1, k2, k3):

        return(self.compute_one_halo(z,k1,k2,k3)+self.compute_three_halo(z, k1, k2, k3))
