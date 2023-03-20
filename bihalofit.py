import numpy as np
import matplotlib.pyplot as plt
from bispectrum import bispectrum

'''Specify the cosmology'''
Omega_b = 0.05
Omega_m = 0.308
Omega_cdm = Omega_m - Omega_b
h = 0.678
A_s = 2.1e-9
n_s = 0.968

"Maximum scale in units of h/Mpc"
k_max = 50

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

'''Create ranges of k and z for the CLASS power spectrum'''
my_k = np.logspace(-3, np.log10(k_max), num=10000) #h/Mpc^-1
my_k_reduced = np.logspace(-4, np.log10(k_max), num=1000) #h/Mpc^-1
my_z = np.linspace(0,5,1000)
my_z_simple = np.array([0,0.55])
my_z_new = np.linspace(0,3,1000)

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

        kvals = np.ndarray(shape=(3, len(k1)))
        kvals[0] = k1
        kvals[1] = k2
        kvals[2] = k3

        sortedlist = np.sort(kvals, axis=0)
        kmin = sortedlist[0]
        kmid = sortedlist[1]
        kmax = sortedlist[2]
        r1 = kmin / kmax
        #print("r1", r1)
        r2 = (kmid + kmin - kmax) / kmax

        an = 10 ** (-2.167 - 2.944 * self.logsigma8 - 1.106 * self.logsigma8 ** 2 - 2.865 * self.logsigma8 ** 3 - 0.310 * r1 ** self.gamman)
        alphan = 10 ** (
            np.minimum(-4.348 - 3.006 * self.neff - 0.5745 * self.neff ** 2 + 10 ** (-0.9 + 0.2 * self.neff) * r2 ** 2, self.paramns))
        betan = 10 ** (-1.731 - 2.845 * self.neff - 1.4995 * self.neff ** 2 - 0.2811 * self.neff ** 3 + 0.007 * r2)
        return (an, alphan, betan)

    def compute_one_halo(self, z, k1, k2, k3):

        knl = self.knl_interp(z)
        qvec = np.ndarray(shape=(3,len(k1)))
        qvec[0] = k1 / knl
        qvec[1] = k2 / knl
        qvec[2] = k3 / knl
        an, alphan, betan = self.compute_dependent_params(k1, k2, k3)

        valuetot = 1
        for q in qvec:
            value = (1 / (an * q ** alphan + self.bn * q ** betan)) * (1 / (1 + (self.cn * q) ** (-1)))
            valuetot = valuetot * value

        #print("one halo", valuetot)
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

class tree_level_bispectrum(bispectrum):

    def __init__(self, cosmo_params, k, kgrid, z):

        bispectrum.__init__(self, cosmo_params, k, kgrid, z)

    def compute_tree_level(self, k1,k2,k3):
        return(2*(self.compute_kernel(k1,k2,k3)*self.PL((k1,0.55))*self.PL((k2, 0.55)) +
                  self.compute_kernel(k2,k3,k1)*self.PL((k2, 0.55))*self.PL((k3, 0.55)) +
                  self.compute_kernel(k3,k1,k2)*self.PL((k3, 0.55))*self.PL((k1, 0.55))))

def f_h3(x1, x2, x3): #https://arxiv.org/pdf/astro-ph/0207454.pdf between eq 12 and 13
    return(np.sqrt(2*x1**2+2*x2**2-x3**2)/2)

def f_psi3(x1,x2,x3):
    h3 = f_h3(x1,x2,x3)
    phi3 = np.arccos((x3**2-x1**2-x2**2)/(-2*x1*x2))
    sin_of_2_psi = (x2**2-x1**2)*x1*x2*np.sin(phi3)/(h3*x3)**2
    cos_of_2_psi = ((x2**2-x1**2)**2 - 4*(x1*x2)**2*np.sin(phi3)**2)/4*(h3*x3)**2
    return(np.arctan2(sin_of_2_psi, cos_of_2_psi)/2)

def f_psi1(x1, x2, x3):
    return(f_psi3(x2, x3, x1))

def f_psi2(x1,x2,x3):
    return(f_psi3(x3,x1,x2))

def transform_gamma(gamma, num, r, u, v):

    x2 = r * np.pi / (60 * 180)
    x3 = u * x2
    x1 = v * x3 + x2

    psi1 = f_psi1(x1,x2,x3)
    psi2 = f_psi2(x1,x2,x3)
    psi3 = f_psi3(x1,x2,x3)

    print(psi1, psi2, psi3)
    if num == 0:
        gamma_transf = gamma * np.exp(2*1j*(psi1+psi2+psi3))

    if num == 1:
        gamma_transf = gamma * np.exp(2 * 1j * (-psi1 + psi2 + psi3))

    if num ==2:
        gamma_transf = gamma * np.exp(2 * 1j * (psi1 - psi2 + psi3))

    if num == 3:
        gamma_transf = gamma * np.exp(2 * 1j * (psi1 + psi2 - psi3))

    return(gamma_transf)


model = bihalofit(params, my_k, my_z_new)
othermodel = bihalofit(params, my_k, my_z_new)

#mo1 = model.compute_one_halo(my_kgrid)
#mo3 = model.compute_three_halo(my_kgrid)
#mo4 = model.compute_all_halo(my_kgrid)

la = np.linspace(100,999,30)
lb = np.logspace(3, np.log10(9000), 10)
lmixed = np.concatenate((la,lb))
l = np.linspace(2,4000,400)
lbb = np.logspace(np.log10(100),np.log10(9000),100)

maximum_distance = 4400
mynz = np.loadtxt("bin_04_desy3_source_nz.dat")

plt.plot(mynz[:,0], mynz[:,1])
plt.show()
#aaa = time.time()
#model.all_halo = np.load("matter_bispectrum_150_kbins_logkmin2and65_kmax50.npy")
#np.save("matter_bispectrum_150_kbins_logkmin3and7_kmax10", model.all_halo)
#model.create_interpolated_bispectrum(my_kgrid, my_k_reduced)
#bbb = time.time()
model.compute_lensing_kernel(1, maximum_distance, 10000, mynz)
#ccc = time.time()

'''input in acrmins'''
my_equilateral_xs = np.logspace(np.log10(3),np.log10(35), num=15)
print(my_equilateral_xs)
#my_equilateral_xs = np.logspace(np.log10(1.5),np.log10(7.5), num=15)
my_u_values = 1*np.ones_like(my_equilateral_xs)
my_v_values = np.zeros_like(my_equilateral_xs)
#my_v_values = 0.532*np.ones_like(my_equilateral_xs)
#my_v_values = 0*np.ones_like(my_equilateral_xs)

limits = [[0, 2*np.pi/2],[0, np.pi/2],[0,50]]

#est_integration_0 = model.gamma0(limits, my_equilateral_xs, my_u_values, my_v_values)
#a_vals = [test_integration_0[i].mean for i in range(len(test_integration_0))]
#result_array_0 = transform_gamma(aa_vals, 0, my_equilateral_xs, my_u_values, my_v_values)
#np.save("Gamma0_15bins_d2min3_d2max35_phi60_no_interps_small", np.real(result_array_0))

test_integration_1 = model.gamma1(limits, my_equilateral_xs, my_u_values, my_v_values)
bb_vals = [test_integration_1[i].mean for i in range(len(test_integration_1))]
result_array_1 = transform_gamma(bb_vals, 1, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma1_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_1))
print(asdg)
test_integration_2 = model.gamma2(limits, my_equilateral_xs, my_u_values, my_v_values)
cc_vals = [test_integration_2[i].mean for i in range(len(test_integration_2))]
result_array_2 = transform_gamma(cc_vals, 2, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma2_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_2))

test_integration_3 = model.gamma3(limits, my_equilateral_xs, my_u_values, my_v_values)
dd_vals = [test_integration_3[i].mean for i in range(len(test_integration_3))]
result_array_3 = transform_gamma(dd_vals, 3, my_equilateral_xs, my_u_values, my_v_values)
np.save("Gamma3_15bins_d2min3_d2max35_phi60_centroid_no_linterp_no_kinterp", np.real(result_array_3))

result_ttt = (result_array_0+result_array_1+result_array_2+result_array_3)/4

#np.save("gamma_ttt_15bins_d2min3_d2max35_phi60_40log_lbins_lmin100_lmax9000_NEW_PENH", result_ttt)
#test_integration_0b = othermodel.gamma0(limits, my_equilateral_xs, my_u_values, my_v_values)
#test_integration_1b = othermodel.gamma1(limits, my_equilateral_xs, my_u_values, my_v_values)
#test_integration_2b = othermodel.gamma2(limits, my_equilateral_xs, my_u_values, my_v_values)
#test_integration_3b = othermodel.gamma3(limits, my_equilateral_xs, my_u_values, my_v_values)

result_array_0b = np.ones_like(test_integration_0b)
result_array_1b = np.ones_like(test_integration_1b)
result_array_2b = np.ones_like(test_integration_2b)
result_array_3b = np.ones_like(test_integration_3b)
for i in range(len(test_integration_0b)):
    result_array_0b[i] = np.real(test_integration_0b[i].mean)
    result_array_1b[i] = np.real(test_integration_1b[i].mean)
    result_array_2b[i] = np.real(test_integration_2b[i].mean)
    result_array_3b[i] = np.real(test_integration_3b[i].mean)

result_tttb = (result_array_0b+result_array_1b+result_array_2b+result_array_3b)/4

diffs = (result_ttt - result_tttb)/result_ttt

plt.plot(my_equilateral_xs, diffs)
plt.show()
#np.save("gamma_ttt_90lbins", result_ttt)

#aaa = time.time()
#model.create_interpolated_bispectrum(my_kgrid, my_k_reduced)
#bbb = time.time()
#model.compute_lensing_kernel(90, maximum_distance, 10000, mynz)
#ccc = time.time()
#model.compute_kappa_bispectrum_equilateral(l,maximum_distance,1000)
#ddd = time.time()
#print("the times are", bbb-aaa, ccc-bbb, ddd-ccc)
#kbsp_equi = model.kappa_bispectrum
#np.save("test_kappa_bispectum_to9000_morez", model.kappa_bispectrum)

#kbsp = np.load("test_kappa_bispectum.npy")
#kbsp = np.load("test_kappa_bispectum_to2000_morez.npy")
#np.save("bihalofit_test_on_grid_1halo", mo1)
#np.save("bihalofit_test_on_grid_3halo", mo3)
#np.save("bihalofit_test_on_grid", mo4)

kbspl = []
print(np.shape(kbsp))
for i in range(len(l)):
    for j in range(len(l)):
        for k in range(len(l)):
            if i==j and i==k:
                kbspl.append(kbsp[i][j][k])
kbspl = np.array(kbspl)
print(len(kbspl))


plt.title("Convergence bispectrum for equilateral triangles")
plt.xlabel("l")
plt.ylabel("B(l,l,l)*l**3")
plt.plot(l,kbspl*l**3, color='m', ls = '--', label="Bihalofit (kappa)")
#plt.plot(l,kbsp_equi*l**3, color='m', ls = '--', label="Bihalofit (kappa)")
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
