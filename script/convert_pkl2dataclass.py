"""
This script converts the Rafael's pkl data to the threepoint data class.
"""
import numpy as np
import pickle
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../python'))
import threepoint

def load_obj(name):
    try:
        with open(name, 'rb') as f:
            return pickle.load(f)
    except:
        with open(name, 'rb') as f:
            return pickle.load(f, encoding='latin1')

def convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3,nsims, fname_out):
    data = load_obj(fname_pkl)
    thpt = threepoint.ThreePointDataClass('map3', 'SSS', sortz=True)
    nt = len(t1)
    for i in range(len(z1)):
        thpt.set_value(z1[i],z2[i],z3[i],t1,t2,t3,data['y_obs'][i*nt:(i+1)*nt])
    thpt.set_covariance(data['cov'], nsims)
    print('writing to', fname_out)
    thpt.to_fits(fname_out)

## Example
def main1():
    """Noiseless, auto-z only"""
    fname_pkl = 'data/nz/map3_dv_COSMOGRID_7Mar24_REDSHIFT_AUTOCORRELATIONS_ONLY.pkl'
    z1 = [1, 2, 3, 4]
    z2 = [1, 2, 3, 4]
    z3 = [1, 2, 3, 4]
    t1 = [7.0, 14.0, 25.0, 50.0]
    t2 = [7.0, 14.0, 25.0, 50.0]
    t3 = [7.0, 14.0, 25.0, 50.0]
    nsims = 100 #?
    fname_out = 'data/nz/map3_dv_COSMOGRID_7Mar24_REDSHIFT_AUTOCORRELATIONS_ONLY.fits'
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

def main2():
    """Noisy, auto-z only"""
    fname_pkl = 'data/nz/map3_dv_NOISY_COSMOGRID_29Mar24_REDSHIFT_AUTOCORRELATIONS_ONLY.pkl'
    z1 = [1, 2, 3, 4]
    z2 = [1, 2, 3, 4]
    z3 = [1, 2, 3, 4]
    t1 = [7.0, 14.0, 25.0, 50.0]
    t2 = [7.0, 14.0, 25.0, 50.0]
    t3 = [7.0, 14.0, 25.0, 50.0]
    nsims = 100 #?
    fname_out = 'data/nz/map3_dv_NOISY_COSMOGRID_29Mar24_REDSHIFT_AUTOCORRELATIONS_ONLY.fits'
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

def main3():
    """Noisy, auto-z + 334+344 cross"""
    fname_pkl = 'data/nz/map3_dv_NOISY_COSMOGRID_29Mar24_AUTO_AND_SOME_CROSS_CORRELATIONS.pkl'
    z1 = [1, 2, 3, 3, 3, 4]
    z2 = [1, 2, 3, 3, 4, 4]
    z3 = [1, 2, 3, 4, 4, 4]
    t1 = [7.0, 14.0, 25.0, 50.0]
    t2 = [7.0, 14.0, 25.0, 50.0]
    t3 = [7.0, 14.0, 25.0, 50.0]
    nsims = 100 #? 
    fname_out = 'data/nz/map3_dv_NOISY_COSMOGRID_29Mar24_AUTO_AND_SOME_CROSS_CORRELATIONS.fits'
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

def main4():
    """Noisy, all auto & cross"""
    fname_pkl = 'data/nz/map3_dv_NOISY_COSMOGRID_10Apr24_ALL_ZCORRELATIONS.pkl'
    z1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4]
    z2 = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 2, 2, 2, 3, 3, 4, 3, 3, 4, 4]
    z3 = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 2, 3, 4, 3, 4, 4, 3, 4, 4, 4]
    t1 = [7.0, 14.0, 25.0, 40.0]
    t2 = [7.0, 14.0, 25.0, 40.0]
    t3 = [7.0, 14.0, 25.0, 40.0]
    nsims = 400
    fname_out = 'data/nz/map3_dv_NOISY_COSMOGRID_10Apr24_ALL_ZCORRELATIONS.fits'
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

def main5():
    """full zbin"""
    fname_pkl = 'data/nz/map3_dv_COSMOGRID_19Apr24.pkl'
    fname_out = 'data/nz/map3_dv_COSMOGRID_19Apr24.fits'
    z = np.array([[1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,2,2], 
                  [1,2,3], [1,2,4], [1,3,3], [1,3,4], [1,4,4], 
                  [2,2,2], [2,2,3], [2,2,4], [2,3,3], [2,3,4], 
                  [2,4,4], [3,3,3], [3,3,4], [3,4,4], [4,4,4]])
    z1, z2, z3 = z.T
    t1 = [7.0, 14.0, 25.0, 40.0]
    t2 = [7.0, 14.0, 25.0, 40.0]
    t3 = [7.0, 14.0, 25.0, 40.0]
    nsims = 400
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

def main6():
    """full bin 320 sims"""
    fname_pkl = 'data/nz/map3_dv_NOISY_COSMOGRID_11Apr24_ALL_ZCORRELATIONS_320SIMS.pkl'
    fname_out = 'data/nz/map3_dv_NOISY_COSMOGRID_11Apr24_ALL_ZCORRELATIONS_320SIMS.fits'
    z = np.array([[1,1,1], [1,1,2], [1,1,3], [1,1,4], [1,2,2], 
                  [1,2,3], [1,2,4], [1,3,3], [1,3,4], [1,4,4], 
                  [2,2,2], [2,2,3], [2,2,4], [2,3,3], [2,3,4], 
                  [2,4,4], [3,3,3], [3,3,4], [3,4,4], [4,4,4]])
    z1, z2, z3 = z.T
    t1 = [7.0, 14.0, 25.0, 40.0]
    t2 = [7.0, 14.0, 25.0, 40.0]
    t3 = [7.0, 14.0, 25.0, 40.0]
    nsims = 400
    convert_pkl2dataclass(fname_pkl, z1, z2, z3, t1, t2, t3, nsims, fname_out)

if __name__ == '__main__':
    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()
