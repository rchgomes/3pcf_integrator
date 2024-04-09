import numpy as np
from astropy.table import Table
from astropy.io import fits

class ThreePointDataClass:
    def __init__(self, name, bin_type, kernel1=None, kernel2=None, kernel3=None):
        """
        Initialize the 3pt data object.

        Args:
            name (str)     : The name of the data.
            bin_type (str) : The type of the bin. 
                             Should be 'SSS', 'SAS', or 'Multipole'.
            kernel1 (str)  : The kernel type for bin1.
            kernel2 (str)  : The kernel type for bin2.
            kernel3 (str)  : The kernel type for bin3.
        
        Description:
            This method initializes the 3pt data object with the provided 
            values. The values of name, bin_type, kernel1, kernel2, and kernel3 
            are stored as attributes of the object. The method also calls the 
            set_empty method to initialize the arrays.

        Note on bin_type:
            If bin_type is 'SSS', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, theta3, signal).
            If bin_type is 'SAS', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, phi, signal).
            If bin_type is 'Multipole', the 3pt data is stored in the form of 
            (z1, z2, z3, theta1, theta2, M, signal).

            Talking about the weak lensing analysis, 'SAS' bin_type is useful
            for shear 3PCF, 'Multipole' bin_type is useful for 3PCF multipole, 
            and 'SSS' bin_type is useful for aperture mass statistics.
        """
        self.name = name
        self.bin_type = bin_type
        self.kernel1 = kernel1 or 'nz_source'
        self.kernel2 = kernel2 or 'nz_source'
        self.kernel3 = kernel3 or 'nz_source'
        self.set_empty()

    def set_empty(self):
        """
        Initialize the arrays of the 3pt data.
        """
        # redshift bins
        self.z1 = np.array([], dtype=int)
        self.z2 = np.array([], dtype=int)
        self.z3 = np.array([], dtype=int)
        # triangle bins
        if self.bin_type == 'SSS':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.theta3 = np.array([])        
        elif self.bin_type == 'SAS':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.phi = np.array([])
        elif self.bin_type == 'Multipole':
            self.theta1 = np.array([])
            self.theta2 = np.array([])
            self.M      = np.array([], dtype=int)
        # signal
        self.signal = np.array([])
        # size
        self.size = 0

    def set_value(self, z1, z2, z3, b1, b2, b3, signal):
        """
        Set the value of the 3pt data.

        Args:
            z1 (float, array): The number of redshift bin.
            z2 (float, array): The number of redshift bin.
            z3 (float, array): The number of redshift bin.
            b1 (float, array): The value of bin1.
            b2 (float, array): The value of bin2.
            b3 (float, array): The value of bin3.
            signal (float, array): The signal value.

        Description:
            This method sets the value of the 3pt data by appending the provided 
            values to the respective arrays. The values of z1, z2, and z3 are 
            appended to the corresponding arrays. Depending on the bin type, the 
            values of b1, b2, and b3 are appended to the respective arrays.
            Finally, the signal value is appended to the signal array.

            If bin_type is 'SSS', the values of b1, b2, and b3 are appended to 
            the arrays theta1, theta2, and theta3 respectively.
            If bin_type is 'SAS', the values of b1, b2, and b3 are appended to
            the arrays theta1, theta2, and phi respectively.
            If bin_type is 'Multipole', the values of b1, b2, and b3 are appended
            to the arrays theta1, theta2, and M respectively.
        """
        if np.isscalar(z1):
            z1 = np.array([z1])
            z2 = np.array([z2])
            z3 = np.array([z3])
            b1 = np.array([b1])
            b2 = np.array([b2])
            b3 = np.array([b3])
            signal = np.array([signal])
        print(z1.size, z2.size, z3.size, b1.size, b2.size, b3.size, signal.size)
        assert z1.size == z2.size == z3.size == b1.size == b2.size == b3.size == signal.size, \
            'All the arrays should have the same size.'
        self.z1 = np.append(self.z1, z1)
        self.z2 = np.append(self.z2, z2)
        self.z3 = np.append(self.z3, z3)
        if self.bin_type == 'SSS':
            self.theta1 = np.append(self.theta1, b1)
            self.theta2 = np.append(self.theta2, b2)
            self.theta3 = np.append(self.theta3, b3)
        elif self.bin_type == 'SAS':
            self.theta1 = np.append(self.theta1, b1)
            self.theta2 = np.append(self.theta2, b2)
            self.phi = np.append(self.phi, b3)
        elif self.bin_type == 'Multipole':
            self.theta1 = np.append(self.theta1, b1)
            self.theta2 = np.append(self.theta2, b2)
            self.M = np.append(self.M, b3)
        self.signal = np.append(self.signal, signal)
        self.size = self.z1.size

    def set_covariance(self, cov, nsim=0):
        """
        Set the covariance matrix.

        Args:
            cov (array): The covariance matrix.
            nsim (int) : The number of simulations used to 
                         estimate the covariance matrix. This is used to 
                         estimate the Hartlap factor.

        Description:
            This method sets the covariance matrix of the 3pt data. The 
            covariance matrix is stored as an attribute of the object.
        """
        self.cov     = cov
        self.nsim4cov= nsim

    def to_fits(self, filename=None):
        """
        Write the 3pt data to the fits file.

        Args:
            filename (str): The name of the fits file.

        Returns:
            hdul (HDUList): The HDUList object.
        """
        ## HDUList
        primary = fits.PrimaryHDU()
        hdul = fits.HDUList([primary])
        ## DATA VECTOR
        # create table
        if self.bin_type == 'SSS':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.theta3, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'THETA3', 'VALUE']
        elif self.bin_type == 'SAS':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.phi, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'PHI', 'VALUE']
        elif self.bin_type == 'Multipole':
            data = [self.z1, self.z2, self.z3, self.theta1, self.theta2, self.M, self.signal]
            names= ['BIN1', 'BIN2', 'BIN3', 'THETA1', 'THETA2', 'M', 'VALUE']
        table = Table(data, names=names)
        # create header
        header = fits.Header()
        header['BIN_TYPE'] = self.bin_type
        header['EXTNAME']  = self.name
        header['KERNEL_1'] = self.kernel1
        header['KERNEL_2'] = self.kernel2
        header['KERNEL_3'] = self.kernel3
        header['3PT_DATA'] = True
        # create hdu
        hdu = fits.BinTableHDU(table, header=header)
        hdul.append(hdu)
        ## COVARIANCE MATRIX
        if hasattr(self, 'cov'):
            # create header
            header = fits.Header()
            header['EXTNAME']  = 'COV'
            header['NSIM']     = self.nsim4cov
            header['3PT_DATA'] = True
            # create hdu
            hdu = fits.ImageHDU(self.cov, header=header)
            hdul.append(hdu)
        # Write to file
        if filename:
            hdul.writeto(filename, overwrite=True)
        return hdul

    @classmethod
    def from_fits(cls, filename_or_hdul):
        """
        Read the 3pt data from the fits file.

        Args:
            filename_or_hdul (str, HDUList): The name of the fits file or HDUList
        
        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        if isinstance(filename_or_hdul, str):
            hdul = fits.open(filename_or_hdul)
        else:
            hdul = filename_or_hdul
        hdu = hdul[1]
        # read header
        header = hdu.header
        name = header['EXTNAME']
        bin_type = header['BIN_TYPE']
        kernel1 = header['KERNEL_1']
        kernel2 = header['KERNEL_2']
        kernel3 = header['KERNEL_3']
        obj = cls(name, bin_type, kernel1, kernel2, kernel3)
        # assign values
        data = hdu.data
        if bin_type == 'SSS':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['THETA3']
        elif bin_type == 'SAS':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['PHI']
        elif bin_type == 'Multipole':
            b1 = data['THETA1']
            b2 = data['THETA2']
            b3 = data['M']
        obj.set_value(data['BIN1'], data['BIN2'], data['BIN3'], b1, b2, b3, data['VALUE'])
        return obj

    def _parse_selection(self, val, which, condition, helper):
        """
        Parse the selection array for the given bin.

        Args:
            val (float, array)    : The bin value to select.
            which (str, array)    : The bin to select. 
            condition (str, array): The equality condition.
            helper (func)         : The helper function to get 
                                    the selection array.
        
        Returns:
            sel (array): The selection array.
        """
        # cast
        val = np.atleast_1d(val)
        which = np.atleast_1d(which)
        condition = np.atleast_1d(condition)
        n1 = val.size
        n2 = which.size
        n3 = condition.size
        n  = max([n1, n2, n3])
        if n>1:
            assert n1==1 or n1==n, 'val should be scalar or array of size n'
            assert n2==1 or n2==n, 'which should be scalar or array of size n'
            assert n3==1 or n3==n, 'condition should be scalar or array of size n'
        if n>1 and n1==1:
            val = np.repeat(val, n)
        if n>1 and n2==1:
            which = np.repeat(which, n)
        if n>1 and n3==1:
            condition = np.repeat(condition, n)

        # ``and'' selection
        sel = np.ones(self.size, dtype=bool)
        for v, w, c in zip(val, which, condition):
            sel &= helper(v, w, c)
        return sel        

    def selection_z_bin(self, z_val, which, condition='=='):
        """
        Get selection array for the given redshift bin.
        
        Args:
            z_val (int, array)    : The redshift bin to select.
            which (str, array)    : The redshift bin to select. 
                                    Should be 'z1', 'z2', or 'z3'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with z1 = 1, use:
            >>> sel = data.selection_zbin(1, 'z1')
            >>> data_z1 = data[sel]
            To select all the data points with z = 2, use:
            >>> sel = data.selection_zbin(2, 'z')
            >>> data_z = data[sel]
            To select all the data points with z1 >= 3, use:
            >>> sel = data.selection_zbin(3, 'z1', '>=')
            >>> data_z1 = data[sel]

        Returns:
            sel (array): The selection array.
        """
        def helper(z_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'z1' or which == 'z':
                sel &= compare(self.z1, z_val, condition)
            elif which == 'z2' or which == 'z':
                sel &= compare(self.z2, z_val, condition)
            elif which == 'z3' or which == 'z':
                sel &= compare(self.z3, z_val, condition)
            else:
                raise ValueError('which should be z1, z2 or z3')
            return sel

        sel = self._parse_selection(z_val, which, condition, helper)
        return sel

    def selection_SAS_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable 
        for SAS bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'phi'.
            condition (str, array): The equality condition.

        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select theta1 and theta2, use:
            >>> sel = data.selection_SASbin(0.1, 'theta')
            >>> data_theta = data[sel]
            To select all the data points with phi = 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'phi')
            >>> data_phi = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_SASbin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'phi':
                sel &= compare(self.phi, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta, or phi')
            return sel
        
        assert self.bin_type == 'SAS', \
            'This method is only applicable for SAS bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel
        
    def selection_SSS_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable
        for SSS bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'theta3'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select all the data points with theta3 = 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta3')
            >>> data_theta3 = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_SSSbin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'theta3' or which == 'theta':
                sel &= compare(self.theta3, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta3 or theta')
            return sel

        assert self.bin_type == 'SSS', \
            'This method is only applicable for SSS bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel

    def selection_Multipole_bin(self, b_val, which, condition='=='):
        """
        Get selection array for the given bin. This is applicable
        for Multipole bin type.

        Args:
            b_val (float, array)  : The bin value to select.
            which (str, array)    : The bin to select. 
                                    Should be 'theta1', 'theta2', or 'M'.
            condition (str, array): The equality condition.
        
        Example:
            To select all the data points with theta1 = 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta1')
            >>> data_theta1 = data[sel]
            To select all the data points with theta2 = 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta2')
            >>> data_theta2 = data[sel]
            To select all the data points with M = 0, use:
            >>> sel = data.selection_Multipole_bin(0, 'M')
            >>> data_M = data[sel]
            To select all the data points with theta1 >= 0.1, use:
            >>> sel = data.selection_Multipole_bin(0.1, 'theta1', '>=')
            >>> data_theta1 = data[sel]
        
        Returns:
            sel (array): The selection array.
        """
        def helper(b_val, which, condition):
            sel = np.ones(self.size, dtype=bool)
            if which == 'theta1' or which == 'theta':
                sel &= compare(self.theta1, b_val, condition)
            elif which == 'theta2' or which == 'theta':
                sel &= compare(self.theta2, b_val, condition)
            elif which == 'M':
                sel &= compare(self.M, b_val, condition)
            else:
                raise ValueError('which must be one of \
                    theta1, theta2, theta or M')
            return sel

        assert self.bin_type == 'Multipole', \
            'This method is only applicable for Multipole bin type.'
        sel = self._parse_selection(b_val, which, condition, helper)
        return sel
    
    def get_signal(self, sel=None):
        """
        Get the signal array.

        Args:
            sel (array): The selection array.
        
        Returns:
            signal (array): The signal array.
        """
        if sel is None:
            return self.signal
        else:
            return self.signal[sel]

    def get_covariance(self, sel=None):
        """
        Get the covariance matrix.

        Args:
            sel (array): The selection array.
        
        Returns:
            cov (array): The covariance matrix.
        """
        if sel is None:
            return self.cov
        else:
            return self.cov[np.ix_(sel, sel)]
    
    def get_inverse_covariance(self, sel=None, Hartlap=True):
        """
        Get the inverse covariance matrix.

        Args:
            sel (array): The selection array.
            Hartlap (bool): Whether to apply Hartlap factor.
        
        Returns:
            icov (array): The inverse covariance matrix.
        """
        cov = self.get_covariance(sel)
        if Hartlap:
            nsim = self.nsim4cov
            n = cov.shape[0]
            f = (nsim - n - 2) / (nsim - 1)
            cov /= f
        icov = np.linalg.inv(cov)
        return icov

    def get_z_bin(self, sel=None):
        """
        Get the redshift bin arrays.

        Args:
            sel (array): The selection array.
        
        Returns:
            z1, z2, z3 (array): The redshift bin arrays.
        """
        if sel is None:
            return self.z1, self.z2, self.z3
        else:
            return self.z1[sel], self.z2[sel], self.z3[sel]
    
    def get_t_bin(self, sel=None):
        """
        Get the triangle bin arrays.

        Args:
            sel (array): The selection array.
        
        Returns:
            b1, b2, b3 (array): The bin arrays.
        """
        if self.bin_type == 'SSS':
            if sel is None:
                return self.theta1, self.theta2, self.theta3
            else:
                return self.theta1[sel], self.theta2[sel], self.theta3[sel]
        elif self.bin_type == 'SAS':
            if sel is None:
                return self.theta1, self.theta2, self.phi
            else:
                return self.theta1[sel], self.theta2[sel], self.phi[sel]
        elif self.bin_type == 'Multipole':
            if sel is None:
                return self.theta1, self.theta2, self.M
            else:
                return self.theta1[sel], self.theta2[sel], self.M[sel]

    def copy(self):
        """
        Create a copy of the 3pt data object.

        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        hdul = self.to_fits()
        obj = ThreePointDataClass.from_fits(hdul)
        return obj

    def replace(self, sel):
        """
        Replace the 3pt data object with the selection/sort array.
        Note that this is a **destructive** method.

        Args:
            sel (array): The selection array.
        """
        self.z1 = self.z1[sel]
        self.z2 = self.z2[sel]
        self.z3 = self.z3[sel]
        self.theta1 = self.theta1[sel]
        self.theta2 = self.theta2[sel]
        if self.bin_type == 'SSS':
            self.theta3 = self.theta3[sel]
        elif self.bin_type == 'SAS':
            self.phi = self.phi[sel]
        elif self.bin_type == 'Multipole':
            self.M = self.M[sel]
        self.signal = self.signal[sel]
        self.size = self.z1.size
        if hasattr(self, 'cov'):
            self.cov = self.cov[np.ix_(sel, sel)]

    def sort(self, reverse=False, priority='z'):
        """
        Sort the order of entries.

        Args:
            reverse (bool): Whether to sort in reverse order.
            priority (str): The priority of sorting. 
                            Should be 'z' or 't', where 'z' sorts
                            the redshift bins and 't' sorts the triangle bins
                            as the primary sorting parameter and the other
                            as the secondary sorting parameter.

        Returns:
            obj (ThreePointDataClass): The 3pt data object.
        """
        # collect bins
        zbins = (self.z3, self.z2, self.z1)
        if self.bin_type == 'SSS':
            tbins = (self.theta3, self.theta2, self.theta1)
        elif self.bin_type == 'SAS':
            tbins = (self.phi, self.theta2, self.theta1)
        elif self.bin_type == 'Multipole':
            tbins = (self.M, self.theta2, self.theta1)
        # sort
        if priority == 'z':
            sel = np.lexsort(tbins+zbins)
        elif priority == 't':
            sel = np.lexsort(zbins+tbins)
        # reverse if needed
        if reverse:
            sel = sel[::-1]
        # return as a new object
        obj = self.copy()
        obj.replace(sel)
        return obj

def compare(array, val, condition):
    if condition == '==':
        return array == val
    elif condition == '>':
        return array > val
    elif condition == '<':
        return array < val
    elif condition == '>=':
        return array >= val
    elif condition == '<=':
        return array <= val
    else:
        raise ValueError('Condition "{}" not recognized.'.format(condition))