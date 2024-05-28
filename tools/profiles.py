import numpy as np
import scipy as sp

class Constants:
    """
    Class for fixing all the constants. We can import the necessary constants from this class to have a consistent value of these constants throughout.
    """
    pi = np.pi
    Msun = 1.989*1e30 # kg
    kpc = 3.086*1e16 # km
    
    G = 4.302*1e-6 # Msun^-1 (km/s)^2 kpc
    
    h = 0.72
    H0 = h * 100 # (km/s) Mpc^-1
    err_H0 = 2 # percentage
    
    rho_crit = 3.*(H0**2)/(8.*pi*G)*1e-6 # Msun kpc^-3
    err_rho_crit = rho_crit * 2. * (err_H0/H0) # percentage
    
class UnitConversion:
  """
  Class for the necessary unit conversions
  """
  GeVbycc_to_Msunbykpc3 = 1e9/(37.96) # Msun kpc^-3
  kpc_to_km = 3.086 * 1e16
  
class MWDensity:
    """
    class of density profiles for various models of MW DM
    halo considering central spikes.
    """
    def __init__(self, halo_types = ['NFW'], spike_types = ['GS','no spike', 'NA','SH','SH-','BM'], **kwargs):
        """
        args:
            name - (string or list of strings) for 
            fixing model. Must be string(s) provided in
            available_names
        kwargs:
        'r'
        'gamma_c'
        """
        if 'r' in kwargs:
          self.r = r
        else:
          self.r = np.logspace(-9,0,200)
          
        if 'rho_sun' in kwargs:
          self.rho_sun = rho_sun
        else:
          self.rho_sun = 0.383
        self.halo_params()
        
        if 'gamma_c' in kwargs:
          self.gamma_c = kwargs.get('gamma_c')
        else:
          self.gamma_c = 0.4 
          
        self.halo_types = halo_types
        self.spike_types = spike_types
      
        self.density = {}
        for halo_type in self.halo_types:
            self.halo_type = halo_type
            self.halo_params()
            self.density[halo_type] = {}
            for spike_type in self.spike_types:
              # print (f"{halo_type}, {spike_type}")
              self.spike_type = spike_type
              self.spike_params()
              self.density[halo_type][spike_type] = self.mass_density()/UnitConversion.GeVbycc_to_Msunbykpc3
        
        
    def mass_density(self):
      """
      DM mass density of MW: 0 for r < 2R_S; sat for 2R_S <= r 
      < R_sat; spike for R_sat <= r < R_sp; halo for r >= R_sp
      """
      density = np.zeros(self.r.shape)
      
      indx1 = np.where(self.r >= self.R_sp)[0]
      density[indx1] = self.halo(self.r[indx1])
      
      indx2 = np.where(self.r < self.R_sp)[0]
      density[indx2] = self.spike(self.r[indx2])
      if not self.spike_type in ['NA', 'no spike']:
        density[indx2] = self.sat(self.r[indx2], density[indx2])
        
      indx3 = np.where(self.r < 2*self.R_S)[0]
      density[indx3] = 0
      
      return density
        
    def sat(self, r, density):
      """
      To model the saturation of the density profile above 
      rho_sat due to DM annihilation. A weak spike above the
      density threshold is assumed where rho_sat(r) = 
      rho_sat(R_sat)*(r/R_sat)^-0.5
      """
      self.rho_sat = 3.17 * 1e11 * UnitConversion.GeVbycc_to_Msunbykpc3
      self.gamma_sat = 0.5
      if np.any(density > self.rho_sat):
        self.R_sat = r[density > self.rho_sat][-1]
        density[density > self.rho_sat] = self.rho_sat*(r[r <= self.R_sat]/self.R_sat)**-self.gamma_sat
        return density
      else:
        return density
            
    def spike_params(self):   
        """
        To get the parameters of the spike model mentioned using 
        the keyword spike_type. These parameters will then be 
        used to calculated the density profile of spike.
        """
        self.gamma = 0
        self.Theat = 1.25*1e9
        self.tau = 0
        self.R_sp = 0.34*1e-3
        
        if self.spike_type == 'no spike':
          self.gamma_sp = 0
          self.R_sp = 0
          
        elif self.spike_type == 'GS':
          self.gamma_sp = 2.25
          
        elif self.spike_type == 'NA':
          self.gamma_sp = 2.25
          
        elif self.spike_type == 'SH':
          self.gamma_sp = 1.5
          
        elif self.spike_type == 'BM':
          self.gamma_sp = 2.25
          self.R_sp = self.R_sp*np.exp(-10.0/(self.gamma_sp - self.gamma))
          
        elif self.spike_type == 'SH-':
          self.gamma_sp = 2.25
          self.gamma_sp_heated = 1.5
          self.R_soft = 0.01 * 1e-3
          
        else:
          print (f"spike_type = {self.spike_type} is not in the available benchmark list: ['GS', 'NA', 'no spike','SH','SH-','BM']")
          
    def spike(self,r):
      """
      This function returns the density of the spike for a given range of radius r if r is a np.ndarray or the density at the radius r if r is a float.
      """
      r_is_a_float = False
      if isinstance(r, float):
        r = np.array([r])
        r_is_a_float = True
      
      density = np.zeros(r.shape)
      density[r >= self.R_sp] = self.halo(r[r >= self.R_sp])
      if self.spike_type in ['GS','NA','SH','BM']:
        density[r < self.R_sp] = self.halo(self.R_sp)*(r[r < self.R_sp]/self.R_sp)**-self.gamma_sp

      elif self.spike_type == 'SH-':
        density[r < self.R_sp] = self.halo(self.R_sp)*(r[r < self.R_sp]/self.R_sp)**-self.gamma_sp_heated
        density_soft = density[r < self.R_soft][-1]
        density[r < self.R_soft] = density_soft*(r[r < self.R_soft]/self.R_soft)**-self.gamma_sp

      elif self.spike_type == 'no spike':
        density = self.halo(r)
      
      else:
        density = np.zeros(np.shape(r))
      
      if r_is_a_float:
        return density[0]
      else:
        return density
        
    def halo_params(self):
      """
      For fixing the model independent parameters of MW.
      -------------
      r_s: NFW scale radius in kpc
      R_sun: Sun's galactocentric radius in kpc
      rho_sun: local density of DM particles at R_sun in GeV/cm^3
      rho_s: NFW scale radius in Msun/kpc^3
      M_BH: Mass of the Supermassive BH in Msun
      v_0: Velocity dispersion of the stars in the inner halo in km/s
      R_s: Schwarzchild radius of the BH in kpc
      r_c: The core radius of the halo in kpc 
      
      R_sp: The radius around which the spike starts to grown in kpc
      gamma: The logarithmic density slope of the DM halo (to be conservative it is assumed to be 0)
      Theat: Heating time in yrs
      tau: Time since the spike formed in units of the heating time
      """
      self.r_s = 18.6
      self.R_sun = 8.2
      self.rho_s = self.rho_sun*(self.R_sun/self.r_s)*(1 + self.R_sun/self.r_s)**2 *(UnitConversion.GeVbycc_to_Msunbykpc3)
      self.M_BH = 4.3 * 1e6
      self.v_0 = 105.0
      self.R_S = 2.95 * (self.M_BH) / (UnitConversion.kpc_to_km)
      self.r_c = 1.0
      
    def nfw(self, r):
      return self.rho_s * (r/self.r_s)**-1 * (1 + r/self.r_s)**-2
    
    def core(self, r):
      return self.nfw(self.r_c) * (r/self.r_c)**(-self.gamma_c)

    def halo(self, r):
      """
      Function for getting the density of the global DM halo density profile, which returns a float if r is a float and a np.ndarray if r is np.ndarray.
      """
      r_is_a_float = False
      if isinstance(r, float):
        r = np.array([r])
        r_is_a_float = True
        
      if self.halo_type == 'NFW':
        density = self.nfw(r)
      elif self.halo_type == 'Core':
        density = np.zeros(r.shape)
        density[r < self.r_c] = self.core(r[r < self.r_c])
        density[r >= self.r_c] = self.nfw(r[r >= self.r_c])
      else:
        print (f"halo_type = {self.halo_type} does not match with any in the available list: ['NFW', 'Core']")
        
      if r_is_a_float:
        return density[0]
      else:
        return density
      #'new'