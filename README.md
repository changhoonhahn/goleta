# Cosmology with Galaxy Populations (CGPop)
The goal of this project is to infer cosmological parameters ($\Omega_m$, $\sigma_8$) from observations of galaxy populations:

$$p(\Omega_m, \sigma_8 | \{ X_i \})$$

where $X_i$ corresponds to photometry for galaxy $i$. We'll do the following: 

![cgpop](cgpop.png)

where we'll sample from $p(\theta^g_i | X_i)$ using [SEDflow](https://changhoonhahn.github.io/SEDflow/) and train a normalizing flow for $p(\theta^g | \Omega_m)$. Both $p(\theta^g_i | X_i)$ and $p(\theta^g | \Omega_m)$ will be trained using data from CAMELS TNG.  

