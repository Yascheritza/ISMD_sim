import numpy as np


def SDESimulator(
        L=10.0,  # domain length
        N=1000,  # number of spatial grid points
        dt=0.01,  # time step
        T=10.0,  # final time
        b=2.0,  # birth rate
        d=1.0,  # intrinsic death rate
        dd=0.1,  # competition rate
        lam=1.0,  # initial density (individuals per unit length)
        phi_0=None,
        sigma_m=1.0,  # standard deviation for dispersal (birth) kernel
        sigma_w=0.1,  # standard deviation for competition kernel
        seed=None,  # random number generator seed
):
    """
    Simulate a spatial birth-death-competition model with multiplicative noise,
    using an Euler-Maruyama scheme with FFT-based convolution for the kernels.

    Parameters
    ----------
    L : float
        Domain length.
    N : int
        Number of spatial grid points.
    dt : float
        Time step.
    T : float
        Final simulation time.
    b : float
        Birth rate.
    d : float
        Intrinsic death rate.
    dd : float
        Competition rate.
    lam : float
        Initial density (individuals per unit length).
    phi_0: np.array
        Initial distribution vector
    sigma_m : float
        Standard deviation of the dispersal (birth) kernel.
    sigma_w : float
        Standard deviation of the competition kernel.
    seed : int, optional
        Random number generator seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary with keys:
          'final_phi': list of final density arrays (length num_runs)
          'x': spatial grid (array)
    """
    # Derived parameters:
    dx = L / N
    steps = int(T / dt)

    # Spatial grid: x from -L/2 to L/2 (endpoint excluded)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    # Build dispersal kernel: raw Gaussian (amplitude scales as 1/(sqrt(2pi)*sigma_w)).
    kernel_disp = np.exp(-x ** 2 / (2 * sigma_m ** 2)) / (np.sqrt(2 * np.pi) * sigma_m)

    # Build competition kernel: raw Gaussian (amplitude scales as 1/(sqrt(2pi)*sigma_w))
    kernel_comp = np.exp(-x ** 2 / (2 * sigma_w ** 2)) / (np.sqrt(2 * np.pi) * sigma_w)

    # Shift kernels for FFT convolution (centered at index 0)
    kernel_disp_fft = np.fft.fft(np.fft.ifftshift(kernel_disp))
    kernel_comp_fft = np.fft.fft(np.fft.ifftshift(kernel_comp))

    def fft_convolve(phi, kernel_fft):
        """
        Compute the convolution of phi with a kernel provided in Fourier space,
        using periodic boundary conditions.
        """
        phi_fft = np.fft.fft(phi)
        conv = np.real(np.fft.ifft(phi_fft * kernel_fft)) * dx
        return conv

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        
    if phi_0 is None:
        # Initial condition: sample from a Poisson point process (PPP)
        n_init = np.random.poisson(lam * dx, size=N)
        # Convert to density (individuals per unit length)
        phi = n_init.astype(float) / dx
    else:
        phi = phi_0

    # Time integration: Euler-Maruyama scheme
    for n in range(1, steps):
        # Convolution terms via FFT
        birth_conv = fft_convolve(phi, kernel_disp_fft)
        comp_conv = fft_convolve(phi, kernel_comp_fft)

        # Deterministic drift
        drift = b * birth_conv - d * phi - dd * phi * comp_conv

        # Noise amplitude (variance per dt)
        Gamma = b * birth_conv + d * phi + dd * phi * comp_conv

        # Generate multiplicative Gaussian noise (independent per grid cell)
        noise = np.sqrt(np.maximum(Gamma, 0)) * np.random.normal(0, 1, size=N)

        # Euler-Maruyama update
        phi = phi + dt * drift + noise * np.sqrt(dt)

        # Enforce non-negativity of density
        phi = np.maximum(phi, 0)

    results = {
        'final_phi': phi,
        'x': x,
    }

    return results
