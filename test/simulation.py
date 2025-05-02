def test_simulation():
    """
    Test the simulation function.
    """
    import jax
    from antarrlib import freq2wnum, freq2wlen
    from antarrlib.position import incirc_trigrid
    from antarrlib.simulation import point_source, noise
    from antarrlib.imaging import subrange, steering_vector
    import numpy as np

    c0 = 299792458.0  # Speed of light [m/s]
    f0 = 47e6  # Center frequency [Hz]
    freq_offsets = np.array([-500e3, -250e3, 0, 250e3, 500e3])   # Frequency offsets [Hz]
    num_freqs = len(freq_offsets)

    # Wavenumber for each frequency.
    k = freq2wnum(f0 + freq_offsets)
    
    # Antenna settings.
    lambda_ = freq2wlen(f0)  # wavelength [m]
    radius = 10  # [m]
    antenna_positions = incirc_trigrid(lambda_ * 0.7, radius)
    num_antennas = len(antenna_positions)


    # Simulate point source.
    target_x = 2000  # m
    target_y = 3000  # m
    target_altitude = 80e3  # [m]
    target_position = np.array([target_x, target_y, target_altitude])

    received_signal = point_source(k, antenna_positions, target_position, rx_power=2.5)

    received_signal = received_signal.reshape((num_freqs*num_antennas, -1))
    
    print(received_signal.conj().T.dot(received_signal))

    nsubr = 32

    range_gate_width = 2e-6
    range_gate_lo = target_altitude - range_gate_width / 2 * c0

    c = subrange(range_gate_lo, range_gate_width*c0, nsubr)

    a = jax.jit(steering_vector)(k, antenna_positions, target_position, c)
    print(a)

