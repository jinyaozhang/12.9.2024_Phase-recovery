import numpy as np
from proplib import propagate_as_tf
import tensorflow as tf


def cost_function(amp1_r, amp2_t, m_1, n_1):  # current cost formular
    cost = np.sqrt(np.sum(np.square(amp1_r - amp2_t)) / (m_1 * n_1))
    return cost


def conjugate_gradient(alpha0, max_iter,  current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling,
                       phase_1=0, tol=1e-7):

    costs = np.zeros(max_iter)
    phase_err = np.zeros(max_iter)
    # Convert inputs to TensorFlow tensors
    amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)
    amp2_tensor = tf.cast(amp2, tf.float64)
    m = tf.convert_to_tensor(m, dtype=tf.float64)
    n = tf.convert_to_tensor(n, dtype=tf.float64)
    wavelength1 = tf.convert_to_tensor(wavelength1, dtype=tf.complex64)
    n_0 = tf.convert_to_tensor(n_0, dtype=tf.complex64)
    sampling = tf.convert_to_tensor(sampling, dtype=tf.complex64)
    z_diff = tf.convert_to_tensor(z_vec1[1] - z_vec1[0], dtype=tf.complex64)

    # Initialize the estimated phase and the residual
    phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)
    r_k = None  # Current residual
    p_k = None  # Current search direction

    for k in range(max_iter):
        with tf.GradientTape() as tape:
            # Calculate the assumed optical field u1
            u1_es = amp1_tensor * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
            # Use the propagation function to compute the estimated optical field u2
            u2_es = propagate_as_tf(u1_es, z_diff, wavelength1, n_0, sampling)
            amp2_es = tf.cast(tf.abs(u2_es), tf.float64) # Calculate the amplitude of the estimated optical field u2
            cost = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor))) # Loss function

        # Compute the gradient
        grad = tape.gradient(cost, phase1_es)

        # Initialize the residual and search direction
        if k == 0:
            r_k = grad
            p_k = -r_k
        else:
            # Compute the conjugate coefficient
            r_k_new = grad
            beta_k = tf.reduce_sum(tf.square(r_k_new)) / tf.reduce_sum(tf.square(r_k))
            p_k = -r_k_new + beta_k * p_k
            r_k = r_k_new

        # Compute the step size alpha_k
        alpha_k = tf.reduce_sum(tf.square(r_k)) / tf.reduce_sum(tf.square(p_k))

        # Update the phase estimate phase1_es
        phase1_es.assign_add(alpha0 * alpha_k * p_k)

        costs[k] = cost
        phase_err[k] = np.sqrt(np.sum(np.square(np.float64(phase1_es) - phase_1)) / (m * n))

        # Check for convergence
        if tf.sqrt(tf.reduce_sum(tf.square(tf.abs(grad)))) < tol:
            print("Gradient value below tolerance -> Iterations terminated")
            break

    return phase1_es.numpy(), amp2_es.numpy(), costs, phase_err


def gradient_descent(alpha, iters, phase1_est, amp1, amp2, m, n, z_vec, wavelength, n0, sampling, phase1=0):
    costs = np.zeros(iters)
    phase_err = np.zeros(iters)
    for i in range(iters):
        phase1_est, amp2_est = gradient_descent_step(alpha, phase1_est, amp1, amp2, m, n, z_vec,
                                                                  wavelength, n0, sampling)
        costs[i] = cost_function(amp2, amp2_est, m, n)
        phase_err[i] = np.sqrt(np.sum(np.square(np.float64(phase1_est) - phase1)) / (m * n))
    return phase1_est, amp2_est, costs, phase_err


def gradient_descent_step(alpha, current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling_):
    amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)  # tf
    amp2_tensor = tf.convert_to_tensor(amp2, dtype=tf.complex64)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.complex64)
    m = tf.convert_to_tensor(m, dtype=tf.float64)
    n = tf.convert_to_tensor(n, dtype=tf.float64)
    wavelength1 = tf.convert_to_tensor(wavelength1, dtype=tf.complex64)
    n_0 = tf.convert_to_tensor(n_0, dtype=tf.complex64)
    sampling_ = tf.convert_to_tensor(sampling_, dtype=tf.complex64)
    z_diff = tf.convert_to_tensor(z_vec1[1] - z_vec1[0], dtype=tf.complex64)

    phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)

    with (tf.GradientTape() as tape):
        # Given the intensity i1, construct the hypothetical light field u1
        u1_es = tf.cast(amp1_tensor, tf.complex64) * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
        # After the propagation function, the estimated light field u2 is calculated
        u2_es = propagate_as_tf(u1_es, z_diff, wavelength1, n_0, sampling_)
        # Calculate the estimated intensity i2 of the light field u2
        amp2_es = tf.abs(u2_es)

        amp2_tensor = tf.cast((tf.abs(amp2_tensor)), tf.float64)
        amp2_es = tf.cast((tf.abs(amp2_es)), tf.float64)

        # use TensorFlow to get cost_
        cost_ = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor)) / (m * n))

    # Calculate the gradient of cost_ with respect to phase1_es,and renew
    dy_dx = tape.gradient(cost_, phase1_es)
    phase1_es.assign_sub(alpha * dy_dx)
    # return new phase1_est
    return phase1_es.numpy(), amp2_es.numpy()