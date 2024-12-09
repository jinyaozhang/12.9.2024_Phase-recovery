import time
import numpy as np
import scipy.io as sio
from utils import propagate_as
from utils import propagate_as_tf
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    ####################################################################################################################
    # Simulation of the measurement data
    ####################################################################################################################
    start_time = time.time()
    d = input("(Sampling data:S / Experimental data:E) :")
    if d == 'S':
        s = input("Please choose algorithm,input A or B (Gradient Descent:A,Conjugate descent:B):")

        wavelength = 0.561  # wavelength
        n0 = 1  # Refractive index of air
        sampling = np.array(
         [2.4, 2.4])  # The physical distance between each pixel of the camera, horizontally and vertically
        z_vec = np.array([3.5e3, 11.5e3])  # The distance between the two samples and the camera
        img_size = [1024, 1024]
        M = img_size[0]
        N = img_size[1]

        filename = "optical_field_at_object_plane.mat"
        if not filename:
            raise Exception
        data_obj = sio.loadmat(filename)  # Load this file as a dictionary
        amp_obj = data_obj['amp_obj']  # Extract the magnitude of the object plane from the dictionary
        phase_obj = data_obj['phase_obj']  # Extract the phase of the object plane from the dictionary

        u_obj = amp_obj * np.exp(1j * phase_obj)

    # Use the propagation function to calculate the light field of sampling surface 1: u1
        u1 = propagate_as(u_obj, z_vec[0], wavelength, n0, sampling)
    # Use the propagation function to calculate the light field of sampling surface 2: u2
        u2 = propagate_as(u_obj, z_vec[1], wavelength, n0, sampling)

    # The amplitude of the light field at sampling surfaces 1 and 2
        i1 = np.abs(u1)**2
        i2 = np.abs(u2)**2
        amp_1 = np.sqrt(i1)
        amp_2 = np.sqrt(i2)

    # The true phase information of the light field u1 is extracted
        phase1 = np.angle(u1)
    else:
        s = input("Please choose algorithm,input A or B (Gradient Descent:A,Conjugate descent:B):")

        wavelength = 0.5610  # wavelength
        n0 = 1  # Refractive index of air
        sampling = np.array([2.4, 2.4])  # The physical distance between each pixel of the camera
        z_vec = np.array([3557.8, 5657])  # The distance between the two samples and the camera

        img_size = [3672, 4500]
        M = img_size[0]
        N = img_size[1]

        filename = "HPB_usaf_Phs_gruby.mat"
        if not filename:
            raise Exception
        data = sio.loadmat(filename)  # 将这个文件加载为一个字典

        i = data['OH']

        amp_1 = i[:, :, 0]
        amp_2 = i[:, :, 1]
        phase1 = np.angle(amp_1)
    ####################################################################################################################
    # The actual forward model
    ####################################################################################################################
    # phase1_est = phase1 # CHECK OUT THIS OPTION - UNCOMMENT THIS LINE

    def error_current_Function(amp1_r, amp2_t, m_1, n_1):  # current cost formular

        cost = np.sqrt(np.sum(np.square(amp1_r - amp2_t)) / (m_1 * n_1))
        return cost


    def conjugate_gradient(phase_1, current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling_, max_iter,
                           tol=1e-7):

        # Convert inputs to TensorFlow tensors
        amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)
        amp2_tensor = tf.convert_to_tensor(amp2, dtype=tf.complex64)
        m = tf.convert_to_tensor(m, dtype=tf.float64)
        n = tf.convert_to_tensor(n, dtype=tf.float64)
        wavelength1 = tf.convert_to_tensor(wavelength1, dtype=tf.complex64)
        n_0 = tf.convert_to_tensor(n_0, dtype=tf.complex64)
        sampling_ = tf.convert_to_tensor(sampling_, dtype=tf.complex64)
        z_diff = tf.convert_to_tensor(z_vec1[1] - z_vec1[0], dtype=tf.complex64)

        # Initialize the estimated phase and the residual
        phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)
        r_k = None  # Current residual
        p_k = None  # Current search direction

        for k in range(max_iter):
            with tf.GradientTape() as tape:
                # Calculate the assumed optical field u1
                u1_es = tf.cast(amp1_tensor, tf.complex64) * tf.exp(
                    tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
                # Use the propagation function to compute the estimated optical field u2
                u2_es = propagate_as_tf(u1_es, z_diff, wavelength1, n_0, sampling_)
                # Calculate the intensity of the estimated optical field u2
                amp2_es = tf.abs(u2_es)

                # Convert intensity to calculate the loss
                amp2_tensor = tf.cast(tf.abs(amp2_tensor), tf.float64)
                amp2_es = tf.cast(tf.abs(amp2_es), tf.float64)

                # Loss function
                cost_ = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor)) / (m * n))

            # Compute the gradient
            grad = tape.gradient(cost_, phase1_es)

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
            phase1_es.assign_add(alpha_k * p_k)

            error_current_in = error_current_Function(amp2, amp2_es, m, n)
            Costs.append(error_current_in)
            phase_error_in = np.sqrt(np.sum(np.square(np.float64(phase1_es) - phase_1)) / (m * n))
            Error_phase.append(phase_error_in)

            # Check for convergence
            if tf.sqrt(tf.reduce_sum(tf.square(tf.abs(grad)))) < tol:
                break

        return phase1_es.numpy(), amp2_es.numpy()

        # Set the initial phase value to facilitate gradient descent optimization


    def gradient_descent(alpha, current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling_):

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

        # Set the initial phase value to facilitate gradient descent optimization


    phase1_est = np.zeros_like(phase1)

    if s == "A":
        iters_ = 500
        Costs = []
        Error_phase = []
        amp2_est = np.zeros_like(amp_2)
        for i in range(iters_):  # execute 200 times Gradient_Descent
            phase1_est, amp2_est = gradient_descent(572, phase1_est, amp_1, amp_2, M, N, z_vec, wavelength, n0,
                                                    sampling)
            cost_current = error_current_Function(amp_2, amp2_est, M, N)
            Costs.append(cost_current)

            phase_error = np.sqrt(np.sum(np.square(np.float64(phase1_est) - phase1)) / (M * N))
            Error_phase.append(phase_error)

    else:
        iters_ = 300
        Costs = []
        Error_phase = []
        phase1_est, amp2_est = conjugate_gradient(phase1, phase1_est, amp_1, amp_2, M, N, z_vec, wavelength, n0,
                                                  sampling, iters_)

    phase1_est = np.float64(phase1_est)
    print(phase1_est)
    print('Final cost = ', error_current_Function(amp_2, amp2_est, M, N))
    print("Running time : ---%s seconds---" % (time.time()-start_time))

    u_1 = amp_1 * np.exp(1j * phase1_est)
    u_0 = propagate_as(u_1, -z_vec[0], wavelength, n0, sampling)
    i0 = np.abs(u_0) ** 2
    ph0_est = np.angle(u_0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(iters_), Costs)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    fig1, axs = plt.subplots(2, 2, figsize=(15, 8))

    # axs 是 2x2 的数组，通过索引访问每个子图
    im1 = axs[0, 0].imshow(amp2_est ** 2)
    fig1.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title("Predicted intensity at z2")

    im2 = axs[0, 1].imshow(amp_2 ** 2)
    fig1.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title("The actual intensity at z2")

    im3 = axs[1, 0].imshow(i0)
    axs[1, 0].set_title("Predicted intensity at z0")
    fig1.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(ph0_est)
    axs[1, 1].set_title("Predicted phase at z0")
    fig1.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()  # 调整布局避免重叠
    plt.show()

