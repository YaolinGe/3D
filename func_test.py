#%% learn tensor
import tensorflow as tf
import numpy as np

#
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

rank_1_tensor = tf.constant([2, 3, 4])
print(rank_1_tensor)

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype = tf.float16)
print(rank_2_tensor)

rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]],])
print(rank_3_tensor)






#%%
print("hello world")
import numpy as np
import matplotlib.pyplot as plt

a = 2
x = np.arange(-10, 10, 0.1)
y = np.sqrt( a ** 2 * (x ** 2 - a ** 2) / x ** 2 )

plt.plot(x, y, 'k-')
plt.show()
#%% generate gaussian random fields

# Main dependencies
import numpy
import scipy.fftpack


def fftind(size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components

        Example:

            print(fftind(5))

            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]
            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]

        """
    k_ind = numpy.mgrid[:size, :size] - int((size + 1) / 2)
    k_ind = scipy.fftpack.fftshift(k_ind)
    return (k_ind)


def gaussian_random_field(alpha=3.0,
                          size=128,
                          flag_normalize=True):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.

        Input args:
            alpha (double, default = 3.0):
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0
        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field

        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """

    # Defines momentum indices
    k_idx = fftind(size)

    # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = numpy.power(k_idx[0] ** 2 + k_idx[1] ** 2 + 1e-10, -alpha / 4.0)
    amplitude[0, 0] = 0

    # Draws a complex gaussian random noise with normal
    # (circular) distribution
    noise = numpy.random.normal(size=(size, size)) \
            + 1j * numpy.random.normal(size=(size, size))

    # To real space
    gfield = numpy.fft.ifft2(noise * amplitude).real

    # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - numpy.mean(gfield)
        gfield = gfield / numpy.std(gfield)

    return gfield


def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    plt.imshow(example, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()

#%% use module to produce gaussian random noise
import gaussian_random_fields as gr
import numpy
import matplotlib
import matplotlib.pyplot as plt

example = gr.gaussian_random_field()
plt.imshow(example, cmap='gray')
plt.show()
print('mean: ', numpy.mean(gfield))
print('std: ', numpy.std(gfield))
plt.plot(example[:,0])
plt.plot(example[0,:])
plt.title('Profiles')

for alpha in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:

    example = gr.gaussian_random_field(alpha = alpha, size = 512)
    my_dpi = 128
    plt.figure(
        figsize = (example.shape[1]/my_dpi, example.shape[0]/my_dpi),
        dpi = my_dpi)
    plt.axis('off')
    plt.title('alpha='+str(alpha))
    plt.imshow(example, interpolation='none', cmap='gray')

#%% plot 3d grid and volume visualisation
# print(np.mgrid[-1:2:10j])  #np.mgrid used to generate vertical meshgrid

print("hello world started")

import plotly.graph_objects as go
import numpy as np
X, Y, Z = np.mgrid[-3:3:3j, 10:20:4j, 8:14:4j]
values = np.sin(X * Y * Z) / (X * Y * Z)

fig = go.Figure(data = go.Volume(
    x = X.flatten(),
    y = Y.flatten(),
    z = Z.flatten(),
    value = values.flatten(),
    isomin = .1,
    isomax = .8,
    opacity = 0.1, # need to be small to see all those surfaces
    surface_count = 17, # needs to be large to render better
))
fig.show()
x = X.flatten()
y = Y.flatten()
z = Z.flatten()
value = values.flatten()

print(X)
print(Y)
print(Z)
print("hello world finished")

#%% small test on flatten
x = np.random.rand(4,3,2)
print(x)
xx = x.flatten() # flatten will go from row by row
print(xx)



#%% test of flip
x = np.arange(0, 10).reshape(2, 5)

t = np.flipud(x)
print(x)
print(t)

def f(X, Y, Z):
    return (X - 10) ** 2 + (Y - 10) ** 2 + Z ** 2

values = f(X, Y, Z)
H_eu = H(X, Y)
H_ver = np.sqrt((X - Y) ** 2)

Cov = Matern_32(H_eu)
for i in range(Cov.shape[0]):
    print(i)
    print(np.linalg.cholesky(Cov[i, :, :]).shape) # find its lower part
    test = np.linalg.cholesky(Cov[i, :, :])
    print(test)
    # L[i] = test
    # samples = np.dot(L.T, np.random.randn(num_total).reshape([num_total, 1]))

print(np.all(H_ver == H_eu))
print('hello world ned')

