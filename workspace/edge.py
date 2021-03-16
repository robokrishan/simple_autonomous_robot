import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    kernel_vector = kernel.reshape(Hk*Wk, 1)
    vector_matrix = np.zeros((Hi*Wi, Hk*Wk))
    for i in range(Hi):
        for j in range(Wi):
            vector_matrix[i*Wi + j,:] = padded[i:i+Hk, j:j+Wk].reshape(1, Hk*Wk)
            
            
    result = np.dot(vector_matrix, kernel_vector)
    out = result.reshape(Hi, Wi)
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = int(size / 2)

    for i in range(size):
        for j in range(size):
            kernel[i, j] = 1 / (2 * np.pi * np.square(sigma)) * np.exp(-1 * (np.square(i - k) + np.square(j - k)) / (2 * np.square(sigma)))
            # This is the implementation of the provided mathematical formula in the question
    ### END YOUR CODE

    return kernel

def robert_partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, 0], [0 ,1]])      # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def robert_partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, -1], [1, 0]])      # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def sobel_partial_x(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])      # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def sobel_partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])      # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def prewitt_partial_x(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def prewitt_partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_e(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_w(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_ne(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_sw(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_n(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[5, 5, 5], [-3, 0, 5], [-3, -3, -3]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_s(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_nw(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kirsch_partial_se(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-3, -3, 5], [-3, 0, 5], [-3, 5, 5]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kiresh_partial_x(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kiresh_partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array(([1, 1, 1], [0, 0, 0], [-1, -1, -1]))     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kiresh_partial_diag1(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def kiresh_partial_diag2(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def log_partial_x(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def log_partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])     # Partial derivative matrix from slides
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def robert_gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = robert_partial_x(img)
    Gy = robert_partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def sobel_gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = sobel_partial_x(img)
    Gy = sobel_partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def prewitt_gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = prewitt_partial_x(img)
    Gy = prewitt_partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kirsch_gradient_ew(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kirsch_partial_e(img)
    Gy = kirsch_partial_w(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kirsch_gradient_nesw(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kirsch_partial_ne(img)
    Gy = kirsch_partial_sw(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kirsch_gradient_ns(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kirsch_partial_n(img)
    Gy = kirsch_partial_s(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kirsch_gradient_nwse(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kirsch_partial_nw(img)
    Gy = kirsch_partial_se(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kiresh_gradient_xy(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kiresh_partial_x(img)
    Gy = kiresh_partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def kiresh_gradient_diag(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = kiresh_partial_diag1(img)
    Gy = kiresh_partial_diag2(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def log_gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = log_partial_x(img)
    Gy = log_partial_y(img)
    G = np.sqrt(np.square(Gx) + np.square(Gy))
    # Implementing the gradient formula from the question
    
    theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = theta % 360
    # The angle formula implementation above
    
    ### END YOUR CODE

    return G, theta

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)
    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    for i in range(H):
        for j in range(W):
            if (theta[i,j]%180==0):
                cmp1=G[i,j+1]if(j+1<W)else 0;
                cmp2=G[i,j-1]if(j-1>-1)else 0;
            elif(theta[i,j]%180==45):
                cmp1=G[i-1,j+1]if((i-1>-1)and(j+1<W))else 0;
                cmp2=G[i+1,j-1]if((i+1<H)and(j-1>-1))else 0;
            elif(theta[i,j]%180==90):
                cmp1=G[i-1,j]if(i-1>-1)else 0;
                cmp2=G[i+1,j]if(i+1<H)else 0;
            elif(theta[i,j]%180==135):
                cmp1=G[i-1,j-1]if((j-1>-1)and(i-1>-1))else 0;
                cmp2=G[i+1,j+1]if((i+1<H)and(j+1<W))else 0;

            if ((G[i,j]>=cmp1)and(G[i,j]>=cmp2)):
                out[i,j]=G[i,j];
            else:
                out[i,j]=0;
    ### END YOUR CODE

    return out


'''
def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    l = np.zeros((3,3))
    l[1,1] = 1
    
    if theta[0,0] == 0.0:
        l[1, 0] = 1
        l[1,2] = 1
    elif theta[0,0] == 45.0:
        l[0,2] = 1
        l[2,0] = 1
    elif theta[0,0] == 90.0:
        l[0,1] = 1
        l[2,1] = 1
    elif theta[0,0] == 135.0:
        l[0,0] = 1
        l[2,2] = 1
    else:
        print("Gradient Value Error!")
    # Above we have the different possible cases for the angle values
    
    img_big = np.zeros((H + 2, W + 2))
    img_big[1:H + 1, 1:W + 1] = G
    
    for i in range(H):
        for j in range(W):
            l_y = i + 1
            l_x = j + 1
            imageArea = img_big[l_y - 1:l_y+2, l_x - 1:l_x + 2]
            if imageArea[1, 1] != np.max(np.multiply(imageArea, l)):
                out[i, j] = 0
            # Suppression!
            else:
                out[i, j] = imageArea[1, 1]
    ### END YOUR CODE

    return out
'''
def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if img[j, i] >= high :
                strong_edges[j, i] = True
                weak_edges[j, i] = False
            elif low <= img[j, i] < high:
                strong_edges[j, i] = False
                weak_edges[j, i] = True
            else:
                strong_edges[j, i] = False
                weak_edges[j, i] = False
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    edges = np.copy(strong_edges)
    for i in range(H):
        for j in range(W):
            neighbors = get_neighbors(i, j, H, W)        # We make use of the given function get_neighbors
            if weak_edges[i, j] and np.any(edges[r, c] for r, c in neighbors):
                edges[i, j] = True
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    r = xs.reshape(-1, 1) * cos_t.reshape(1, -1) + ys.reshape(-1, 1) * sin_t.reshape(1, -1)
    r = (r.reshape(-1) + diag_len).astype(int)
    np.add.at(accumulator, (r, np.tile(np.arange(len(thetas)), len(xs))), 1)
    ### END YOUR CODE

    return accumulator, rhos, thetas
