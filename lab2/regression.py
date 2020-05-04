import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    m_v = np.zeros(2)
    print("m_v shape: " ,m_v.shape)
    c_v = np.array( [ [ beta , 0 ] , [ 0 , beta ] ] )
    print("c_v shape: ",c_v.shape)
    x_s = []
    for i in np.linspace(-1 , 1 , 150):
        for j in np.linspace(-1 , 1 , 150):
            x_s.append([i,j])
    x_s = np.array(x_s)
    print("x_s shape: ",x_s.shape)
    density = util.density_Gaussian(m_v , c_v , x_s)
    #print(density)
    print("length density ",len(density))
    X,Y = np.meshgrid( np.linspace(-1,1,150) , np.linspace(-1,1,150) )
    plt.contour( X , Y , np.reshape(density , (150, 150 )) )
    plt.plot(-0.1 , -0.5 , marker = 'o' , MarkerSize = 10 , label = 'point a')
    plt.xlabel('a0 ')
    plt.ylabel(' a1 ')
    plt.legend()
    plt.title('p(a)')
    plt.show()    
    print('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x')
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
    mu = 0
    Cov = 0

    x_s = []
    for i in np.linspace(-1 , 1 , 150):
        for j in np.linspace(-1 , 1 , 150):
            x_s.append([i,j])
    x_s = np.array(x_s)

    X = []
    for i in x:
        j = [1,i[0]]
        X.append(j)
    X = np.array(X)

    common = np.matmul( X.T , X) + np.identity(2) * sigma2/beta
    common = np.linalg.inv(common)
    Cov = common * sigma2
    mu = np.matmul(common , np.matmul (X.T , z) )
    mu = mu.flatten()
    print("X.shape: " , X.shape)
    print("z.shape: ",z.shape)
    print("Cov.shape" , Cov.shape)
    print("mu.shape: ",mu.shape)
    density = util.density_Gaussian(mu , Cov , x_s).reshape(150 , 150 ).T
    print("density.shape",density.shape)
    X,Y = np.meshgrid( np.linspace(-1,1,150) , np.linspace(-1,1,150) )

   

    plt.contour( X , Y , np.reshape(density , (150, 150 )))
    plt.plot(-0.1 , -0.5 , marker = 'o' , MarkerSize = 10 , label = 'point a')
    plt.xlabel('a0 ')
    plt.ylabel(' a1 ')
    plt.legend()
    plt.xlim = (-1,1)
    plt.ylim = (-1,1)
    plt.title('p(a|x1,z1....xn,zn) for '+ str(len(x)) +' samples')
    plt.show() 
    print('-x-x-x-x-x-x-x-x-x')

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    X = []
    for i in x:
        j = [1,i]
        X.append(j)
    X = np.array(X)
    print("X.shape", X.shape)
    print("mu.sape", mu.shape)
    print("Cov.shape ",Cov.shape)
    mu_new = np.matmul(X , mu)
    cov_new = sigma2 + np.matmul( X, np.matmul( Cov,X.T ) )
    var = np.sqrt(cov_new.diagonal())
    
    plt.figure(1)
    plt.xlabel("X VALUES")
    plt.ylabel("Z VALUES")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title("Prediction with " + str(len(x_train)) + " samples")
    plt.errorbar(x , mu_new , var , label='predicted values')
    plt.scatter(x_train , z_train, color='r', label="Samples")
    plt.legend()
    plt.show()

    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 100
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

   

    
    
    

    
