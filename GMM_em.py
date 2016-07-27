# encoding: shift_jis
import numpy
import random
import math
import pylab
import matplotlib
import matplotlib.pyplot as plt

# �O���t�ɕ`��
def draw_data( data, classes, mus, sigmas, colors = ("r" , "b" , "g" , "c" , "m" , "y" , "k" )):
    #pylab.clf()
    # �f�[�^��`��
    for d,k in zip(data, classes):
        pylab.scatter( d[0] , d[1] , color=colors[k] )

    # ���ς�`��
    for i,m in enumerate(mus):
        pylab.scatter( m[0] , m[1] , color=colors[i] , marker="x" )

    # ���U��\��
    def to_transform(mu, sigma):
        val, vec = numpy.linalg.eigh(sigma)
        trans = numpy.diag(numpy.sqrt(val)).dot(vec)
        return matplotlib.transforms.Affine2D.from_values(*trans.flatten(), e=mu[0], f=mu[1])

    ax = pylab.gca()
    circles = [pylab.Circle((0, 0), radius=1, transform=to_transform(mu, sigma)) for mu, sigma in zip(mus, sigmas)]
    ax.add_collection( matplotlib.collections.PatchCollection(circles, alpha=0.2))

def draw_line( p1 , p2 , color="k" ):
    pylab.plot( [p1[0], p2[0]] , [p1[1],p2[1]], color=color )

def calc_gauss_pdf(x, mu, sigma):

    # ���K����
    C = 1 / (math.pow(2 * math.pi, len(x)/2.0) * math.pow(numpy.linalg.det(sigma), 1.0/2.0))

    # x - mu
    x_mu = numpy.matrix(x-mu)

    # exp�̍�
    P = numpy.exp(-0.5 * x_mu * numpy.linalg.inv(sigma) * x_mu.T)
    return float(C * P)

def E_step( data, mus, sigmas, alphas ):
    K = len(mus)
    P = numpy.zeros( (len(data), K) )

    # �e�N���X�ɓ���m�����v�Z
    for d in range(len(data)):
        for k in range(K):
            P[d][k] = alphas[k] * calc_gauss_pdf( data[d] , mus[k] , sigmas[k] )
        P[d] = P[d] / numpy.sum(P[d])

    return P

def M_step( data, P ):
    K = len(P[0])
    N = P.sum(0)
    dim = len(data[0])

    mus = [ numpy.zeros(dim) for k in range(K) ]
    sigmas = [ numpy.zeros((dim,dim)) for k in range(K) ]

    # ���ς��v�Z
    for k in range(K):
        for d in range(len(data)):
            mus[k] += P[d,k] * data[d]
        mus[k] /= N[k]

    # ���U���v�Z
    for k in range(K):
        for d in range(len(data)):
            sigmas[k] += P[d,k] * (data[d]-mus[k]).reshape(dim,1) * (data[d]-mus[k]).reshape(1,dim)
        sigmas[k] /= N[k]

    # ��������v�Z
    alphas = N / N.sum()

    return mus, sigmas, alphas

def calc_logliklihood( data, mus, sigmas, alphas ):
    lik = 0
    for d in range(len(data)):
        sum = 0
        for k in range(len(mus)):
            sum += alphas[k] + calc_gauss_pdf( data[d] , mus[k] , sigmas[k] )
        lik += numpy.log( sum )

    return lik




# gmm���C��
def gmm( data, K, num_trial=1 ):
    pylab.ion()
    # �f�[�^�̎�����
    dim = len(data[0])

    # �ޓx�̍ő�l
    max_lik = -99999

    # �K�E�X���z�̏����l������
    mus = [ None for k in range(K) ]
    sigmas = [ None for k in range(K) ]
    alphas = [ 1.0/K for k in range(K) ]

    for i in range(num_trial):
        print "----------- %d��� -----------" % (i)

        # ������
        for k in range(K):
            mus[k] = data[random.randint(0,len(data)-1)]  # �K���ȃf�[�^�𕽋ςɂ���
            sigmas[k] = numpy.eye(dim)                    # ���U���K��

        # �����l�Ńf�[�^�𕪗�
        P = E_step( data , mus, sigmas, alphas )
        classes = numpy.argmax( P , 1 )

        # �O���t�\��
        draw_data( data, classes, mus, sigmas )
        pylab.draw()
        pylab.pause(1.0)

        lik = 0
        liks = []
        for j in range(20):

            # EM�A���S���Y��
            P = E_step( data , mus, sigmas, alphas )
            mus, sigmas, alphas = M_step( data , P )

            # �N���X�^�����O��������
            classes = numpy.argmax( P , 1 )

            # �吔�ޓx�v�Z
            lik = calc_logliklihood( data, mus, sigmas, alphas )
            liks.append( lik )

            # �ΐ��ޓx�̌v�Z
            print "%02dstep\t�ΐ��ޓx =" % (j),lik

            # �O���t�\��
            pylab.clf()
            pylab.subplot("121")
            pylab.title("result")
            draw_data( data, classes, mus, sigmas )
            pylab.subplot("122")
            pylab.title( "liklihood" )
            pylab.plot( range(len(liks)) , liks )
            pylab.draw()
            pylab.pause(1.0)


        # �ŏI�I�Ȍ��ʂ�\��
        if lik>max_lik:
            pylab.ioff()
            pylab.clf()
            pylab.subplot("121")
            pylab.title("result")
            draw_data( data, classes, mus, sigmas )
            for d,k in zip(data,classes):
                draw_line( d , mus[k] )
            pylab.subplot("122")
            pylab.title( "liklihood" )
            pylab.plot( range(len(liks)) , liks )
            pylab.show()

            # �ޓx�̍ő�l�X�V
            max_lik = lik



def main():
    data = numpy.loadtxt( "data2.txt" )
    gmm( data , 2, 10 )

if __name__ == '__main__':
    main()