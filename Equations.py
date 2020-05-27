"""
Every function below must accept a numpy array inputs of shape (no_samples, no_vars) and return a target array of shape (no_samples, )
It is highly recommended to use vector operations instead of looping through rows as numpy is faster with vectorization
Use the get_column function for this easily
----------------------------------------------
-----------------------------------------------
At the bottom is a dictionary that must be updated in order for the equations to work
"""

import numpy as np

e = np.e
pi = np.pi
sqrt = np.sqrt
exp = np.exp
sin = np.sin
cos = np.cos
tan = np.tan
arcsin = np.arcsin
ln = np.log

def get_column(inputs, col):
    try:
        return inputs[:, col]
    except:
        print(f"inputs had shape {inputs.shape} but was asked for column {col}")


def I62a(inputs):
    return exp(-get_column(inputs, 0)**2/2)/sqrt(2*pi)

def I62(inputs):
    return exp(-(get_column(inputs, 0)/get_column(inputs, 1))**2/2)/(sqrt(2*pi)*get_column(inputs, 1))

def I62b(inputs):
    theta = get_column(inputs, 0)
    theta1 = get_column(inputs, 1)
    sigma = get_column(inputs, 2)
    return exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)


def I814(inputs):
    return np.sqrt((get_column(inputs, 1)-get_column(inputs, 0))**2+(get_column(inputs, 3)-get_column(inputs, 2))**2)


def I918(inputs):
    G =  6.67259 * 10**(-11)
    m1 = get_column(inputs, 0)
    m2 = get_column(inputs, 1)
    x1 = get_column(inputs, 2)
    x2 = get_column(inputs, 3)
    y1 = get_column(inputs, 4)
    y2 = get_column(inputs, 5)
    z1 = get_column(inputs, 6)
    z2 = get_column(inputs, 7)
    return G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)

def I107(inputs):
    m_0 = get_column(inputs, 0)
    v = get_column(inputs, 1)
    c = get_column(inputs, 2)
    return m_0/sqrt(1-v**2/c**2)

def I1119(inputs):
    x1 = get_column(inputs, 0)
    x2 = get_column(inputs, 1)
    x3 = get_column(inputs, 2)
    y1 = get_column(inputs, 3)
    y2 = get_column(inputs, 4)
    y3 = get_column(inputs, 5)
    return x1*y1+x2*y2+x3*y3

def I121(inputs):
    return get_column(inputs, 0)*get_column(inputs, 1)

def I122(inputs):
    q1 = get_column(inputs, 0)
    q2 = get_column(inputs, 1)
    r = get_column(inputs, 2)
    epsilon = get_column(inputs, 3)
    return q1*q2*r/(4*pi*epsilon*r**3)

def I124(inputs):
    q1 = get_column(inputs, 0)
    r = get_column(inputs, 1)
    epsilon = get_column(inputs, 2)
    return q1*r/(4*pi*epsilon*r**3)


def I125(inputs):
    q2 = get_column(inputs, 0)
    Ef = get_column(inputs, 1)
    return q2*Ef


def I1211(inputs):
    q = get_column(inputs, 0)
    Ef = get_column(inputs, 1)
    B = get_column(inputs, 2)
    v = get_column(inputs, 3)
    theta = get_column(inputs, 4)
    return q*(Ef+B*v*sin(theta))


def I134(inputs):
    m = get_column(inputs, 0)
    v = get_column(inputs, 1)
    u = get_column(inputs, 2)
    w = get_column(inputs, 3)
    return 1/2*m*(v**2+u**2+w**2)



def I1312(inputs):
    G = 6.67259 * 10**(-11)
    m1 = get_column(inputs, 0)
    m2 = get_column(inputs, 1)
    r2 = get_column(inputs, 3)
    r1 = get_column(inputs, 2)
    return G*m1*m2*(1/r2-1/r1)

def I143(inputs):
    m = get_column(inputs, 0)
    g = get_column(inputs, 1)
    z = get_column(inputs, 2)
    return m*g*z


def I144(inputs):
    k_spring = get_column(inputs, 0)
    x = get_column(inputs, 1)
    return 1/2*k_spring*x**2



def I153x(inputs):
    x = get_column(inputs, 0)
    u = get_column(inputs, 1)
    t = get_column(inputs, 2)
    c = get_column(inputs, 3)
    return (x-u*t)/sqrt(1-u**2/c**2)


def I153t(inputs):
    x = get_column(inputs, 0)
    u = get_column(inputs, 1)
    t = get_column(inputs, 2)
    c = get_column(inputs, 3)
    return (t-u*x/c**2)/sqrt(1-u**2/c**2)


def I151(inputs):
    m_0 = get_column(inputs, 0)
    v = get_column(inputs, 1)
    c = get_column(inputs, 2)
    return m_0*v/sqrt(1-v**2/c**2)


def I166(inputs):
    u = get_column(inputs, 0)
    v = get_column(inputs, 1)
    c = get_column(inputs, 2)
    return (u+v)/(1+u*v/c**2)


def I184(inputs):
    m1 = get_column(inputs, 0)
    m2 = get_column(inputs, 1)
    r1 = get_column(inputs, 2)
    r2 = get_column(inputs, 3)
    return (m1*r1+m2*r2)/(m1+m2)


def I1812(inputs):
    r = get_column(inputs, 0)
    F = get_column(inputs, 1)
    theta = get_column(inputs, 2)
    return r*F*sin(theta)


def I1814(inputs):
    m = get_column(inputs, 0)
    r = get_column(inputs, 1)
    v = get_column(inputs, 2)
    theta = get_column(inputs, 3)
    return m*r*v*sin(theta)


def I246(inputs):
    m = get_column(inputs, 0)
    omega = get_column(inputs, 1)
    omega_0 = get_column(inputs, 2)
    x = get_column(inputs, 3)
    return 1/2*m*(omega**2+omega_0**2)*1/2*x**2


def I2513(inputs):
    q = get_column(inputs, 0)
    C = get_column(inputs, 1)
    return q/C


def I262(inputs):
    # Can only take inputs within a certain range
    n = get_column(inputs, 0)
    theta2 = get_column(inputs, 1)
    return arcsin(n*sin(theta2))


def I276(inputs):
    d1 = get_column(inputs, 0)
    d2 = get_column(inputs, 1)
    n = get_column(inputs, 2)
    return 1/(1/d1+n/d2)


def I294(inputs):
    omega = get_column(inputs, 0)
    c = get_column(inputs, 0)
    return omega/c


def I2916(inputs):
    x1 = get_column(inputs, 0)
    x2 = get_column(inputs, 1)
    theta1 = get_column(inputs, 2)
    theta2 = get_column(inputs, 3)
    return sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))


def I303(inputs):
    Int_0 = get_column(inputs, 0)
    n = get_column(inputs, 1)
    theta = get_column(inputs, 2)
    return Int_0*sin(n*theta/2)**2/sin(theta/2)**2


def I305(inputs):
    lambd = get_column(inputs, 0)
    n = get_column(inputs, 1)
    d = get_column(inputs, 2)
    return arcsin(lambd/(n*d))


def I325(inputs):
    q = get_column(inputs, 0)
    a = get_column(inputs, 1)
    epsilon = get_column(inputs, 2)
    c = get_column(inputs, 3)
    return q**2*a**2/(6*pi*epsilon*c**3)

def I3217(inputs):
    epsilon = get_column(inputs, 0)
    c = get_column(inputs, 1)
    Ef = get_column(inputs, 2)
    r = get_column(inputs, 3)
    omega = get_column(inputs, 4)
    omega_0 = get_column(inputs, 5)
    return (1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)


def I348(inputs):
    q = get_column(inputs, 0)
    v = get_column(inputs, 1)
    B = get_column(inputs, 2)
    p = get_column(inputs, 3)
    return q*v*B/p


def I341(inputs):
    omega_0 = get_column(inputs, 0)
    v = get_column(inputs, 1)
    c = get_column(inputs, 2)
    return omega_0/(1-v/c)


def I3414(inputs):
    v = get_column(inputs, 0)
    c = get_column(inputs, 1)
    omega_0 = get_column(inputs, 2)
    return (1+v/c)/sqrt(1-v**2/c**2)*omega_0


def I3427(inputs):
    h = get_column(inputs, 0)
    omega = get_column(inputs, 1)
    return (h/(2*pi))*omega


def I374(inputs):
    I1 = get_column(inputs, 0)
    l1 = I1
    I2 = get_column(inputs, 1)
    l2 = I2
    delta = get_column(inputs, 2)
    return I1+I2+2*sqrt(I1*I2)*cos(delta)


def I3812(inputs):
    epsilon = get_column(inputs, 0)
    h = get_column(inputs, 1)
    m = get_column(inputs, 2)
    q = get_column(inputs, 3)
    return 4*pi*epsilon*(h/(2*pi))**2/(m*q**2)


def I391(inputs):
    pr = get_column(inputs, 0)
    V = get_column(inputs, 1)
    return 3/2*pr*V


def I3911(inputs):
    gamma = get_column(inputs, 0)
    pr = get_column(inputs, 1)
    V= get_column(inputs, 2)
    return 1/(gamma-1)*pr*V


def I3922(inputs):
    n = get_column(inputs, 0)
    kb = get_column(inputs, 1)
    T = get_column(inputs, 2)
    V = get_column(inputs, 3)
    return n*kb*T/V


def I401(inputs):
    n_0 = get_column(inputs, 0)
    m = get_column(inputs, 1)
    g = get_column(inputs, 2)
    x = get_column(inputs, 3)
    T = get_column(inputs, 4)
    kb = 1.38*(10**-23)
    return n_0*exp(-m*g*x/(kb*T))


def I4116(inputs):
    h = get_column(inputs, 0)
    omega = get_column(inputs, 1)
    c = get_column(inputs, 2)
    kb = get_column(inputs, 3)
    T = get_column(inputs, 4)
    return h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))


def I4316(inputs):
    mu_drift = get_column(inputs, 0)
    q = get_column(inputs, 1)
    Volt = get_column(inputs, 2)
    d = get_column(inputs, 3)
    return mu_drift*q*Volt/d


def I4331(inputs):
    mob= get_column(inputs, 0)
    kb = get_column(inputs, 1)
    T = get_column(inputs, 2)
    return mob*kb*T


def I4343(inputs):
    gamma = get_column(inputs, 0)
    kb = get_column(inputs, 1)
    v = get_column(inputs, 2)
    A = get_column(inputs, 3)
    return 1/(gamma-1)*kb*v/A


def I444(inputs):
    n = get_column(inputs, 0)
    kb = get_column(inputs, 1)
    T = get_column(inputs, 2)
    V2 = get_column(inputs, 3)
    V1 = get_column(inputs, 4)
    return n*kb*T*ln(V2/V1)


def I4723(inputs):
    gamma = get_column(inputs, 0)
    pr = get_column(inputs, 1)
    rho = get_column(inputs, 2)
    return sqrt(gamma*pr/rho)


def I482(inputs):
    m = get_column(inputs, 0)
    c = get_column(inputs, 1)
    v = get_column(inputs, 2)
    return m*c**2/sqrt(1-v**2/c**2)

######################################################################################################################
# Self Defined Equations
######################################################################################################################
def sine(inputs):
    x = get_column(inputs, 0)
    return sin(x)



equation_dict = {
    "sine"   : sine,
    "I.6.2a" : I62a,
    "I.6.2"  : I62,
    "I.6.2b" : I62b,
    "I.8.14" : I814,
    "I.9.18" : I918,
    "I.10.7" : I107,
    "I.11.19": I1119,
    "I.12.1" : I121,
    "I.12.2" : I122,
    "I.12.4" : I124,
    "I.12.5" : I125,
    "I.12.11": I1211,
    "I.13.4" : I134,
    "I.13.12":I1312,
    "I.14.3" : I143,
    "I.14.4" : I144,
    "I.15.3x": I153x,
    "I.15.3t": I153t,
    "I.15.1" : I151,
    "I.16.6" : I166,
    "I.18.4" : I184,
    "I.18.12": I1812,
    "I.18.14": I1814,
    "I.24.6" : I246,
    "I.25.13": I2513,
    "I.26.2" : I262,
    "I.27.6" : I276,
    "I.29.4" : I294,
    "I.29.16": I2916,
    "I.30.3" : I303,
    "I.30.5" : I305,
    "I.32.5" : I325,
    "I.32.17": I3217,
    "I.34.8" : I348,
    "I.34.1" : I341,
    "I.34.14": I3414,
    "I.34.27": I3427,
    "I.37.4" : I374,
    "I.38.12": I3812,
    "I.39.1" : I391,
    "I.39.11": I3911,
    "I.40.1" : I401,
    "I.41.16": I4116,
    "I.43.16": I4316,
    "I.43.31": I4331,
    "I.43.43": I4343,
    "I.44.4" : I444,
    "I.47.23": I4723,
    "I.48.2" : I482,
    }




















