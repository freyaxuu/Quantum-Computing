from qiskit.circuit.library import QFT
from qiskit.circuit import Gate, Parameter
import numpy as np
import cmath
import graycode
from qiskit import ClassicalRegister, QuantumRegister
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np
print(qiskit.__version__)


def find_alphas(target):

    n = int(np.log(len(target))/np.log(2))

    # We first calculate alpha^{y} using
    # Eq(8) of 0407010.
    alphas_y = []

    for j in range(n):
        alpha_j = []

        for mu in range(2**(n-j-1)):

            num = 0
            for l in range(2**j):
                num = num + np.abs(target[(2*mu+1)*(2**(j))+l])**2
            num = np.sqrt(num)

            den = 0
            for l in range(2**(j+1)):
                den = den + np.abs(target[(mu)*(2**(j+1))+l])**2
            den = np.sqrt(den)

            if (den<num):
                raise ValueError("something is not right. Argument of arcsin has to be less than 1.")
            elif den==num:
                ratio = 1
            else:
                ratio = num/den

            alpha_j.append(2*np.arcsin(ratio))

        alphas_y.append(alpha_j)

    # We now calculate the alpha^{z} using
    # Eq(5) of 0407010.
    alphas_z = []
    phases = [cmath.phase(t) for t in target]

    for j in range(n):
        alpha_j = []

        for mu in range(2**(n-j-1)):

            sum = 0
            for l in range(2**j):
                sum = sum + (phases[(2*mu+1)*(2**j)+l]-phases[(2*mu)*(2**j)+l])
            sum = sum/(2**j)

            alpha_j.append(sum)

        alphas_z.append(alpha_j)

    # We now calculate the global phase
    # as defined in Eq(7) of 0407010.

    global_phase = 2*np.mean(phases)

    return alphas_y, alphas_z, global_phase


def find_thetas(alphas):

    # We calculate thetas using
    # Eq(3) of 0407101.

    thetas = []

    for alpha in alphas:

        theta = []

        for i in range(len(alpha)):
            theta_i = 0

            for j in range(len(alpha)):
                theta_i = theta_i + M(i,j)*alpha[j]
            theta_i = theta_i/len(alpha)

            theta.append(theta_i)

        thetas.append(theta)

    return thetas


def M(i,j):

    # This calculates the matrix M
    # defined in Eq(3) of 0407101.
    # However, our definition is different by a factor of 2^{-k}.

    bj = bin(j)[2:]
    bj_rev = bj[::-1]
    gi = bin(graycode.tc_to_gray_code(i))[2:]
    gi_rev = gi[::-1]

    mij = 0
    for x,y in zip(bj_rev,gi_rev):
        mij = mij + int(x)*int(y)

    return (-1)**mij

def cnot_position(r):

    g1 = bin(graycode.tc_to_gray_code(r))[2:]
    g2 = bin(graycode.tc_to_gray_code(r-1))[2:]

    if len(g2)<len(g1):
        g2 = '0' + g2

    g1_rev = g1[::-1]
    g2_rev = g2[::-1]

    for p in range(len(g1)):
        if g1_rev[p] != g2_rev[p]:
            return p+1
        
        
'''
Function to find the optimal parameters to represent target_function for
n ansatz qubits and m ZGR qubits
'''
def zgr_parameters(m,n,target_function):
    
    # m,n are integers
    # target_function is either a python function or a list.
    
    if type(target_function) == list:
        f = target_function
    elif type(target_function) == np.array or type(target_function) == np.ndarray:
        f = list(target_function)
    else:
        # Note since we only need first m
        # Fourier coefficients, we do not need the 
        # grid size of 2^n. Instead, we need a grid
        # of size 2^(2m). This will keep the cost constant
        # if we take n to be large.
        y = np.linspace(0,1,2**(2*m)+1) 
        f = [target_function(yi) for yi in y]
        
    print('f', f)
    fourier_coeffs = np.fft.ifft(f)
    print('FFT',fourier_coeffs)
    fourier_coeffs = fourier_coeffs[:(2**m)]
    fourier_coeffs[0] = fourier_coeffs[0]/np.sqrt(2)
    norm = np.linalg.norm(fourier_coeffs)
    
    
    alpha_y,alpha_z,global_phase = find_alphas(fourier_coeffs/norm)
    theta_y = find_thetas(alpha_y)
    theta_z = find_thetas(alpha_z)
    theta_y = theta_y[::-1]  
    theta_z = theta_z[::-1]
    flat_theta_y =  [th for th_k in theta_y for th in th_k]
    flat_theta_z =  [th for th_k in theta_z for th in th_k]
    params = [np.sqrt(2**(n+1))*norm, *flat_theta_y, *flat_theta_z, global_phase]
    
    return params

def unflatten_params(m,params):
    # input: params = [flatten theta_y, flatten theta_z, global phase]
    
    # return unlfatten_theta_y, unflatten_theta_z, global phase
    
    thetas_y = []
    thetas_y.append([params[0]])

    index = 1
    for i in range(m-1):
        thetas_y.append(params[index:index+2**(i+1)])
        index += 2**(i+1)
    thetas_y = thetas_y[::-1]

    #thetas_y goes into ZGR_Z
    thetas_z = []
    thetas_z.append([params[index]])
    index += 1

    for i in range(m-1):
        thetas_z.append(params[index:index+2**(i+1)])
        index += 2**(i+1)
    thetas_z = thetas_z[::-1]
    
    global_phase = params[-1]
    
    return thetas_y, thetas_z, global_phase


def UCRs(axes, thetas, n):
    
    qr = QuantumRegister(n, name ='qr')
    qc = QuantumCircuit(qr)
    
    
    if axes == 'y':
        rot = qc.ry
    elif axes == 'z':
        rot = qc.rz
    else:
        raise ValueError("Only y or z axes are allowed.")
    
    qr = list(qr)
    qr.reverse()
    
        
    r = len(thetas)-1
    if r>0:
        qc.cx(qr[-1],qr[0])
        
    rot(thetas[r], qr[0])

    rs = [(len(thetas)-2-r) for r in range(len(thetas)-1)]
    for r in rs:
        qc.cx(qr[cnot_position(r+1)], qr[0])
        rot(thetas[r], qr[0])
    ucr_gate = qc.to_gate()
    return ucr_gate




def CAdder(qc, n, q, inverse = False):
    ctrl = q[0]
    a = q[1:n-1]
    w = q[n-1:]

    if inverse == False:
        qc.cx(ctrl, w[n-1])
        qc.ccx(ctrl, w[n-1], w[n-2])
        qc.ccx(ctrl, w[n-1], a[0])
        qc.ccx(w[n-2], a[0], a[1])

        for i in range(n-4):
            qc.cx(a[i+1], w[n-i-3])
            qc.ccx(w[i+1], w[n-i-3], a[i+2])
        
        qc.cx(a[n-3], w[1])
        qc.ccx(a[n-3], w[1], w[0])

        for i in range(n-3):
            qc.ccx(w[i+2], a[n-i-4], a[n-i-3])

        qc.ccx(ctrl, w[n-1], a[0])
    else:
        qc.ccx(ctrl, w[n-1], a[0])
        
        for i in range(n-4, -1, -1):
            qc.ccx(q[i+2], a[n-i-4], a[n-i-3])
        qc.ccx(a[n-3], q[1], q[0])
        qc.cx(a[n-3], q[1])
        
        for i in range(n-5, -1, -1):
            qc.ccx(a[i+1], q[n-i-3], a[i+2])
            qc.cx(a[i+1], q[n-i-3])
            
        qc.ccx(q[n-2], a[0], a[1])
        qc.ccx(ctrl, q[n-1], a[0])
        qc.ccx(ctrl, q[n-1], q[n-2])
        qc.cx(ctrl, q[n-1])
        
        


def ansatz_circuit(n, params):
    """
    Constructs the ansatz circuit.

    Returns:
    QuantumCircuit: The constructed ansatz quantum circuit.
    """

    total_wires = n
    m1 = int(np.log(len(params)+1)/np.log(2))-1
    m2 = total_wires - m1
    thetas_y, thetas_z, phase = unflatten_params(m1,params[1:])
    
    # Initialize the quantum register and circuit
    qr = QuantumRegister(n, name="qr")
    qc = QuantumCircuit(qr)
    
    qubits = [k for k in range(m1)]

    qc.h(0)
    qc.x(0)
    qc.crz(-phase, 0, -1)
    qc.x(0)
    qc.crz(phase, 0, -1)

    # UCRs of type 'y'
    for w in range(m1):
        ws = [m2 + j for j in range(w + 1)]
        ucr1 = UCRs('y', thetas_y[m1 - 1 - w], len(ws))
        qc.append(ucr1, ws)

    qc.x(0)
    
    # UCRs of type 'z' with control
    for w in range(m1):
        ws = [m2 + j for j in range(w + 1)]
        ucr2 = UCRs('z', thetas_z[m1 - 1 - w], len(ws)).control(1)
        qc.append(ucr2, [0] + ws)
    qc.x(0)

    # UCRs of type 'z' with flipped theta values
    for w in range(m1):
        ws = [m2 + j for j in range(w + 1)]
        ucr3 = UCRs('z', [-1 * thz for thz in thetas_z[m1 - 1 - w]], len(ws)).control(1)
        qc.append(ucr3, [0] + ws)

    # "Adder" block
    if m1 == 1:
        qc.cx(0, -1)
        qc.ccx(0, -1, 1)
    elif m1 == 2:
        qc.cx(0, -1)
        qc.ccx(0, -1, -2)
        qc.mcx([0, -1, -2], 1)
    elif m1 > 2 and total_wires > 2 * m1:
        wires_adder = [qr[0], *qr[2:(m1 + 1)], qr[1], *qr[m2:]]
        CAdder(qc, (len(wires_adder) + 1) // 2, wires_adder)
    else:
        wires_adder = [0, 1] + [i for i in range(m2, len(qr))]
        qft_gate = QFT(num_qubits=len(wires_adder) - 1, do_swaps=True, approximation_degree=0).to_gate()
        controlled_qft = qft_gate.control(1)
        qc.append(controlled_qft, [wires_adder[0]] + wires_adder[:0:-1])
        for i in range(m1 + 1):
            phi_i = -(1 / (2 ** i))
            qc.cp(phi_i * np.pi, wires_adder[0], wires_adder[i + 1])
        qft_gate_inverse = QFT(num_qubits=len(wires_adder) - 1, do_swaps=True, inverse=True, approximation_degree=0).to_gate()
        controlled_qft_inverse = qft_gate_inverse.control(1)
        qc.append(controlled_qft_inverse, [wires_adder[0]] + wires_adder[:0:-1])

    # "CNOT" gates
    qc.cx(qr[0], qr[1])
    for i in range(m1):
        qc.cx(0, m2 + i)

    # "Toffoli" gates
    for i in range(2, m2):
        qc.ccx(0, 1, i)

    # Control Hadamard and MCX gates
    qc.ry(np.pi / 4, 0)
    for i in range(m1):
        qc.x(m2 + i)
    control_qubits = qr[m2:]
    target_qubit = qr[0]
    qc.mcx(control_qubits, target_qubit)
    for i in range(m1):
        qc.x(qr[m2 + i])
    qc.ry(-np.pi / 4, qr[0])

    # inverse QFT
    qc.append(QFT(num_qubits=n, do_swaps=True, inverse=True, approximation_degree=0), range(n - 1, -1, -1))

    # Return the constructed circuit
    return qc


def simulate_f(n, params):
    qc = ansatz_circuit(n, params)
    sim = Aer.get_backend('statevector_simulator')
    result_qiskit = sim.run(transpile(qc, sim)).result()
    output_state = result_qiskit.get_statevector(qc)
    simulated_f = np.real(output_state)*params[0]
    
    return simulated_f




# usage
# set n, m, f
# params = zgr_parameters(m, n, f) 
# simulated_function = simulate_f(n, params)