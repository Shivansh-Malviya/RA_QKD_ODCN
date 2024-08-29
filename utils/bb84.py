'''
File containing the functions and libraries required for performing the bb84 protocol
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sys import getsizeof
from scipy.optimize import curve_fit
from contextlib import contextmanager
import sys

from qiskit import *
from qiskit.qasm2 import dumps
from qiskit_aer import Aer

import base64
import random

# import operation as op
from functools import reduce

####################################################################### In bb84_reservoir #######################################################################

### Imports :

# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import os
# from sys import getsizeof
# from scipy.optimize import curve_fit
# from contextlib import contextmanager
# import sys


### Functions used : 

# Define the hyperbolic function
def hyperbolic_fit(x, a, b):
    return a / x + b


#from sys import getsizeof
def Size(var):
    print(f" Sys size : {getsizeof(var)}", end = ", ".rjust(8 - len(f"{getsizeof(var)}")))
    
    try:
        print(f" np size : {var.nbytes}", end = " ")
    except:
        print(" np size : NA", end = " ")


def test(vars, labels):
    '''
    Size and Type : Prints the label of the variable and the corresponding size(with overhead) and the numpy size(if applicable). Also shows the 
    datatype of the variable. All in a justified manner.

    Data Examples : Prints the first 10 elements of the very first dimension of an array/list (e.g. in case of a 3-D array, will print the 
    array[0, 0, :10] element). If the array is only 1-D, will print the first 10 elements. If it's a single variable, the value will be printed.
    Next to each array example, '*' will be printed. The number of '*' printed corresponding to an array shows its dimensions.
    '''
    max_len = len(max(labels, key = len))

    print("\nSize and Type :\n")
    for item, label in zip(vars, labels):
        print(f"{label} {':'.rjust(max_len + 2 - len(label))} ", end = " ") 
        Size(item), print("    ", type(item), end = " "), print("")

    print("\n\nData Examples :\n ")
    for item, label in zip(vars, labels):
        print(f"{label} {':'.rjust(max_len + 2 - len(label))} ", end = " ") 
        
        try :
            try :
                print(item[0, :10], "**")
            except :
                print(item[:10], "*")
        
        except :
            print(item)    


######################################################################## In Qiskit_rebuilt_4 ################################################################


# from qiskit import *
# from qiskit.qasm2 import dumps
# from qiskit_aer import Aer

# import base64
# import numpy as np
# import random


### Functions used

# from qiskit import *
# from qiskit.qasm2 import dumps
# import random
def NoisyChannel(qc1, qc2, qc1_name, errors, noise = 5e-4):
    ''' This function takes the output of a circuit qc1 (made up only of x and 
        h gates), simulates a noisy quantum channel where Pauli errors (X - bit flip; Z - phase flip)
        will occur in qc2, and then initializes another circuit qc2 with the introduced noise.
    ''' 
    
    # Retrieve quantum state from qasm code of qc1
    qs = [dumps(qc1[i]).split('\n') for i in range(len(qc1))]
    
    # Process the code to get the instructions
    parsed_instructions = []
    for i, qasm_code in enumerate(qs):
        for line in qasm_code:
            line = line.strip()    # removing leading/trailing whitespace
            if line.startswith(('x', 'h', 'measure')):
                line = line.replace('0', str(i))
                parsed_instructions.append(line)
    
    # Apply parsed instructions to qc2
    for instruction in parsed_instructions:
        if instruction.startswith('x'):
            old_qr = int(instruction.split()[1][2:-2])
            qc2[old_qr].x(0)
            
        elif instruction.startswith('h'):
            old_qr = int(instruction.split()[1][2:-2])
            qc2[old_qr].h(0)
        
        elif instruction.startswith('measure'):
            continue    # exclude measuring
            
        else:
            print(f"Unable to parse instruction: {instruction}")
            raise Exception('Unable to parse instruction')
    
    # Introducing noise (taking input)
    for instruction in parsed_instructions:
        if random.random() < noise:
            old_qr = int(instruction.split()[1][2:-2])
            qc2[old_qr].x(0)     # Apply bit-flip error
            errors[0] += 1
            
        if random.random() < noise:
            old_qr = int(instruction.split()[1][2:-2])
            qc2[old_qr].z(0)     # Apply phase-flip error
            errors[1] += 1

    return errors


# import random
def generate_random_bits(num):
    """This function generates a random array of bits(0/1) of size = num"""
    # bits = np.array([random.randint(0, 1) for _ in range(num)])    # Randomly fills the array with 0/1

    bit_string = ""
    for _ in range(num):
        rand_bit = random.randint(0, 1)     # Flip Coin
        bit_string += str(rand_bit)
        
    return bit_string


# import random
def generate_random_bases(num_of_bases):
    """This function selects a random basis for each bit"""
    
    bases_string = ""
    for _ in range(num_of_bases):
        randBasis = random.randint(0, 1)     # Flip Coin

        if randBasis == 0:
            bases_string += "Z" 
        else:
            bases_string += "X"
            
    return bases_string
	

# from qiskit import *
def encode(bits, bases):
    """This function encodes each bit into the given basis."""
    
    encoded_qubits = []
    
    for bit, basis in zip(bits, bases):
        qc = QuantumCircuit(1, 1)     # Create a quantum circuit for each qubit
        
        # Possible Cases
        if bit == "1" :
            qc.x(0)

        if basis == 'X' :
            qc.h(0)
            
        encoded_qubits.append(qc)
            
    return encoded_qubits


# from qiskit_aer import Aer
# from qiskit import *
def measure(qubits, bases):
    """This function measures each qubit in the corresponding basis chosen for it.
        - qubits : a series of 1-qubit Quantum Circuit
        - bases : a string of random [X, Z] bases"""

    # bits = np.zeros(len(bases), dtype = int)    # The results of measurements
    bits = ""
        
    for idx, (qubit, basis) in enumerate(zip(qubits, bases)):

        if basis == "X" :
            qubit.h(0)
            
        qubit.measure(0, 0)
        
        # Execute on Simulator
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(qubit, simulator)
        result = simulator.run(transpiled_circuit, shots=1).result()
        counts = result.get_counts()
        measured_bit = max(counts, key=counts.get)     # Max doesn't matter for simulator since there is only one shot.

        bits += str(measured_bit)
        # bits[idx] = int(measured_bit)
        
    return bits


# import numpy as np
def array_to_string(array):
    result = np.array2string(
        array, 
        separator = "", 
        max_line_width = (len(array)+3))
    return result.strip('[').strip(']')


# import base64
def convert_to_octets(key):

    octets = []
    num_octets = len(key) // 8

    for i in range(num_octets):
        start = i * 8
        end = start + 8
        octet = key[start:end]
        octets.append(int(octet, 2))

    return bytearray(octets)


######################################################################## In Hamming ########################################################################

# Importing Qiskit
# from qiskit import *
# from qiskit.qasm2 import dumps
# from qiskit_aer import Aer

# import numpy as np
# import operation as op
# from functools import reduce
# import random

### Functions used

# from functools import reduce
def hamming(bits, order):
    '''
    Takes a string of bits to be corrected (bob bits). Bit-wise sums the indices of elements which are '1'. The 0th bit stores the parity
    of the entire block. The location of the error is returned. 
    If the location is not '0', the current 0th parity is matched with that of the parity obtained after flipping the bit at the location obtained. 
    If the parity matches, then the error is found and corrected. If the parity doesn't match then there are more than 1 error.

    If the location is '0', then no error is present.
    
    '''
    
    loc = reduce(lambda x, y : x^y, [i for i, bit in enumerate(bits) if bit == 1])    # x^y will apply xor to the binary rep of i -> index of 1s
    # loc = reduce(op.XOR, [i for i, bit in enumerate(bob_bits) if bit == 1])
    print(f"{loc = }")
    
    binary_rep = f"{bin_rep(loc, order)}"

    par = sum(bits[i] for i in range(0, len(bits)))%2    # Parity of the entire block. It should be 0-Even

    print(f"\n Hamming : ", end = " ")
    if loc != 0 :
        if par != 0 :
            err_count = 1
            print(f"Error found at location : {loc}")

        else :
            err_count = 2
            print("2 errors found")
            
    else : 
        err_count = 0
        print("No errors found")

    print(f" {err_count = }, {loc = }, {binary_rep = }")
    
    return err_count, loc, binary_rep


# import numpy as np
def Order(bits):
    try : return np.ceil(np.log2(len(bits))).astype(int)
    except : return np.ceil(np.log2(Unprocessed_key_len)).astype(int)
        

def bin_rep(loc, order):
    '''
    Takes a number(int) and order/precision(int) as an input, and returns the binary form with the requested precision.
    '''
    bin_loc = bin(loc)[2:]
    bin_rep = f"{'0'*(order - len(bin_loc))}{bin_loc}"
    
    return bin_rep


# import numpy as np
def parity(order):
    '''
    Takes in order(int) as a parameter. Returns 2 arrays : 
       - parity_bits : an array containing '0' and the powers of 2 till 2^(order-1)
       - bin_parity : an array of the binary representation of elements of parity_bits   
    '''
    PARITY_DICT = {0:0, **{2**i : 0 for i in range(order)}}    # Initializes the PARITY_DICT
    # parity_bits = np.array([0] + [2**i for i in range(order)]).astype(int)
    bin_parity = np.array([bin_rep(int(i), int(order)) for i in PARITY_DICT.keys()])

    return PARITY_DICT, bin_parity
    
    
# import numpy as np
def parity_locs(order):
    '''
    Takes in order(int) as a parameter. Returns an array :
        - parity_locs : A block(array reshaped as square matrix) with 1 at the locations of parity bits
    '''
    parity_locs = np.full(2**order, '-', dtype = object)
    PARITY_DICT = parity(order)[0]
    
    for loc in PARITY_DICT.keys() : parity_locs[loc] = '1'

    return parity_locs


# import numpy as np
def block(bits, order):
    dim = int(2**(order/2))
    # bits = bits.astype(int)
    # print(f" {len(bits) = } ")
    # print(f'{type(bits[0])= }')
    
    if len(bits) != 2**order : return f"key size (= {len(bits)}) not an exponent of 2 : {bits}"
        
    elif not order%2 : 
        try :
            return(f"block : \n {bits.reshape(dim, dim)} \n Shape of the block : {dim}*{dim}")
        except : 
            int_bits = np.array([int(bit) for bit in bits])
            return(f"block : \n {int_bits.reshape(dim, dim)} \n Shape of the block : {dim}*{dim}")

    else :
        return(f"bit string(Order is odd, can't project to a block) : \n")#{bits} \n Shape of the block : {bits.shape}")

# import numpy as np
def create_parity_block(bits, order, PARITY_DICT):
    '''
    A function to take a string of bits shared through QKD, and to morph them into a parity block with parity bits(unchecked) embedded
    '''
    block = np.zeros(2**order).astype('uint8')
    ### Encode alice_keys and get the PARITY_DICT before proceeding to bob_key
    
    j = 0 
    for i in range(2**order) : 
        if i in PARITY_DICT.keys() : block[i] = PARITY_DICT[i]
        elif j < len(bits) :    
            block[i] = bits[j]
            j += 1

    block, PARITY_DICT = encode_parity(block, order, PARITY_DICT)
    
    return block


# import numpy as np
def encode_parity(bits, order, PARITY_DICT):
    # order = np.ceil(np.log2(len(alice_bits)))
    
    sub_block = int(2**(order - 1))
    parity_of = np.zeros((len(PARITY_DICT), sub_block)).astype(int)   # An array to store the locations affecting the parity p
    
    for p in range(1, order+1) :    # checking for 1 at position p. eg : bin(45) = 101101
    
        bit_index = 2**(p-1)
        highlight = np.zeros(2**order).astype(int)                        # Highlights the locations affected by the current parity bit
        # print(f"bin rep of {bit_index = } : {bin_parity[p]}")
        Sum = 0
    
        for i in range(sub_block):                                         #  Order-1 = 5. range(5) = 0, 1, 2, 3, 4 => order-2
            bin_index = bin_rep(i, order-1)                                # Index(in binary formin binary form) for the data bits : 5 digits : 00010
            bin_index = bin_index[: order-p] + '1' + bin_index[order-p :]
            index = int(bin_index, base = 2)                                # Gives the index(int) of the elements to be considered for the current parity element
            
            parity_of[p, i] = index
            # highlight[index] = 1

            if bit_index != index :
                Sum = np.mod(Sum + bits[index], 2)

        # PARITY_DICT[bit_index] = np.mod( sum( bits[parity_of[p, i]] for i in range(sub_block) if bit_index != parity_of[p, i] ), 2 )

        PARITY_DICT[bit_index]= Sum
        bits[bit_index] = PARITY_DICT[bit_index]
            
        # print(highlight.reshape(dim, dim))
    
    PARITY_DICT[0] = sum( bits[i] for i in range(1, 2**order) )%2
    bits[0] = PARITY_DICT[0]
    # print(f"Parity locations : \n{parity_of[1:]}") 
    
    print("\n Hamming Results : ", hamming(bits, order))
    print(f" Uncorrected {block(bits, order)}")
    
    return bits, PARITY_DICT