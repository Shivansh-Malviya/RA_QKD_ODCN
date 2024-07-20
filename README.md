# RA_QKD_ODCN
This repository is for the codes to reproduce the results seen in a paper by Bowen Chen et al., titled Resource allocation in Quantum-Key-distribution Optical Data Center Network.
The notebook "Assembled" is used as a workbench, where all the different classes are imported and the heuristics is established.

The class CR has a class method "generate_crs(numCRs)" to generate a set of X CRs, and initialize them as cr(index, s, d, tk, sl, status)
The class Links has a class method "initialize_links(edges)" to create a set of links as (end_nodes, lambda_tdc, lambda_q1, lambda_q2, lambda_q3) from the edges data 
FF(path, cr) is a class method in the class Links, which allocates the qsc and tdc resources to the cr based on first-fit approach. 

## QKD
The protocol in use is BB84. Channel noise is considered to be in the range of 10^(-3) to 10^(-4). Key length of the order of 1000. In general 1 or 2 error could take place(in absence of Eve).
Added the error-correction protocol. As of now, using the Hamming code. It has the code to encode the parity bits on Alice's side, and correct 1-bit error on Bob's side(can detect 2-bit errors).
