'''
File containing functions and libraries required for executing Resource allocation protocol.
'''

import numpy as np
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
import time

from contextlib import contextmanager
import sys, os
import tqdm
import warnings

############################################################### Generating_Network_Topology.ipynb ###############################################################
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np 


def create_topology(edges, save_fig = False, draw = True):
	'''
	for creating and printing a network topology with node and edge labels.
	Requires :
		edges(2d-list) : contains tuples of form (source node, destination node, weight)
	'''
	
	all_values = set([nodes for row in edges for nodes in row[:2]])    # A list of all the values/nodes in first 2 columns : s & d
	correction = -min(all_values) + 1     # Correction for the convention in indexing of nodes in the data of edges : nodes start form 0
	
	edges = [(u + correction, v + correction, w) for u, v, w in edges]
	
	g = nx.Graph()
	numNodes = max(all_values) + correction
	
	nodes = np.arange(1, numNodes + 1)
	g.add_nodes_from(nodes)
	
	g.add_weighted_edges_from(edges)
	weights = {(item[0], item[1]): item[2] for item in edges}
	
	# Position of nodes
	pos = nx.spring_layout(g, seed = 5) 
	nx.draw_networkx_nodes(g, pos, node_size =200)
	nx.draw_networkx_edges(g, pos, edgelist = edges, width = 2) 
	nx.draw_networkx_labels(g, pos, font_size = 7)
	nx.draw_networkx_edge_labels(g, pos, weights)
	
	print("Number of Nodes: ", numNodes)
	print("Number of Links :", len(g.edges()))
	
	if save_fig == True:
		figname = str(numNodes) + "-nodes.png"
		plt.savefig(figname)
	
	#Configuraiton
	ax = plt.gca()
	ax.margins(0.08)
	plt.axis('off')
	plt.tight_layout()
	plt.show()
	
	return g


def create_bi_topology(edges, save_fig = False, draw = True):
	'''
	for creating and printing a BIDIRECTIONAL network topology with node and edge labels.
	Requires :
		edges(2d-list) : contains tuples of form (source node, destination node, weight)
	
	NOTE : Requires the edges to contain every unidirectional edge; i.e. 2 entries for a single bidirectional edge
	'''
	
	all_values = set([nodes for row in edges for nodes in row[:2]])    # A list of all the values/nodes in first 2 columns : s & d
	correction = -min(all_values) + 1     # Correction for the convention in indexing of nodes in the data of edges : nodes start form 0
	
	edges = [(u + correction, v + correction, w) for u, v, w in edges]
	
	g = nx.DiGraph()
	numNodes = max(all_values) + correction
	
	nodes = np.arange(1, numNodes + 1)
	g.add_nodes_from(nodes)
	
	g.add_weighted_edges_from(edges)
	weights = {(item[0], item[1]): item[2] for item in edges}
	
	# Position of nodes
	pos = nx.spring_layout(g, seed = 5) 
	nx.draw_networkx_nodes(g, pos, node_size =200)
	nx.draw_networkx_edges(g, pos, edgelist = edges, width = 2) 
	nx.draw_networkx_labels(g, pos, font_size = 7)
	nx.draw_networkx_edge_labels(g, pos, weights)
	
	print("Number of Nodes: ", numNodes)
	print("Number of Links :", len(g.edges()))
	
	if save_fig == True:
		figname = str(numNodes) + "-nodes.png"
		plt.savefig(figname)
	
	#Configuraiton
	ax = plt.gca()
	ax.margins(0.08)
	plt.axis('off')
	plt.tight_layout()
	plt.show()
	
	return g


def k_sp(g, k = 10):
	"""Calculates k shortest paths for all node pairs in a graph using NetworkX.
	
	Args:
		g: The input graph.
		node_list: A list of nodes in the graph.
		k: The number of shortest paths to calculate.
	
	Returns:
		A dictionary of dictionaries mapping node pairs to path-cost pairs.
	"""
	# Nodes start from 0
	
	num_nodes = len(g)
	k_sp_dict = {}   # {(n1, n2) : {p1 : c1, ..., pk : ck}, ..., (n_k_1, n_k) : {p1 : c1, ..., pk : ck} } ; p_i = (l1, l2, ..., l_j)
	nodes = list(g.nodes)
	
	for i in nodes  :    # (i, )
		for j in nodes :    # (, j)
			if i == j:
				continue
				
			k_shortest_paths = list(nx.shortest_simple_paths(g, source = i, target = j, weight='weight'))
			path_costs = [sum(g[s][d]['weight'] for s, d in zip(path, path[1:])) for path in k_shortest_paths]
	
			k_sp_dict[(i, j)] = {tuple(path) : cost for path, cost in zip(k_shortest_paths, path_costs)}
	
	return k_sp_dict

###################################################################### Generating_CRs.ipynb ###################################################################


class CR:

	"""
	This class represents a Connection Request (CR) object. A CR can be in different states (allocated, blocked) and has various attributes 
	such as source node, destination node, security level, status, allocated resources, path, and index.
	
	**Attributes:**
		index (int): Unique identifier for the CR object.
		s (int): Source node of the CR.
		d (int): Destination node of the CR.
		sl (str): Security Level of the CR ("high", "medium", "low").
		tk (int): time slots required (currently set to 1).
		status (str): Current status of the CR ("allocated", "blocked", "initialized").
		text (str): Text associated with the CR.
		path (list): List of nodes representing the current path for the CR (obtained using k_sp or shortest_path).
		allocated_resources (list): List containing allocated resources (QSC, ts).
	
	**Methods:**
		__init__(self, index, s, d, sl, tk, status): Initializes the CR object.
		update_status(self, status, allocated_resources=None, path=None): Updates the CR's status, allocated resources, and path.
		display_info(self): Prints information about the CR object.
	
	**Class Methods:**
		def generate_crs(cls, numCR=1): Generates a list of CR objects with random attributes.
		def create_priority_queue(cls, queues='PQ'): Creates a priority queue based on security level from the list of CRs.
		def SRCR(cls, CRs): Calculates and prints the Success Rate of Connection Requests (SRCR).
		def NSP(cls, CRs, channel='total'): Calculates and prints the Network Security Provision (NSP).
		def display_all(cls, CRs): Prints detailed information for all CR objects in the list.
	"""
	
	#CRs = []
	sl_values = ["high", "medium", "low"]
	sfw = {"high" : 5, "medium" : 4, "low" : 3}    # Security factor weight
	
	def __init__(self, index, s, d, sl, tk):     # Constructor : Gets called as soon as an object is declared
		self.index = index
		
		self.s = s
		self.d = d
		
		self.sl = sl    # SL can be hign, mediuum, low
		self.tk = tk
		self.status = 'initialized'    # Status can be Allocatted, blocked
		self.text = ''
	
		self.path = None
		self.allocated_resources = [None, None]    # [QSC, ts]
		
	
	def update_status(self, status, allocated_resources = [None, None], path = None) :    # A method to update the status
		"""
		Updates the CR object's status, allocated resources, and path.
		
		Args:
			status (str): The new status for the CR ("allocated", "blocked", etc.).
			allocated_resources (list, optional): List containing allocated resources (QSC, ts). Defaults to None.
			path (list, optional): List of nodes representing the allocated path. Defaults to None.
		"""
		self.status = status
	
		self.path = path
		self.allocated_resources = allocated_resources
	
	
	def display_info(self):
		print(f"CR: Index : {self.index}, Source Node ={self.s}, Destination Node={self.d}, Security Level={self.sl}, status={self.status}, allocated resources = {self.allocated_resources}, Path : {self.path}")
	
	###################################################################################################################################################
	
	@classmethod
	def generate_crs(cls, numCR = 1):    
		"""
		Generates a list of CR objects with random attributes (source, destination, security level, status) based on the specified number.
		
		Args:
			numCR (int, optional): The desired number of CR objects to generate. Defaults to 1.
		
		Returns:
			list: A list of CR objects.
		"""
	
		
		cls.CRs = []
	
		# Creating a list of uniformly sampled SLs
		n = math.floor(numCR/3)
		uniform_sample = n*cls.sl_values
	
		for i in range(numCR - 3*n):    # the range would be either 1 or 2. i can either be 0 or (0, 1)
			 uniform_sample.append(random.choice(cls.sl_values))
	
		# Initializing individual CRs
		for i in range(numCR):
			# Generating (source, destination) pair
			s = random.randint(1, numNodes)
			d = random.randint(1, numNodes)   
	
			while s == d : d = random.randint(1, numNodes)    # Checking if the source and destination nodes are same
	
			# Assigning sl            
			sl = random.choice(uniform_sample)    # Randomly selecting from a uniform sample
			uniform_sample.remove(sl)    # Removing the first(any) equivalent sl, to maintain the uniformity
	
			# currently only dealing with tk = 1.
			tk = 1
	
			cr = cls(i, s, d, sl, tk)
			cls.CRs.append(cr)
		##
		return cls.CRs
	
	@classmethod
	def create_priority_queue(cls, queues = 'PQ'):
		"""
		Creates a priority queue for the CR objects based on their security level (high, medium, low).
		
		Args:
			queues (str, optional): Specifies whether to return all queues or just the combined priority queue.
				- 'PQ' (default): Returns only the combined priority queue.
				- 'all': Returns all three queues (high, medium, low) along with the combined priority queue.
		
		Returns:
			list or tuple:
				- If queues='PQ': Returns a list containing the combined priority queue.
				- If queues='all': Returns a tuple containing four elements:
					- The combined priority queue (list).
					- The high-priority queue (list).
					- The medium-priority queue (list).
					- The low-priority queue (list).
		"""
		CR_1 = []
		CR_0 = []
		CR_minus1 = []
	
		for cr in cls.CRs :
			if cr.sl == "high": 
				CR_1.append(cr)
			elif cr.sl == "medium": 
				CR_0.append(cr)
			else: 
				CR_minus1.append(cr)
	
		PQ = CR_1 + CR_0 + CR_minus1    # Priority queue
	
		if queues == 'all': 
			return PQ, CR_1, CR_0, CR_minus1
		else: 
			return PQ
			
	
	@classmethod
	def SRCR(cls, CRs):    # Returns the SRCR
		"""
		Calculates and prints the Success Rate of Connection Requests (SRCR).
		
		Args:
			CRs (list): List of CR objects.
		
		Returns:
			float: The SRCR value.
		
		Prints:
			The SRCR value to the console.
		"""
	
		num_allocated_crs = sum(1 for cr in CRs if cr.status == "allocated" )
		
		SRCR = num_allocated_crs/len(CRs)
		print(f"The Success rate of connection requests(SRCR) is : {SRCR}")
		
		return SRCR
		
	
	@classmethod
	def NSP(cls, CRs, channel = 'total'):    # Returns the total NSP, if no 2nd argument given
		"""
		Calculates and prints the Network Security Performance (NSP) for each security level and the total NSP.
		
		Args:
			CRs (list): List of CR objects.
			channel (str, optional): Specifies whether to return all NSP values or just the total NSP.
				- 'total' (default): Returns and prints the total NSP.
				- 'all': Returns a list containing NSP values for each security level and prints the total NSP.
		
		Returns:
			float or list:
				- If channel='total': Returns the total NSP value.
				- If channel='all': Returns a list containing NSP values for each security level.
		
		Prints:
			The total NSP and/or individual NSP values for each security level to the console.
		"""
	
		nsp = [0, 0, 0, 0]
		for i, priority in enumerate(cls.sl_values) :
			nsp[i+1] = sum(cls.sfw[priority] for cr in CRs if cr.status == "allocated" and cr.sl == priority)
			nsp[0] += nsp[i+1]
	
		print(f"The NSP of the network is : {nsp[0]}")
	
		if channel == 'all':
			return nsp
		else :
			return nsp[0]
	
	
	@classmethod
	def display_all(cls, CRs):
		for cr in CRs:
			cr.display_info()



def plot_crs(*CRs) :
	"""
	Plots the distribution of security levels (SL) for varying numbers of Connection Requests (CRs).
	
	Generates a line plot showing the counts of high, medium, and low security level CRs
	across different numbers of total CRs.
	
	Args:
		*CRs: Variable length argument list of CR objects (not used in this function).
	
	Returns:
		None
	
	Side Effects:
		Generates a plot showing the distribution of security levels.
	"""
	high = []
	mid = []
	low = []
	numNodes = 14
	for X in range(25, 850, 25):
		CRs = CR.generate_crs(X) 
		counts = dict(Counter([CR.sl for CR in CRs]))
		high.append(counts["high"])
		mid.append(counts["medium"])
		low.append(counts["low"])
	
	plt.plot(range(25, 850, 25), high, label = "high")
	plt.plot(range(25, 850, 25), mid, label = "mid")
	plt.plot(range(25, 850, 25), low, label = "low")
	
	plt.scatter(range(25, 850, 25), high, label = "high")
	plt.scatter(range(25, 850, 25), mid, label = "mid")
	plt.scatter(range(25, 850, 25), low, label = "low")
	
	plt.legend(loc = 'best')
	plt.title("Distribution of SL levels for various number of CRs")
	plt.xlabel("Num of CRs")
	plt.ylabel("Counts of SL")

####################################################################### Links.ipynb #############################################################################

# The number of dictionaries in the class can be reduced by combining the values with different keys

#import numpy as np
class Links:
	"""
	Represents a network of links connecting nodes. Each link has specific resources (time slots)
	allocated for different security levels (SL) and traditional data channels (TDC).
	
	Attributes:
		channel_ts (dict): A dictionary mapping security level ("tdc", "high", "medium", "low") to the
							corresponding total number of time slots in the channel.
		n_ts (dict): A dictionary mapping security level ("tdc", "high", "medium", "low") to the
					 number of time slots in the corresponding channel.
		channel (dict): A dictionary mapping security level ("tdc", "high", "medium", "low") to the
						 channel index.
		priority (dict): A dictionary mapping channel index (0-3) to the corresponding security level.
		total_ts (int): The total number of time slots in each link (sum of all security level time slots).
		links (np.ndarray, optional): A 2D NumPy array containing Link objects at corresponding node indices
									   (initialized in `initialize_links`).
		indices (np.ndarray, optional): A mask indicating the location of each link in the `links` array
										(initialized in `initialize_links`).
		ordered_indices (list, optional): A list of tuples representing ordered pairs of nodes for each link
										  (initialized in `initialize_links`).
	
	"""
	
	
	## Class variables
	
	#links = []
	#indices : a mask
	#unordered_indices = []    # Not useful in case of bidirectional graph
	#numLinks = len(ordered_indices)
	
	
	# Creating a dictionary with sl and the corresponding number of time-slots in the channel
	num_ts = {"tdc" : 47, "high" : 8, "medium" : 10, "low" : 12,
			  0 : 47, 1 : 8, 2 : 10, 3 : 12}
	link_ts = num_ts[1] + num_ts[2] + num_ts[3]    # Total ts in each link
	sl_channel = {"tdc" : 0, "high" : 1, "medium" : 2, "low" : 3, 
				  0 : "tdc", 1 : "high", 2 : "medium", 3 : "low"}
	
	# channel = {"tdc" : 0, "high" : 1, "medium" : 2, "low" : 3}    # A dictionary for channel from sl 
	# priority = {0 : "tdc", 1 : "high", 2 : "medium", 3 : "low"}    # A dictionary for sl from channel
	
	
	def __init__(self, nodes, weight):    # nodes = (s, d); is a tuple so as to not confuse s and d individually
		self.nodes = nodes
		self.weight = weight
		
		self.lambda_tdc = np.ones(47, dtype = bool)
		
		#self.total_ts = Links.total_ts
		
		self.occupied_ts = np.zeros(4).astype(int)    # Stores the number of occupied time slots [total, q1, q2, q3]
		self.available_ts = np.array([Links.total_ts,    # Stores the number of available time slots [total, q1, q2, q3]
									 Links.num_ts[1], Links.num_ts[2], Links.num_ts[3]])
		
		self.lambda_q1 = np.ones(Links.num_ts[1], dtype = bool)    # for high sl
		self.lambda_q2 = np.ones(Links.num_ts[2], dtype = bool)    # for medium sl    
		self.lambda_q3 = np.ones(Links.num_ts[3], dtype = bool)    # for low sl
	
	
	def update_link(self, channel, slot):
		"""
		Updates the availability of a time slot in a specific channel.
	
		Args:
			channel (int): The channel index (0 for TDC, 1-3 for security levels).
			slot (int): The time slot index.
		"""
		#QSC = ["lambda_q1", "lambda_q2", "lambda_q3"]
		QSC = [1, 2, 3]
		TDC = [0]
		AC = QSC + TDC
		#if channel not in AC or ts < 0:
			#print("Invalid Update request!")
	
		if channel == 0:
			self.lambda_tdc[slot] = False
		
		elif channel == 1 and slot < len(self.lambda_q1):
			self.lambda_q1[slot] = False 
			
		elif channel == 2 and slot < len(self.lambda_q2):
			self.lambda_q2[slot] = False
			
		elif channel == 3 and slot < len(self.lambda_q3):
			self.lambda_q3[slot] = False
	
		else:
			raise ValueError("Invalid time slot value")
	
		if channel != 0:
			self.occupied_ts[channel] += 1 
			self.occupied_ts[0] += 1
	
			self.available_ts[channel] -= 1 
			self.available_ts[0] -= 1
		
	
	def display_info(self, wl_info = False):
		"""
		Displays information about the link, including available and occupied time slots.
	
		Args:
			wl_info (bool, optional): If True, also displays detailed wavelength information. Defaults to False.
		"""
		q1_count = self.available_ts[1] # nonzero => Available slots
		q2_count = self.available_ts[2] #== True
		q3_count = self.available_ts[3] #== True
		
		tdc_count = np.count_nonzero(self.lambda_tdc) #== True
		print(f"Link {self.nodes} : lambda_tdc_count = {tdc_count}, lambda_q1_count = {q1_count}, lambda_q2_count = {q2_count}, lambda_q3_count = {q3_count}, occupied_ts = {self.occupied_ts}, available_ts = {self.available_ts}")
	
		if wl_info:    # To show the wavelength occupancy
			print(f"QSC: \n q1 : {self.lambda_q1}, q2 : {self.lambda_q2}, q3 : {self.lambda_q3}, tdc : {self.lambda_tdc}")
		
	###################################################################################################################################################
	
	@classmethod
	def initialize_links(cls, edges):    # Initializing all the links and the individual link resources
		"""
		Initializes all links in the network based on the provided edges data.
	
		Args:
			edges (list): A list of tuples representing edges in the graph (source, destination, weight).
	
		Returns:
			tuple: A tuple containing the `links`, `indices`, and `ordered_indices` arrays.
		"""
		num_Nodes = max(set([nodes for row in edges for nodes in row[:2]]))    # A list of all the values/nodes in first 2 columns : s & d
		cls.total_ts = cls.link_ts * num_Nodes
		cls.links = np.zeros([num_Nodes+1, num_Nodes+1], dtype = object)    # Matrix containing link object at position s, d. Will have redundant entries.
	
		cls.ordered_indices = []
		for (s, d, w) in edges:
			nodes = (s, d)
			link = cls(nodes, w)
			cls.links[s, d] = link   
			cls.ordered_indices.append(nodes)
		# In the above call of __init__ constructor, the wavelength resources have also been initialized to all available(True)
					
		cls.indices = cls.links != 0
				
	
	@classmethod
	def path_resources(cls, path, sl):    # To check for the available time slots, following the continuity constraint
		"""
		Checks for available time slots along a specified path for a given security level.
	
		Args:
			path (list): A list of nodes representing the path.
			sl (str): The security level.
	
		Returns:
			tuple: A tuple containing lists of available time slots for TDC and the security level.
		"""
		available_tdcs = [True] * cls.num_ts["tdc"]    # For traditional data channel slots
		available_ts = [True] * cls.num_ts[sl]    # Creating a base boolean array of the size corresponding to the specific CR's sl
	
		for s, d in zip(path, path[1:]):    # Taking consecutive pairs of nodes and selecting the particular channel
			if sl == "high":
				band = cls.links[s, d].lambda_q1
			elif sl == "medium":
				band = cls.links[s, d].lambda_q2
			else:
				band = cls.links[s, d].lambda_q3
	
			# Checking for continuity constraints in ts of quantum channel
			available_ts = [a and b for a, b in zip(available_ts, band)]
			available_tdcs = [a and b for a, b in zip(available_tdcs, cls.links[s, d].lambda_tdc)]
			
		return available_ts, available_tdcs
	
	
	@classmethod
	def ASLC(cls, ch, path, aslc, beta_1 = 1, beta_2 = 0.5):
		"""
		Implements the Adaptive Spectrum Leasing Channel (ASLC) algorithm.
	
		Args:
			ch (int): The channel index.
			path (list): The path to be considered.
			aslc (str): The ASLC strategy (ASSL or AWSL).
			beta_1 (float, optional): The first threshold parameter for AWSL. Defaults to 1.
			beta_2 (float, optional): The second threshold parameter for ASSL. Defaults to 0.5.
	
		Returns:
			str: The selected security level.
		"""
		
		NT = []    # equivalent to dict channel_ts. starts from index 0 : for high priority
		OT = []    # Number of ts occupied along the ENTIRE path
		for i in range(1, 4):
			nt = cls.channel_ts[i]    # Total time slot in channel/n-th wavelength
			NT.append(nt)
				
			available_ts, available_tdcs = cls.path_resources(path, cls.priority[i])    # Available ts in the path for a particular wavelength
			ot = nt - np.count_nonzero(available_ts)    # Denotes the number of time-slot continuity occupied along the path for the given sl
			OT.append(ot)
				
		# ch could be 1, 2 or 3
		# n = ch - 1    # n could be 0, 1 or 2    # no need for n
	
		if aslc == "ASSL" and ch != 1:    # ch = 2 or 3. n = 1 or 2
			if OT[ch-1] >= beta_2 * NT[ch-1]:    # If the occupied resources in the higher priority are greater than a threshold, don't allocate to it
				QW = cls.priority[ch]
			else:
				QW = cls.priority[ch-1]
	
		elif aslc == "AWSL" and ch != 3:    # ch = 1 or 2. n = 0 or 1
			if OT[ch] >= beta_1 * NT[ch]:    # If the resources in current priority are more than a threshold, allocate to a lower priority
				QW = cls.priority[ch+1]
			else:
				QW = cls.priority[ch]
	
		else:    # For SSL and cases in ASSL but high priority, or AWSL but low priority
			QW = cls.priority[ch]
		 
		return QW
			
		
	@classmethod
	def FF(cls, path, cr, aslc = "ssl"):
		"""
		Performs First Fit (FF) spectrum allocation for a Connection Request (CR).
	
		Args:
			path (list): The path for the CR.
			cr (CR): The Connection Request object.
			aslc (str, optional): The ASLC strategy to use. Defaults to "ssl".
	
		Returns:
			bool: True if allocation is successful, False otherwise.
		"""
		# First check for availability of resources
		
		cr.sl = cls.ASLC(cls.channel[cr.sl], path, aslc)
		sl = cr.sl
		available_ts, available_tdcs = cls.path_resources(path, sl)    # List of available time slots in the channel along the entire path
	
		if cr.tk > len(available_ts):    # ts is 1 in our case by default. But in case we change it, this line will be useful
			raise ValueError("CR duration cannot be longer than the available time-slots in a channel")
			# Can also block it instead
		 
		if np.count_nonzero(available_ts) == 0 or np.count_nonzero(available_tdcs) == 0:    # A pre-condition to check if available slots are present
			cr.update_status("blocked")
			return False
	
		# Some issue here
		qscs = np.nonzero(available_ts)[0][0]    # This function will give the index of 1st non-zero/True element in the list
		tdcs = np.nonzero(available_tdcs)[0][0]    # At present, assuming only one tdcs is needed
		
		for s, d in zip(path, path[1:]):    # A loop to update all the links in the path 
			link = links[s, d]
	
			link.update_link(cls.channel["tdc"], tdcs)
			link.update_link(cls.channel[sl], qscs)    # Updating the i_th ts in channel for sl to False
	
		allocated_resources = [cls.channel[sl], qscs]
		cr.update_status("allocated", allocated_resources, path)
		#print(f"allocated ts_{qscs} successfully to CR {cr.index}")
		return True
		
	
	@classmethod
	def cipher(cls, cr) :
	
		path = cr.path
		
		for s, d in zip(path, path[1:]):    # A loop to update all the links in the path 
			link = links[s, d]
	
			key = link.bb84()    # Incomplete
			link.tdc(key, cr.text)    # Incomplete
			
			link.update_link(cls.channel["tdc"], tdcs)
			link.update_link(cls.channel[sl], qscs)    # Updating the i_th ts in channel for sl to False
	
		allocated_resources = [cls.channel[sl], qscs]
		cr.update_status("allocated", allocated_resources, path)
		
		return True
	
		
	@classmethod
	def TUR(cls, channel = 'total'):    # Returns the total TUR, if no 2nd argument given    
		"""
		Calculates the Timeslot Utilization Ratio (TUR) for the network.
	
		Args:
			channel (str, optional): Specifies whether to calculate TUR for all channels or a specific channel.
				- 'total' (default): Calculates the overall TUR.
				- Other values: Calculates TUR for the specified channel (not implemented).
	
		Returns:
			float or list:
				- If `channel` is 'total', returns the overall TUR.
				- Otherwise, returns a list of TUR values for each channel.
		"""
		# Only for quantum channel
		util_ts = [0, 0, 0, 0]
		tur = util_ts
		num_links = len(cls.ordered_indices)    # This also takes in consideration the symmetric part of a unidirectional link
		total_network_ts = num_links * cls.total_ts    # = 1260
		
		for nodes in cls.ordered_indices :
			link = cls.links[nodes]
			util_ts += link.occupied_ts
	
		tur[0] = util_ts[0]/total_network_ts
		for i in range(1, 4):
			tur[i] = util_ts[i]/(len(cls.ordered_indices)*cls.channel_ts[i])
		
		print("The time-slot utilization ratio(TUR) is : ", tur)
		
		if channel == 'all':
			return tur
	
		else: 
			return tur[0]       
			
	
	@classmethod
	def display_all_links(cls, wl_info = False):
		for nodes in cls.ordered_indices :
			cls.links[nodes].display_info(wl_info)
