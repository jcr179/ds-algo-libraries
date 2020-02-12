from collections import defaultdict
from heapq import heappush, heappop

class Graph(object):
    """ Creates Graph object with a self.graph defaultdict with 
    key: node name, val: set of tuples node names that are adjacent to that node. 
    
    By default, graph is undirected. Can make it directed by passing directed=True. 
    If self.directed=True, self.graph still shows nodes that are directly connected but
    when adding edges only the one direction is added to self.graph
    
    May provide an iterable nodeRange listing the node names you would like to use.
    Essential if the number of nodes is known but there may be isolated nodes."""
    
    def __init__(self, directed=False, nodeRange=None, weightedNodes=False):
        self.graph = defaultdict(dict)     # dict. key: node name, val: dict of adjacent node names, with key: adjacent node, val: weight  
        self.directed = directed            # bool. defaults to False, meaning undirected graph. Else directed graph 
        if nodeRange:
            # Initialize so can keep track of isolated nodes 
            for x in nodeRange:
                self.graph[x] = {}
        if weightedNodes: # Create dictionary of key: node name, val: node weight 
            self.nodeWeights = {node: 1 for node in self.getGraph()}
        else: # Make empty dict so can use setNodeWeight method without throwing exception, but avoid overhead
            self.nodeWeights = {}
                
    def addNode(self, node, weight=1):
        """ Add node to graph without connecting it to any other nodes. Pass in name of new node. 
            Returns False if node not added due to it already being there, True if a new node is added. """
        if node in self.graph:
            return False
        else:
            self.graph[node] = {}
            return True
            
    def removeNode(self, node):
        """ Remove node from graph, eliminating all edges it had at the same time. Pass in name of new node. 
            Returns False if node is not in the graph, returns True otherwise on removal from graph. """
        if node not in self.graph:
            return False
        else:
            for n in self.graph: # is it faster for big graphs to check if node is in graph[n]?
                try:
                    self.graph[n].pop(node)
                except KeyError:
                    pass
            self.graph.pop(node)
            return True
        
    def addEdge(self, nodes, weight=1):
        """ (undirected) Take tuple (node1, node2) to add to graph 
            (directed)   Take tuple (node1, node2) meaning node1 directed to node2, add to graph """
        n1 = nodes[0]
        n2 = nodes[1]
        if n2 not in self.graph[n1]: # If edge doesn't already exist 
            self.graph[n1][n2] = weight
            if not self.directed:
                self.graph[n2][n1] = weight
        else: # "Overwrite" costlier path with cheaper one, for Dijkstra
            if weight < self.graph[n1][n2]:
                self.graph[n1][n2] = weight 
                if not self.directed:
                    self.graph[n2][n1] = weight
            
    def removeEdge(self, nodes):
        """ Removes edge and returns True if it exists. Otherwise returns False. 
            Takes tuple nodes (node1, node2) and removes edge between them if possible. 
            If graph is directed, tuple should be (from node, to node) """
        n1 = nodes[0]
        n2 = nodes[1]
        if not self.directed:
            if not (n1 in self.graph[n2] and n2 in self.graph[n1]):
                return False
            else:
                self.graph[n1].pop(n2)
                self.graph[n2].pop(n1)
                return True 
        else: # If graph is directed 
            if n2 not in self.graph[n1]:
                return False
            else:
                self.graph[n1].pop(n2)
                return True
        
            
    def getGraph(self):
        # Returns a dictionary of key: node name, val: nodes adjacent to key node 
        return dict(self.graph)
     
    """
    def getEdges(self):
        # Returns a dictionary of key: frozenset of adjacent nodes, val: weight of edge
        return dict(self.edges)
    """
        
    def getMaxNumberOfEdges(self):
        # Returns maximum number of edges knowing the number of nodes in the graph using closed formula
        numNodes = len(self.getGraph())
        return ( numNodes * (numNodes - 1) ) // 2
                    
    
    def setNodeWeight(self, node, weight):
        self.nodeWeights[node] = weight
        
    def getNodeWeights(self):
        return self.nodeWeights
    
    def shortestPath(self, start):
        """ Use Dijkstra's algorithm to find shortest path from start to finish, or from start to all nodes if returnOnFinish=False.
            All edge weights must be non-negative. (Bellman-Ford works on negative weights)
            There must be no isolated nodes; the graph must be connected. 
            Works on directed AND undirected graphs. """
            
        # Initialize dictionary of distances from start to each node
        g = self.getGraph()
        nodes = set([node for node in g])
        visited = set()
        distances = {}
        for node in nodes:
            distances[node] = float("inf")
        distances[start] = 0 # Since start node is initialized to 0, it is picked as minimum first 
        
        #debug print(nodes)
        #debug print(visited)
        #debug print(distances)
        
        
        while nodes:
            
            minDist = float("inf")
            minNode = None
            for node in nodes:
                if distances[node] < minDist:
                    minDist  = distances[node]
                    minNode = node
                    
            if minNode is None: # All reachable nodes visited?
                #debug print("No minimum distance node that has not been visited found, terminating")
                break
                
            #debug print("Minimum distance: ", minDist, "\tNode: ", minNode)
            visited.add(minNode)
            
            for adjacentNode in g[minNode]:
                if adjacentNode not in visited:
                    if distances[minNode] + g[minNode][adjacentNode] < distances[adjacentNode]:
                        distances[adjacentNode] = distances[minNode] + g[minNode][adjacentNode]
                
            #debug print("\nUpdated distances:")
            #debug print(distances)
            
            nodes.remove(minNode)
            
        return distances # FAILS IF MORE THAN 1 EDGE IS SHARED BY THE SAME 2 NODES
        
    def shortestPath2(self, start): # NEED TO ADJUST BASED ON WHETHER GRAPH NODES USE 0- OR 1-INDEXING!!!
        # By Janne Karila https://codereview.stackexchange.com/questions/79025/dijkstras-algorithm-in-python 
        A = [None] * len(self.getGraph())
        queue = [(0, start)]
        while queue:
            path_len, v = heappop(queue)
            if A[v] is None: # v is unvisited
                A[v] = path_len
                for w, edge_len in self.getGraph()[v].items():
                    if A[w] is None:
                        heappush(queue, (path_len + edge_len, w))

        # to give same result as original, assign zero distance to unreachable vertices             
        return [float('inf') if x is None else x for x in A] 
    
                    
    def minSpanTree(self, start):
        """ Make a minimum spanning tree from a graph using Prim's algorithm starting from node 'start'. 
            Returns dictionary of key values of each node in the MST, the sum of which's values being the minimum value of the MST. """
        g = self.getGraph()
        
        mstSet = set()
        nodes = set()
        
        vals = {}
        for node in g:
            vals[node] = float("inf")
            nodes.add(node)
        vals[start] = 0
        
        while nodes:
            tmp = []
            for node, value in vals.items():
                if node not in mstSet:
                    tmp.append(vals[node])
                    
            minVal = min(tmp)    
            
            minNode = None 
            for node, value in vals.items():
                if value == minVal and node not in mstSet:
                    minNode = node
                    break
            
            if minNode is None:
                break
                
            mstSet.add(minNode)
            
            for adjacentNode in g[minNode]:
                if adjacentNode not in mstSet:
                    if g[minNode][adjacentNode] < vals[adjacentNode]:
                        vals[adjacentNode] = g[minNode][adjacentNode]
                        #print(adjacentNode, "value updated to ", vals[adjacentNode])
                    
            nodes.remove(minNode)
            
        return vals
        
    def bfsFindAllNodes(self, start, stopDepth):
        # Time complexity O(V+E), V: num of nodes, E: num of edges ###
        #Returns set of nodes reachable within stopDepth 
        
        g = self.getGraph()
        graphNodes = set(g)
        visited = set()
        visited.add(start)
        depth = 0
        
        nodesToCheck = [start]
        
        while depth < stopDepth and visited != graphNodes:
            depth += 1
            newNodesToCheck = []
            for node in nodesToCheck:
                adjacent = g[node].keys()
                toCheck = adjacent - visited

                visited |= set(toCheck)
                #print('visited: ', visited)
                newNodesToCheck.extend(toCheck)
                #print('newnodestocheck: ', newNodesToCheck)
            nodesToCheck = newNodesToCheck
                
        return visited
# end of class Graph
        
            
nodes1 = (1, 2)
nodes2 = (3, 1)

# Test: initialize graph with 4 nodes
print("### TEST: Initialize graph with nodes 1-4")
g = Graph(nodeRange=[1, 2, 3, 4])
print(g.getGraph())
print("Graph is directed?", g.directed)

# Test: add adjacent nodes 
print()
print("------------------------------------------")
print("### TEST: Add adjacent nodes")
g.addEdge(nodes1)
g.addEdge(nodes2)

print(g.getGraph())

# Test: Find all nodes with BFS starting from node 1 - removed for now for Graph2
"""
print()
print("------------------------------------------")
print("### TEST: Find all nodes with BFS")
print(g.bfsFindAllNodes(1))
"""

# Test: remove edge 
print()
print("------------------------------------------")
print("### TEST: Remove existing edge")
result = g.removeEdge(nodes1)
print()
print(g.getGraph())
print("Edge removed: ", result)

# Test: remove non-existent edge 
print()
print("------------------------------------------")
print("### TEST: Remove non-existent edge")
result = g.removeEdge((1,4))
print()
print(g.getGraph())
print("Edge removed: ", result)

# Test: add isolated node, then fail to add it again, then connect it to a different node  
print()
print("------------------------------------------")
print("### TEST: Add isolated node")
newNode = 'A'
result = g.addNode(newNode)
print()
print(g.getGraph())
print("Node added: ", result)

print()
print("Trying to add same node again...")
result = g.addNode(newNode)
print(g.getGraph())
print("Node added: ", result)

print()
print("Connecting the new node to an existing one...")
nodes = (newNode, 1)
g.addEdge(nodes)
print(g.getGraph())

# Test: remove a node and all its associations, then try to remove the same node  
print()
print("------------------------------------------")
print("### TEST: Remove node; all its edges must go")
node = 1
result = g.removeNode(node)

print(g.getGraph())
print("Node removed: ", result)

print()
print("Trying to remove the same node...")
result = g.removeNode(node)

print(g.getGraph())
print("Node removed: ", result)

# Test: Weighted nodes 
print()
print("------------------------------------------")
print("### TEST: Initialize graph with weighted nodes, then change weights")
gw = Graph(nodeRange=range(3), weightedNodes=True)

print("\nGraph architecture...")
print(gw.getGraph())

print("\nNode weights...")
print(gw.nodeWeights)

print("\nChanging node weights...")
gw.setNodeWeight(1, 1000)
gw.setNodeWeight(2, -2)
print(gw.getNodeWeights())

# Test: Dijkstra shortest path algorithm
print()
print("------------------------------------------")
print("### TEST: Dijkstra's algorithm")
g2 = Graph(nodeRange=range(9))

print("\nGraph before adding edges...")
print(g2.getGraph())

print("\nGraph after adding edges...")
g2.addEdge((0,1), 4)
g2.addEdge((0,7), 8)
g2.addEdge((1,2), 8)
g2.addEdge((1,7), 11)
g2.addEdge((7,8), 7)
g2.addEdge((7,6), 1)
g2.addEdge((6,5), 2)
g2.addEdge((2,8), 2)
g2.addEdge((8,6), 6)
g2.addEdge((2,3), 7)
g2.addEdge((2,5), 4)
g2.addEdge((3,5), 14)
g2.addEdge((3,4), 9)
g2.addEdge((5,4), 10)
g2.addNode('UNREACHABLE')
print(g2.getGraph())

print("\nShortest distances to each node...")
res = g2.shortestPath(0)
print(res)

# Test: Prim's minimum spanning tree algorithm
print()
print("------------------------------------------")
print("### TEST: Prim's minimum spanning tree algorithm")
res = g2.minSpanTree(0)
print(res)

# Test: Heap implementation of Dijkstra 
print()
print("------------------------------------------")
print("### TEST: Dijkstra V2")
res = g2.shortestPath2(0)
print(res)

# Test: BFS with variable depth stopping 
print()
print("------------------------------------------")
print("### TEST: BFS depth 0")
res = g2.bfsFindAllNodes(0,0)
print(res)
print("### TEST: BFS depth 1")
res = g2.bfsFindAllNodes(0,1)
print(res)
print("### TEST: BFS depth 2")
res = g2.bfsFindAllNodes(0,2)
print(res)
print("### TEST: BFS depth 3")
res = g2.bfsFindAllNodes(0,3)
print(res)