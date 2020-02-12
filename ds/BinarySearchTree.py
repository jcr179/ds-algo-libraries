class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

def insert(root, node):
    if root is None:
        root = node
    else:
        if root.data > node.data:
            if root.left is None:
                root.left = node
            else:
                insert(root.left, node)
        else:
            if root.right is None:
                root.right = node
            else:
                insert(root.right, node)

def inOrderPrint(root): # prints globally sorted values 
    if not root:
        return
        
    inOrderPrint(root.left)
    print(root.data)
    inOrderPrint(root.right)
    
def levelOrderPrint(root): # prints sorted values of level 0, then level 1, ...
    queue = []
    temp_node = root 
    if temp_node is None:
        temp_node = root

    while temp_node != None:
        print(temp_node.data)
        if temp_node.left:
            queue.append(temp_node.left)
        if temp_node.right:
            queue.append(temp_node.right)
        try:
            temp_node = queue.pop(0)
        except IndexError as e:
            temp_node = None    

def preOrderPrint(root): # print all left subtrees first then right subtrees 
    if not root:
        return        
        
    print(root.data)
    preOrderPrint(root.left)
    preOrderPrint(root.right) 
    
r = Node(3)
insert(r, Node(7))
insert(r, Node(1))
insert(r, Node(5))
insert(r, Node(2))
insert(r, Node(8))

inOrderPrint(r)
print()
preOrderPrint(r) 
print()
levelOrderPrint(r)   