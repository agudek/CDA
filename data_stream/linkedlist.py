
class llist(object):
    """docstring for llist"""
    def __init__(self):
        super(llist, self).__init__()
        self.root = None
        self.end = None
        self.length = 0

    def root(self):
        return self.root

    def pop(self):
        if self.end == None:
            return
        prev = self.end.prev
        self.end.prev = None
        if prev !=None:
            prev.next = None
        self.end = prev
        self.length -= 1
        
    def insert(self,left,right,key,val):
        n = node(key,val)
        if self.root == None:
            self.root = n
            self.end = n
        elif right == self.root:
            self.root = n
            right.prev = n
            n.next = right
        elif left == self.end:
            self.end = n
            left.next = n
            n.prev = left
        else:
            left.next = n
            n.prev = left
            n.next = right
            right.prev = n
        self.length += 1

    def remove(self, n):
        if self.root == n:
            self.root = n.next
        if self. end == n:
            self.end = n.prev
        if n.prev != None:
            n.prev.next = n.next
        if n.next != None:
            n.next.prev = n.prev
        self.length-=1

    def aslist(self):
        l = []
        n = self.root
        while n!=None:
            l.append((n.key, n.value))
            n = n.next
        return l

class node(object):
    """docstring for node"""
    def __init__(self, key, val):
        super(node, self).__init__()
        self.key = key
        self.value = val
        self.prev = None
        self.next = None        
     
