"""Writing a Linked List"""

class LinkedNode(object):

    def __init__(self, data, next_ = None):

        self.data = data
        self.next_ = next_

    def __str__(self):
        return str(self.data)

    
class LinkedList(object):

    def __init__(self):

        self.head = None
        self.tail = None

    def add_node(self, data):

        new_node = LinkedNode(data)

        if self.head == None:
            self.head = new_node

        if self.tail != None:
            self.tail.next_ = new_node

        self.tail = new_node

    def del_ndxnode(self, index):
        prev = None
        node = self.head
        i = 0

        while node != None and i < index:
            prev = node
            node = node.next_

        if prev == None:
            self.head = node.next
        else:
            prev.node = node.next
    
    def print_list(self):
        cur_node = self.head

        while cur_node != None:
            print(cur_node.data)
            cur_node.next_

       
            
