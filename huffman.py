"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    new_dict = dict()

    for word in text:
        if word in new_dict.keys():
            new_dict[word] += 1
        else:
            new_dict[word] = 1
    return new_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    new_lst = []

    if len(freq_dict) == 0:
        return None
    else:
        new_tup = [(freq, key) for (key, freq) in freq_dict.items()]
        new_tup.sort()
    for (freq, value) in new_tup:
        new_lst.append((freq, HuffmanNode(value)))

    if len(new_lst) == 1:
        new_lst.append((0, HuffmanNode(None)))

    while len(new_lst) > 1:
        left, right = new_lst.pop(0), new_lst.pop(0)
        new = HuffmanNode(None, left[1], right[1])
        new_lst.append((left[0] + right[0], new))
        new_lst.sort()

    return new_lst[0][1]

def get_codes_subtree(node, code=''):
    """Return a dict mapping symbols from tree to codes. Tree is rooted at
    HuffmanNode and we're assuming the code at the root node of the tree is
    parameter code.

    @param HuffmanNode node: a HuffmanTree
    @param str code: the code assighed to root node
    @rtype: dict(int: str)
    """
    new_dict = {}
    if node.left is not None:
        if node.left.symbol is not None:
            new_dict[node.left.symbol] = code + "0"
        left_nodes = get_codes_subtree(node.left, code + "0")
        for item in left_nodes:
            new_dict[item] = left_nodes[item]
    if node.right is not None:
        if node.right.symbol is not None:
            new_dict[node.right.symbol] = code + "1"
        right_nodes = get_codes_subtree(node.right, code + "1")
        for item in right_nodes:
            new_dict[item] = right_nodes[item]
    return new_dict


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> d = get_codes(tree)
    >>> d == {3:"00", 2:"01", 9:"1"}
    True
    """
    if tree.symbol is not None:
        return {tree.symbol: "0"}
    else:
        return get_codes_subtree(tree, '')


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.number
    1
    """
    thelist = []

    def inner_nodes(tree):
        """Traverses the tree and numbers internal nodes.
        @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
        @rtype: None
        """
        if tree is not None and (tree.right is not None or
                                 tree.left is not None):
            inner_nodes(tree.left)
            inner_nodes(tree.right)
            thelist.append(tree)
        for i in range(len(thelist)):
            thelist[i].number = i

    inner_nodes(tree)


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    average = get_codes(tree)
    total_characters = 0
    total_freq = 0
    for k in average:
        total_characters += len(average[k]) * freq_dict[k]
        total_freq += freq_dict[k]
    return total_characters / total_freq

def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    the_text = list(text)
    new_str = ''
    for item in the_text:
        new_str += codes[item]
    while len(new_str) % 8 != 0:
        new_str += "0"
    result = []
    for i in range(int(len(new_str) / 8)):
        new_byte = bits_to_byte(new_str[8 * i: 8 * (i + 1)]) # padding
        result.append(new_byte)
    return bytes(result)

def helper_tree_to_bytes(tree, thelist):
    """ Traverses tree and labels internal nodes with 1, leaf nodes with 0.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param list[int, int] thelist: list representation from tree_to_bytes
    @rtype: None
    """

    if tree is not None:
        helper_tree_to_bytes(tree.left, thelist)
        helper_tree_to_bytes(tree.right, thelist)
        if tree.left is None and tree.right is None:
            thelist.append(0)
            thelist.append(tree.symbol)
        else:
            thelist.append(1)
            thelist.append(tree.number)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    thelist = []
    helper_tree_to_bytes(tree, thelist)
    del thelist[-1]
    del thelist[-1]

    return bytes(thelist)


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    root = HuffmanNode(None, None)
    if node_lst[root_index].l_type == 1:
        left = generate_tree_general(node_lst, node_lst[root_index].l_data)
        root.left = left
    else:
        root.left = HuffmanNode(node_lst[root_index].l_data, None, None)
    if node_lst[root_index].r_type == 1:
        right = generate_tree_general(node_lst, node_lst[root_index].r_data)
        root.right = right
    else:
        root.right = HuffmanNode(node_lst[root_index].r_data, None, None)
    return root


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    root = HuffmanNode(None, None)
    length = len(node_lst)
    if node_lst[root_index].l_type == 1:
        left = generate_tree_general(node_lst,
                                     node_lst[root_index].l_data - length)
        root.left = left
    else:
        root.left = HuffmanNode(node_lst[root_index].l_data, None, None)
    if node_lst[root_index].r_type == 1:
        right = generate_tree_general(node_lst, node_lst[root_index].r_data - (
            length - 1))
        root.right = right
    else:
        root.right = HuffmanNode(node_lst[root_index].r_data, None, None)
    return root


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    >>> t = HuffmanNode(None, HuffmanNode(None, HuffmanNode(1), \
       HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))), HuffmanNode(None, \
       HuffmanNode(4), HuffmanNode(5)))
    >>> text = bytes([216, 0])
    >>> size = 4
    >>> result = generate_uncompressed(t, text, size)
    >>> result == bytes([5, 3, 1, 1])
    True
    """
    the_bits = ''
    uncompress_list = []
    current = tree

    for byte in text:
        the_bits += byte_to_bits(byte)
    for bit in the_bits:
        if bit == '0':
            current = current.left
    # keep going down the tree until you reach a leaf
        elif bit == '1':
            current = current.right
        if current.symbol is not None:
            uncompress_list.append(current.symbol)
            if len(uncompress_list) == size:
                # once it equals size, then we're done
                break
            current = tree

    return bytes(uncompress_list)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.
 cur_size = 0
    cur_bit = 0
    while cur_size < size:

        if the_bits[cur_bit] == '0':
            cur_tree_node = tree.left
        elif the_bits[cur_bit] == '1':
            cur_tree_node = tree.right

        if tree.is_leaf():
            uncompress_list.append(cur_tree_node.symbol)
        cur_bit += 1
        cur_size += 1
    return bytes(uncompress_list)
    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i + 1]
        r_type = buf[i + 2]
        r_data = buf[i + 3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """

    list_ = sorted(freq_dict, key=lambda x: freq_dict[x])
    the_list = []
    the_list.append(tree)
    i = 0
    while len(the_list) != 0:
        check_node = the_list.pop()
        if check_node.is_leaf():
            check_node.symbol = list_[i]
            i += 1
        else:
            if check_node.left is not None:
                the_list.append(check_node.left)
            if check_node.right is not None:
                the_list.append(check_node.right)

if __name__ == "__main__":
    import python_ta

    python_ta.check_all(config="huffman_pyta.txt")
    import doctest
    doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
