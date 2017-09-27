# you need to install hypothesis for this to work

"""
Property testing for functions in huffman.py.
"""

import unittest
from random import shuffle
from huffman import byte_to_bits, bits_to_byte, get_bit, make_freq_dict
from huffman import huffman_tree, get_codes, number_nodes
from huffman import generate_compressed, generate_uncompressed
from huffman import avg_length, tree_to_bytes, num_nodes_to_bytes
from nodes import HuffmanNode
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text

settings.register_profile("norand", settings(derandomize=True,
                                             max_examples=200))
settings.load_profile("norand")


class TestByteUtilities(unittest.TestCase):
    """Property tests for byte functions"""
    
    @given(integers(0, 255))
    def test_byte_to_bits(self, b):
        """byte_to_bits produces binary strings of length 8"""

        self.assertTrue(set(byte_to_bits(b)).issubset({"0", "1"}))
        self.assertEqual(len(byte_to_bits(b)), 8)
        
    @given(text(["0", "1"], 0, 4, 8))
    def test_bits_to_byte(self, s):
        """bits_to_byte produces byte"""

        b = bits_to_byte(s)
        self.assertTrue(isinstance(b, int))
        self.assertTrue(0 <= b <= 255)
        
    @given(integers(0, 255), integers(0, 7))
    def test_get_bit(self, byte, bit_pos):
        """get_bit(byte, bit) produces  bit values"""

        b = get_bit(byte, bit_pos)
        self.assertTrue(isinstance(b, int))
        self.assertTrue(0 <= b <= 1)


class TestCompressionCode(unittest.TestCase):
    """Property tests for Huffman functions"""

    @given(binary(0, 100, 1000))
    def test_make_freq_dict(self, byte_list):
        """make_freq_dict returns dictionary whose values
        sum to the number of bytes consumed"""

        b, d = byte_list, make_freq_dict(byte_list)
        self.assertTrue(isinstance(d, dict))
        self.assertEqual(sum(d.values()), len(b))

    @given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256, 256))
    def test_huffman_tree(self, d):
        """huffman_tree returns a non-leaf HuffmanNode"""

        t = huffman_tree(d)
        self.assertTrue(isinstance(t, HuffmanNode))
        self.assertTrue(not t.is_leaf())

    @given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256, 256))
    def test_get_codes(self, d):
        """the sum of len(code) * freq_dict[code] is optimal, so it
        must be invariant under permutation of the dictionary"""
        # NB: this also tests huffman_tree indirectly

        t = huffman_tree(d)
        c1 = get_codes(t)
        d2 = list(d.items())
        shuffle(d2)
        d2 = dict(d2)
        t2 = huffman_tree(d2)
        c2 = get_codes(t2)
        self.assertEqual(sum([d[k] * len(c1[k]) for k in d]), 
                         sum([d2[k] * len(c2[k]) for k in d2]))
        
    @given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256, 256))
    def test_number_nodes(self, d):
        """if the root is an interior node, it must be numbered
        two less than the number of symbols"""
        # a complete tree has one fewer interior nodes than
        # it has leaves, and we are numbering from 0
        # NB: this also tests huffman_tree indirectly

        t = huffman_tree(d)
        assume(not t.is_leaf())
        count = len(d)
        number_nodes(t)
        self.assertEqual(count, t.number + 2)

    @given(dictionaries(integers(0, 255), integers(1, 1000), dict, 2, 256, 256))
    def test_avg_length(self, d):
        """avg_length should return a float in the
        interval [0, 8]"""

        t = huffman_tree(d)
        f = avg_length(t, d)
        self.assertTrue(isinstance(f, float))
        self.assertTrue(0 <= f <= 8.0)

    @given(binary(2, 100, 1000))
    def test_generate_compressed(self, b):
        """generate_compressed should return a bytes
        object that is no longer than the input bytes, and
        the size of the compressed object should be
        invariant under permuting the input"""
        # NB: this also indirectly tests make_freq_dict, huffman_tree,
        # and get_codes

        d = make_freq_dict(b)
        t = huffman_tree(d)
        c = get_codes(t)
        compressed = generate_compressed(b, c)
        self.assertTrue(isinstance(compressed, bytes))
        self.assertTrue(len(compressed) <= len(b))
        l = list(b)
        shuffle(l)
        b = bytes(l)
        d = make_freq_dict(b)
        t = huffman_tree(d)
        c = get_codes(t)
        compressed2 = generate_compressed(b, c)
        self.assertEqual(len(compressed2), len(compressed))
        
    @given(binary(2, 100, 1000))
    def test_tree_to_bytes(self, b):
        """tree_to_bytes generates a bytes representation of
        a post-order traversal of a trees internal nodes"""
        # Since each internal node requires 4 bytes to represent,
        # and there are 1 fewer internal node than distinct symbols,
        # the length of the bytes produced should be 4 times the
        # length of the frequency dictionary, minus 4"""
    # NB: also indirectly tests make_freq_dict, huffman_tree, and
    # number_nodes

        d = make_freq_dict(b)
        assume(len(d) > 1)
        t = huffman_tree(d)
        number_nodes(t)
        output_bytes = tree_to_bytes(t)
        dictionary_length = len(d)
        leaf_count = dictionary_length
        self.assertEqual(4 * (leaf_count - 1), len(output_bytes))

    @given(binary(2, 100, 1000))
    def test_num_nodes_to_bytes(self, b):
        """num_nodes_to_bytes returns a bytes object that
        has length 1 (since the number of internal nodes cannot
        exceed 256)"""
        # NB: also indirectly tests make_freq_dict and huffman_tree

        d = make_freq_dict(b)
        assume(len(d) > 1)
        t = huffman_tree(d)
        number_nodes(t)
        n = num_nodes_to_bytes(t)
        self.assertTrue(isinstance(n, bytes))
        self.assertEqual(len(n), 1)


class TestRoundTrip(unittest.TestCase):
    """Property test for round trip"""

    @given(binary(1, 100, 1000))
    def test_round_trip(self, b):
        """test inverting generate_compressed and generate_uncompressed"""

        orig_text = b
        freq = make_freq_dict(orig_text)
        assume(len(freq) > 1)
        tree = huffman_tree(freq)
        codes = get_codes(tree)
        compressed = generate_compressed(orig_text, codes)
        uncompressed = generate_uncompressed(tree, compressed, len(orig_text))
        assert orig_text == uncompressed

if __name__ == "__main__":
    unittest.main()
