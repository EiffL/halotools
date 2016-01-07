"""
Module storing the TabularAsciiReader, a class providing a memory-efficient 
algorithm for reading a very large ascii file that stores tabular data 
of a data type that is known in advance. 

"""

__all__ = ('TabularAsciiReader', )

import os
import gzip
from time import time
import numpy as np

class TabularAsciiReader(object):
    """
    Class providing a memory-efficient algorithm for 
    reading a very large ascii file that stores tabular data 
    of a data type that is known in advance. 

    When reading ASCII data with 
    `~halotools.sim_manager.TabularAsciiReader.read_ascii`, user-defined 
    cuts on columns are applied on-the-fly using a python generator 
    to yield only those columns whose indices appear in the 
    input ``columns_to_keep_dict``. 

    As the file is read, the data is generated in chunks, 
    and a customizable mask is applied to each newly generated chunk. 
    The only aggregated data from each chunk are those rows  
    passing all requested cuts, so that the 
    `~halotools.sim_manager.TabularAsciiReader` 
    only requires you to have enough RAM to store the *cut* catalog, 
    not the entire ASCII file. 

    The primary method is the class is  
    `~halotools.sim_manager.TabularAsciiReader.read_ascii`. 
    The output of this method is a structured Numpy array, 
    which can then be stored in your preferred binary format 
    using the built-in Numpy methods, h5py, etc. 

    The algorithm assumes that data of known, unchanging type is 
    arranged in a consecutive sequence of lines within the ascii file, 
    and that the appearance of an empty line demarcates the end 
    of the data stream. 
    """
    def __init__(self, input_fname, columns_to_keep_dict, 
        header_char='#', row_cut_min_dict = {}, row_cut_max_dict = {}, 
        row_cut_eq_dict = {}, row_cut_neq_dict = {}):
        """
        """
        self.input_fname = self._get_fname(input_fname)

        self.header_char = self._get_header_char(header_char)

        self._determine_compression_safe_file_opener()

        self._process_columns_to_keep(columns_to_keep_dict)

        self.row_cut_min_dict = row_cut_min_dict
        self.row_cut_max_dict = row_cut_max_dict
        self.row_cut_eq_dict = row_cut_eq_dict
        self.row_cut_neq_dict = row_cut_neq_dict

        self._verify_input_row_cuts_keys()
        self._verify_min_max_consistency()
        self._verify_eq_neq_consistency()

    def _verify_input_row_cuts_keys(self, **kwargs):
        """ Require all columns upon which a row-cut is placed to also appear in 
        the input ``columns_to_keep_dict``. For purposes of good bookeeping, 
        you are not permitted to place a cut on a column that you do not keep.        
        """
        potential_row_cuts = ('row_cut_min_dict', 'row_cut_max_dict', 
            'row_cut_eq_dict', 'row_cut_neq_dict')
        for row_cut_key in potential_row_cuts:

            row_cut_dict = getattr(self, row_cut_key)

            for key in row_cut_dict:
                try:
                    assert key in self.columns_to_keep_dict.keys()
                except AssertionError:
                    msg = ("\nThe ``"+key+"`` key does not appear in the input \n"
                        "``columns_to_keep_dict``, but it does appear in the "
                        "input ``"+row_cut_key+"``. \n"
                        "It is not permissible to place a cut "
                        "on a column that you do not keep.\n")
                    raise KeyError(msg)

    def _verify_min_max_consistency(self, **kwargs):
        """ Verify that no min_cut column has a value greater to the corresponding max_cut. 

        Such a choice would laboriously result in a final catalog with zero entries. 
        """

        for row_cut_min_key, row_cut_min in self.row_cut_min_dict.iteritems():
            try:
                row_cut_max = self.row_cut_max_dict[row_cut_min_key]
                if row_cut_max <= row_cut_min:
                    msg = ("\nFor the ``"+row_cut_min_key+"`` column, \n"
                        "you set the value of the input ``row_cut_min_dict`` to "
                        +str(row_cut_min)+"\nand the value of the input "
                        "``row_cut_max_dict`` to "+str(row_cut_max)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise ValueError(msg)
            except KeyError:
                pass

        for row_cut_max_key, row_cut_max in self.row_cut_max_dict.iteritems():
            try:
                row_cut_min = self.row_cut_min_dict[row_cut_max_key]
                if row_cut_min >= row_cut_max:
                    msg = ("\nFor the ``"+row_cut_max_key+"`` column, \n"
                        "you set the value of the input ``row_cut_max_dict`` to "
                        +str(row_cut_max)+"\nand the value of the input "
                        "``row_cut_min_dict`` to "+str(row_cut_min)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise ValueError(msg)
            except KeyError:
                pass


    def _verify_eq_neq_consistency(self, **kwargs):
        """ Verify that no neq_cut column has a value equal to the corresponding eq_cut. 

        Such a choice would laboriously result in a final catalog with zero entries. 
        """ 

        for row_cut_eq_key, row_cut_eq in self.row_cut_eq_dict.iteritems():
            try:
                row_cut_neq = self.row_cut_neq_dict[row_cut_eq_key]
                if row_cut_neq == row_cut_eq:
                    msg = ("\nFor the ``"+row_cut_eq_key+"`` column, \n"
                        "you set the value of the input ``row_cut_eq_dict`` to "
                        +str(row_cut_eq)+"\nand the value of the input "
                        "``row_cut_neq_dict`` to "+str(row_cut_neq)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise ValueError(msg)
            except KeyError:
                pass

        for row_cut_neq_key, row_cut_neq in self.row_cut_neq_dict.iteritems():
            try:
                row_cut_eq = self.row_cut_eq_dict[row_cut_neq_key]
                if row_cut_eq == row_cut_neq:
                    msg = ("\nFor the ``"+row_cut_neq_key+"`` column, \n"
                        "you set the value of the input ``row_cut_neq_dict`` to "
                        +str(row_cut_neq)+"\nand the value of the input "
                        "``row_cut_eq_dict`` to "+str(row_cut_eq)+"\n"
                        "This will result in zero selected rows and is not permissible.\n")
                    raise ValueError(msg)
            except KeyError:
                pass

    def _process_columns_to_keep(self, columns_to_keep_dict):
        """ Private method performs sanity checks in the input ``columns_to_keep_dict`` 
        and uses this input to define two attributes used for future bookkeeping, 
        ``self.column_indices_to_keep`` and ``self.dt``. 
        """

        for key, value in columns_to_keep_dict.iteritems():
            try:
                assert type(value) == tuple
                assert len(value) == 2
            except AssertionError:
                msg = ("\nThe value bound to every key of the input ``columns_to_keep_dict``\n"
                    "must be a two-element tuple.\n"
                    "The ``"+key+"`` is not the required type.\n"
                    )
                raise TypeError(msg)

            column_index, dtype = value
            try:
                assert type(column_index) == int
            except AssertionError:
                msg = ("\nThe first element of the two-element tuple bound to every key of \n"
                    "the input ``columns_to_keep_dict`` must an integer.\n"
                    "The first element of the ``"+key+"`` is not the required type.\n"
                    )
                raise TypeError(msg)
            try:
                dt = np.dtype(dtype)
            except:
                msg = ("\nThe second element of the two-element tuple bound to every key of \n"
                    "the input ``columns_to_keep_dict`` must be a string recognized by Numpy\n"
                    "as a data type, e.g., 'f4' or 'i8'.\n"
                    "The second element of the ``"+key+"`` is not the required type.\n"
                    )
                raise TypeError(msg)
        self.columns_to_keep_dict = columns_to_keep_dict

        # Create a hard copy of the dict keys to ensure that 
        # self.column_indices_to_keep and self.dt are defined 
        # according to the same sequence
        column_key_list = list(columns_to_keep_dict.keys())

        # Only data columns with indices in self.column_indices_to_keep 
        # will be yielded by the data_chunk_generator
        self.column_indices_to_keep = list(
            [columns_to_keep_dict[key][0] for key in column_key_list]
            )

        # The rows of data yielded by the data_chunk_generator 
        # will be assumed to be the following Numpy dtype
        self.dt = np.dtype(
            [(key, columns_to_keep_dict[key][1]) for key in column_key_list]
            )

    def _get_fname(self, input_fname):
        """ Verify that the input fname does not already exist. 
        """
        # Check whether input_fname exists. 
        if not os.path.isfile(input_fname):
            # Check to see whether the uncompressed version is available instead
            if not os.path.isfile(input_fname[:-3]):
                msg = "Input filename %s is not a file" 
                raise IOError(msg % input_fname)
            else:
                msg = ("Input filename ``%s`` is not a file. \n"
                    "However, ``%s`` exists, so change your input_fname accordingly.")
                raise IOError(msg % (input_fname, input_fname[:-3]))

        return os.path.abspath(input_fname)

    def _get_header_char(self, header_char):
        """ Verify that the input header_char is 
        a one-character string or unicode variable. 

        """
        try:
            assert (type(header_char) == str) or (type(header_char) == unicode)
            assert len(header_char) == 1
        except AssertionError:
            msg = ("\nThe input ``header_char`` must be a single string character.\n")
            raise TypeError(msg)
        return header_char

    def _determine_compression_safe_file_opener(self):
        """ Determine whether to use *open* or *gzip.open* to read 
        the input file, depending on whether or not the file is compressed. 
        """
        f = gzip.open(self.input_fname, 'r')
        try:
            f.read(1)
            self._compression_safe_file_opener = gzip.open
        except IOError:
            self._compression_safe_file_opener = open
        finally:
            f.close()

    def header_len(self):
        """ Number of rows in the header of the ASCII file. 

        Parameters 
        ----------
        fname : string 

        Returns 
        -------
        Nheader : int

        Notes 
        -----
        The header is assumed to be those characters at the beginning of the file 
        that begin with ``self.header_char``. 

        All empty lines that appear in header will be included in the count. 

        """
        Nheader = 0
        with self._compression_safe_file_opener(self.input_fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]==self.header_char) or (l=="\n") ):
                    Nheader += 1
                else:
                    break

        return Nheader

    def data_len(self):
        """ 
        Number of rows of data in the input ASCII file. 

        Returns 
        --------
        Nrows_data : int 
            Total number of rows of data. 

        Notes 
        -------
        The returned value is computed as the number of lines 
        between the returned value of `header_len` and 
        the next appearance of an empty line. 

        The `data_len` method is the particular section of code 
        where where the following assumptions are made:

            1. The data begins with the first appearance of a non-empty line that does not begin with the character defined by ``self.header_char``. 

            2. The data ends with the next appearance of an empty line. 

        """
        Nrows_data = 0
        with self._compression_safe_file_opener(self.input_fname, 'r') as f:
            for i, l in enumerate(f):
                if ( (l[0:len(self.header_char)]!=self.header_char) and (l!="\n") ):
                    Nrows_data += 1
        return Nrows_data

    def data_chunk_generator(self, chunk_size, f):
        """
        Python generator uses f.readline() to march 
        through an input open file object to yield 
        a chunk of data with length equal to the input ``chunk_size``. 
        The generator only yields columns that were included 
        in the ``columns_to_keep_dict`` passed to the constructor. 

        Parameters 
        -----------
        chunk_size : int 
            Number of rows of data in the chunk being generated 

        f : File
            Open file object being read

        Returns 
        --------
        chunk : tuple 
            Tuple of data from the ascii. 
            Only data from ``column_indices_to_keep`` are yielded. 

        """
        cur = 0
        while cur < chunk_size:
            line = f.readline()    
            parsed_line = line.strip().split()
            yield tuple(parsed_line[i] for i in self.column_indices_to_keep)
            cur += 1 

    def apply_row_cut(self, array_chunk):
        """ Method applies a boolean mask to the input array 
        based on the row-cuts determined by the 
        dictionaries passed to the constructor. 

        Parameters 
        -----------
        array_chunk : Numpy array  

        Returns 
        --------
        cut_array : Numpy array             
        """ 
        mask = np.ones(len(array_chunk), dtype = bool)

        for colname, lower_bound in self.row_cut_min_dict.iteritems():
            mask *= array_chunk[colname] > lower_bound

        for colname, upper_bound in self.row_cut_max_dict.iteritems():
            mask *= array_chunk[colname] < upper_bound

        for colname, equality_condition in self.row_cut_eq_dict.iteritems():
            mask *= array_chunk[colname] == equality_condition

        for colname, inequality_condition in self.row_cut_neq_dict.iteritems():
            mask *= array_chunk[colname] != inequality_condition

        return array_chunk[mask]

    def read_ascii(self, chunk_memory_size = 500.):
        """ Method reads the input ascii and returns 
        a structured Numpy array of the data 
        that passes the row- and column-cuts. 

        Parameters 
        ----------
        chunk_memory_size : int, optional 
            Determine the approximate amount of Megabytes of memory 
            that will be processed in chunks. This variable 
            must be smaller than the amount of RAM on your machine; 
            choosing larger values typically improves performance. 
            Default is 500 Mb. 

        Returns 
        --------
        full_array : array_like 
            Structured Numpy array storing the rows and columns 
            that pass the input cuts. The columns of this array 
            are those selected by the ``column_indices_to_keep`` 
            argument passed to the constructor. 

        See also 
        ----------
        data_chunk_generator
        """
        print("\n...Processing ASCII data of file: \n%s\n " % self.input_fname)
        start = time()

        file_size = os.path.getsize(self.input_fname) 
        chunk_memory_size *= 1e6 # convert to bytes to match units of file_size
        num_data_rows = self.data_len()
        print("Total number of rows in detected data = %i" % num_data_rows)

        # Set the number of chunks to be filesize/chunk_memory, 
        # but enforcing that 0 < Nchunks <= num_data_rows
        try:
            Nchunks = max(1, min(file_size / chunk_memory_size, num_data_rows))
        except ZeroDivisionError:
            msg = ("\nMust choose non-zero size for input ``chunk_memory_size``")
            raise ValueError(msg)

        num_rows_in_chunk = int(num_data_rows / float(Nchunks))
        num_full_chunks = num_data_rows / num_rows_in_chunk
        num_rows_in_chunk_remainder = num_data_rows - num_rows_in_chunk*Nchunks

        header_length = self.header_len()
        print("Number of rows in detected header = %i \n" % header_length)

        chunklist = []
        with self._compression_safe_file_opener(self.input_fname, 'r') as f:

            for skip_header_row in xrange(header_length):
                _s = f.readline()

            for _i in xrange(num_full_chunks):
                print("... working on chunk "+str(_i)+" of "+str(num_full_chunks))

                chunk_array = np.array(list(
                    self.data_chunk_generator(num_rows_in_chunk, f)), dtype=self.dt)
                cut_chunk = self.apply_row_cut(chunk_array)
                chunklist.append(cut_chunk)

            # Now for the remainder chunk
            chunk_array = np.array(list(
                self.data_chunk_generator(num_rows_in_chunk_remainder, f)), dtype=self.dt)
            cut_chunk = self.apply_row_cut(chunk_array)
            chunklist.append(cut_chunk)

        full_array = np.concatenate(chunklist)
                
        end = time()
        runtime = (end-start)

        if runtime > 60:
            runtime = runtime/60.
            msg = "Total runtime to read in ASCII = %.1f minutes\n"
        else:
            msg = "Total runtime to read in ASCII = %.2f seconds\n"
        print(msg % runtime)
        print("\a")

        return full_array



