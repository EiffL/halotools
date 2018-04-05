r"""
Module contains the `~halotools.empirical_models.TensorflowGalpropModel` class
used to map any galaxy property to a halo catalog using a tensorflow model.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from astropy.extern import six
from abc import ABCMeta
import tensorflow as tf

__all__ = ('TensorflowGalpropModel',)
__author__ = ('Francois Lanusse',)

@six.add_metaclass(ABCMeta)
class TensorflowGalpropModel(object):
    r"""
    Container class for a tensorflow-based galaxy property sampler
    """

    def __init__(self, model_dir, function_name, seed=None, **kwargs):
        """
        """
        # Opens a tensorflow session
        self._sess = tf.Session()

        # Load the saved tensorflow model
        self._model = tf.saved_model.loader.load(sess,
                                                 tags=['serve'],
                                                 export_dir=model_dir)

        # Extracts the signature definition of the function to use
        self._definition = self._model.signature_def[function_name]

        # Stores the requested and generated quantities
        self._requested_quantities = tuple([n for n in self._definition.inputs])
        self._generated_quantities = tuple([n for n in self._definition.outputs])

        self._inputs = {input_name : self._definition.inputs[input_name].name
                            for input_name in self._requested_quantities}
        self._outputs = [self._definition.outputs[output_name].name
                            for output_name in self._generated_quantities]




    def _sampler(self, seed=None, batch_size=10000, **kwargs):
        r"""
        Function that samples from the graph
        """
        if 'table' in kwargs:
            table = kwargs['table']

        # Get size of table
        x_size = len(table)

        # Computes number of batches
        n_batches = x_size // batch_size
        n_batches += 0 if (x_size % batch_size) == 0 else 1

        # list storing the results
        outputs = {}
        for k in self._generated_quantities:
            outputs[k] = []

        # Resets the graph random seed
        if seed is not None:
            tf.set_random_seed(seed)

        # Create data dictionary
        feed_dict = {}
        # Runs through the catalog
        for b in range(n_batches):

            # Feed-in the data
            for input_name in self._inputs:
                feed_dict[self._inputs[input_name]] = table[input_name][b*batch_size:min((b+1)*batch_size, table_size)].astype('float32')

            # Process through
            smpl = sess.run(self._outputs, feed_dict=feed_dict)

            for i,k in enumerate(self._generated_quantities):
                outputs[k].append(smpl[i])

        # Concatenate results and add the columns to the table
        for k in generated_quantities:
            outputs[k] = np.concatenate(outputs[k])
            table[k][:] = outputs[k]

        return outputs
