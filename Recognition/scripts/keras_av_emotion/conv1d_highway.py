from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.layers.convolutional import ZeroPadding1D

class Conv1DHighway(Layer):

    def __init__(self,
                 nb_filter, 
                 filter_length,init='glorot_uniform',
                 border_mode='same', 
                 subsample_length=1,
                 transform_bias=-1,
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 input_length=None,
                 **kwargs):
        if 'transform_bias' in kwargs:
            kwargs.pop('transform_bias')
            warnings.warn('`transform_bias` argument is deprecated and '
                          'will be removed after 5/2017.')
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.transform_bias = transform_bias
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample_length = subsample_length
        self.subsample = (subsample_length, 1)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(Conv1DHighway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[2]
        self.W_shape = (self.filter_length, 1, input_dim, self.nb_filter)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.W_gate = self.init(self.W_shape, name='{}_W_carry'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.b_gate = K.variable(np.ones(self.nb_filter,), name='{}_b_gate'.format(self.name))
        else:
            self.b_gate = None

        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):

        #1d convolution
        print("x shape:", x._keras_shape)
        print("W shape:", K.shape(self.W))
        extended_x = K.expand_dims(x, 2)  # add a dummtransform dimension
        transform = K.conv2d(extended_x, self.W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering='tf',filter_shape=self.W_shape)
        transform = K.squeeze(transform, 2)  # remove the dummtransform dimension
        if self.bias:
            transform += K.reshape(self.b, (1, 1, self.nb_filter))
        transform = self.activation(transform)

        transform_gate = K.conv2d(extended_x, self.W_gate, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering='tf',filter_shape=self.W_shape)
        transform_gate = K.squeeze(transform_gate, 2)  # remove the dummtransform dimension
        
        if self.bias:
            transform_gate += K.reshape(self.b_gate, (1, 1, self.nb_filter))

        transform_gate = K.sigmoid(transform_gate)
        #transform_gate = K.activation(transform_gate)
        #print("transform_gate shape: ", K.int_shape(transform_gate))
        #we need zero padding for transform gate and carry gate
        padded = x._keras_shape[1] - K.int_shape(transform_gate)[1]
        #transform_gate = K.asymmetric_temporal_padding(transform_gate, left_pad=0, right_pad=padded)
        #print("padded transform_gate shape : ", K.int_shape(transform))
        carry_gate = 1.0 - transform_gate
        carry_gate = K.asymmetric_temporal_padding(carry_gate, left_pad=0, right_pad=padded)
        #print("padded carry_gate shape : ", K.int_shape(carry_gate))
        x_carried = x * carry_gate    

        #print("transform shape : ", K.int_shape(transform))
        #transform = K.asymmetric_temporal_padding(transform, left_pad=0, right_pad=padded)
        #print("padded transform shape : ", K.int_shape(transform))
        transform = transform * transform_gate
        transform = K.asymmetric_temporal_padding(transform, left_pad=0, right_pad=padded)
        
        output = transform  + x_carried
        print("output shape: ", K.int_shape(output))
        return output

    def get_output_shape_for(self, input_shape):
        #length =conv_output_length(input_shape[1],self.filter_length,self.border_mode,self.subsample[0])
        length = input_shape[1] #because of zero pedding                  
        return (input_shape[0], length, self.nb_filter)

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'filter_length': self.filter_length,
                  'transform_bias': self.transform_bias,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample_length': self.subsample_length,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Conv1DHighway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
