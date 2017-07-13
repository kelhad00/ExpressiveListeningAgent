import numpy

from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer

class Conv3DHighway(Layer):
    def __init__(self, nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, transform_bias=-1,
                 init='glorot_uniform', activation='relu', weights=None,
                 border_mode='same', subsample=(1, 1, 1), dim_ordering='th',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        
        self.nb_filter = nb_filter
        self.kernel_dim1 = kernel_dim1
        self.kernel_dim2 = kernel_dim2
        self.kernel_dim3 = kernel_dim3
        self.transform_bias = transform_bias
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        assert border_mode in {'valid', 'same'}, 'border_mode must be in {valid, same}'
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=5)]
        self.initial_weights = weights
        super(Conv3DHighway, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 5

        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.kernel_dim1, self.kernel_dim2, self.kernel_dim3)
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[4]
            self.W_shape = (self.kernel_dim1, self.kernel_dim2, self.kernel_dim3, stack_size, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.W_gate = self.init(self.W_shape, name='{}_W_carry'.format(self.name))

        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.b_gate = K.variable(numpy.ones(self.nb_filter,), name='{}_b_gate'.format(self.name))
            self.trainable_weights = [self.W, self.b, self.W_gate, self.b_gate]
        else:
            self.trainable_weights = [self.W, self.W_gate]
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

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            conv_dim1 = input_shape[2]
            conv_dim2 = input_shape[3]
            conv_dim3 = input_shape[4]
        elif self.dim_ordering == 'tf':
            conv_dim1 = input_shape[1]
            conv_dim2 = input_shape[2]
            conv_dim3 = input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        conv_dim1 = conv_output_length(conv_dim1, self.kernel_dim1,
                                  self.border_mode, self.subsample[0])
        conv_dim2 = conv_output_length(conv_dim2, self.kernel_dim2,
                                  self.border_mode, self.subsample[1])
        conv_dim3 = conv_output_length(conv_dim3, self.kernel_dim3,
                                  self.border_mode, self.subsample[2])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, conv_dim1, conv_dim2, conv_dim3)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], conv_dim1, conv_dim2, conv_dim3, self.nb_filter)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # compute the candidate hidden state
        
        input_shape = self.input_spec[0].shape
        conv_out = K.conv3d(x, self.W, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            volume_shape=input_shape,
                            filter_shape=self.W_shape)

        if self.dim_ordering == 'th':
            transform = conv_out + K.reshape(self.b, (1, self.nb_filter, 1, 1, 1))
        elif self.dim_ordering == 'tf':
            transform = conv_out + K.reshape(self.b, (1, 1, 1, 1, self.nb_filter))
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        transform = self.activation(transform)

        transform_gate = K.conv3d(x, self.W_gate, strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering=self.dim_ordering,
                            volume_shape=input_shape,
                            filter_shape=self.W_shape)

        if self.bias:
            if self.dim_ordering == 'th':
                transform_gate += K.reshape(self.b_gate, (1, self.nb_filter, 1, 1, 1 ))
            elif self.dim_ordering == 'tf':
                transform_gate += K.reshape(self.b_gate, (1, 1, 1, 1, self.nb_filter))
            else:
                raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
        
        transform_gate = K.sigmoid(transform_gate)
        #transform_gate = self.activation(transform_gate)
        carry_gate = 1.0 - transform_gate

        return transform * transform_gate + x * carry_gate

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'kernel_dim1': self.kernel_dim1,
                  'kernel_dim2': self.kernel_dim2,
                  'kernel_dim3': self.kernel_dim3,
                  'transform_bias': self.transform_bias,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias}
        base_config = super(Conv3DHighway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
