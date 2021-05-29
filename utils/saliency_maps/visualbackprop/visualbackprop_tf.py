import tensorflow as tf
import os
import numpy as np
import cv2

class ConvolutionHead:

    def __init__(self,num_filters=8,features_per_filter=4):
        self.num_filters = num_filters 
        self.features_per_filter = features_per_filter
        self._w_value_list = None

    def __call__(self,x):
        image_height = int(x.shape[-3])
        image_width = int(x.shape[-2])
        image_channels = int(x.shape[-1])
        # Do Image whitening (Standardization)
        self.x = x
        head = tf.reshape(x,shape=[-1,image_height,image_width,image_channels])
        head = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),head)

        self.conv_layers = [head]
        with tf.variable_scope('perception'):
            conv_layers = [(24,5,2),(36,5,2),(48,3,2),(64,3,1),(self.num_filters,3,1)]
            for filters,kernel_size,stride in conv_layers:
                head = tf.layers.conv2d(
                    inputs=head,
                    filters=filters,
                    kernel_size=[kernel_size, kernel_size],
                    strides=stride,
                    padding="same",
                    activation=tf.nn.relu,
                    bias_initializer=tf.constant_initializer(0.01)
                )
                self.conv_layers.append(head)

        filter_output  = tf.split(head, num_or_size_splits=self.num_filters, axis=3)

        print("Each filter output is of shape "+str(filter_output[0].shape))
        self.filter_out_width = filter_output[0].shape[2]
        self.filter_out_height = filter_output[0].shape[1]
        filter_out_flattened = int(self.filter_out_width*self.filter_out_height)

        print("Filter out width: "+str(self.filter_out_width))
        print("Filter out height: "+str(self.filter_out_height))
        print("Filter out flattened: "+str(filter_out_flattened))

        feature_layer_list = []
        self._w_list = []
        for i in range(self.num_filters):
            flatten = tf.reshape(filter_output[i],[-1,filter_out_flattened])

            layer = tf.layers.Dense(units=self.features_per_filter, activation=tf.nn.relu,bias_initializer=tf.constant_initializer(0.01))
            feats = layer(flatten)
            self._w_list.append(layer.weights[0])
            # feats = tf.layers.dense(inputs=flatten, units=self.features_per_filter, activation=tf.nn.relu,bias_initializer=tf.constant_initializer(0.01))
            feature_layer_list.append(feats)

        self.feature_layer = tf.concat(feature_layer_list,1)
        print("Feature layer shape: "+str(self.feature_layer.shape))
        total_features = int(self.feature_layer.shape[1])

        feature_layer = tf.reshape(self.feature_layer,shape=[tf.shape(x)[0],tf.shape(x)[1],total_features])
        return feature_layer


    # Generates saliency mask according to VisualBackprop (https://arxiv.org/abs/1611.05418).

    def visual_backprop(self,tf_session,x_value):
        if(self._w_value_list is None):
            self._w_value_list = tf_session.run(self._w_list)


        A,feats = tf_session.run([self.conv_layers,self.feature_layer], feed_dict={self.x: x_value})

        aux_list = []
        means = []
        for i in range(len(A)): #for each feature map
            # layer index, batch_dimension
            a = A[i][0] # feature map [h,w,c]
            per_channel_max = a.max(axis=0).max(axis=0)
            a /= (per_channel_max.reshape([1,1,-1])+0.0001)
            means.append( np.mean( a, 2 ) )

        feat_act = []
        for i in range(len(self._w_value_list)):
            w_l = np.split(self._w_value_list[i],self.features_per_filter,axis=1)

            for w_i in range(len(w_l)):
                w = np.reshape(w_l[w_i],[self.filter_out_height,self.filter_out_width])
                feat_map = np.abs(w)*feats[0,i*self.features_per_filter+w_i]
                # feat_map = feat_map-feat_map.min()
                # feat_map = feat_map/feat_map.max()
                feat_act.append(feat_map)

        feat_act = np.stack(feat_act,axis=-1)
        for i in range(A[-1][0].shape[-1]):
            aux_list.append(("feat_{:d}".format(i),A[-1][0][:,:,i]))

        # print("feat act shape: ",str(feat_act.shape))
        feat_act = np.mean(feat_act,axis=-1)
        # print("feat act reshape: ",str(feat_act.shape))
        # means.append(feat_act)
        # print("last means: ",str(means[-1].shape))

        for i in range(len(means)-2, -1, -1):
            smaller = means[i+1]
            aux_list.append(("layer_{:d}".format(i),smaller))

            scaled_up = cv2.resize(smaller, (means[i].shape[::-1]))
            means[i] = np.multiply(means[i],scaled_up)

        mask = means[0]
        mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
        # mask = np.exp(mask)
        # mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
        mask = np.clip(mask, 0,1)


        return mask,aux_list