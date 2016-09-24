import argparse
import csv
import dicom
import gzip
import numpy as np
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import scipy.misc
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import tflearn
import sys
import time

def super_print(statement, f):
    """
    This basically prints everything in statement.
    We'll add a new line character for the output file.
    We'll just use print for the output.
    INPUTS:
    - statement: (string) the string to print.
    - f: (opened file) this is the output file object to print to
    """
    sys.stdout.write(statement + '\n')
    sys.stdout.flush()
    f.write(statement + '\n')
    return 0

def create_test_splits(path_csv_test):
    """
    Goes through the data folder and divides for testing.
    INPUTS:
    - path_csv_test: (string) path to test csv
    """
    X_tr = []
    X_te = []
    Y_tr = []
    Y_te = []
    # First, let's map examID and laterality to fileName
    dict_X_left = {}
    dict_X_right = {}
    counter = 0
    with open(path_csv_test, 'r') as file_crosswalk:
        reader_crosswalk = csv.reader(file_crosswalk, delimiter='\t')
        for row in reader_crosswalk:
            if counter == 0:
                counter += 1
                continue
            if row[3].strip()=='R':
                dict_X_right[row[0].strip()] = row[4].strip()
                X_te.append((row[0].strip(), 'R', row[4].strip()))
            elif row[3].strip()=='L':
                dict_X_left[row[0].strip()] = row[4].strip()
                X_te.append((row[0].strip(), 'L', row[4].strip()))
    #for key_X in set(dict_X_left.keys()) & set(dict_X_right.keys()):
    #    X_te.append((dict_X_left[key_X], dict_X_right[key_X]))
    return X_tr, X_te, Y_tr, Y_te

def create_data_splits(path_csv_crosswalk, path_csv_metadata):
    """
    Goes through data folder and divides train/val.
    INPUTS:
    - path_csv_crosswalk: (string) path to first csv file
    - path_csv_metadata: (string) path to second csv file
    There should be two csv files.  The first will relate the filename
    to the actual patient ID and L/R side, then the second csv file
    will relate this to whether we get the cancer.  This is ridiculous.
    Very very very bad filesystem.  Hope this gets better.
    """
    # First, let's map the .dcm.gz file to a (patientID, examIndex, imageView) tuple.
    dict_img_to_patside = {}
    counter = 0
    with open(path_csv_crosswalk, 'r') as file_crosswalk:
        reader_crosswalk = csv.reader(file_crosswalk, delimiter='\t')
        for row in reader_crosswalk:
            if counter == 0:
                counter += 1
                continue
            dict_img_to_patside[row[5].strip()] = (row[0].strip(), row[4].strip())
    # Now, let's map the tuple to cancer or non-cancer.
    dict_tuple_to_cancer = {}
    counter = 0
    with open(path_csv_metadata, 'r') as file_metadata:
        reader_metadata = csv.reader(file_metadata, delimiter='\t')
        for row in reader_metadata:
            if counter == 0:
                counter += 1
                continue
            dict_tuple_to_cancer[(row[0].strip(), 'L')] = int(row[3])
            dict_tuple_to_cancer[(row[0].strip(), 'R')] = int(row[4])
    # Alright, now, let's connect those dictionaries together...
    X_tot = []
    Y_tot = []
    for img_name in dict_img_to_patside:
        X_tot.append(img_name)
        Y_tot.append(dict_tuple_to_cancer[dict_img_to_patside[img_name]])
    # Making train/val split and returning.
    X_tr, X_te, Y_tr, Y_te = train_test_split(X_tot, Y_tot, test_size=0.001)
    return X_tr, X_te, Y_tr, Y_te

def read_in_one_image(path_img, name_img, matrix_size, data_aug=False):
    """
    This is SUPER basic.  This can be improved.
    Basically, all data is stored as a .dcm.gz.
    First, we'll uncompress and save as temp.dcm.
    Then we'll read in the dcm to get to the array.
    We'll resize the image to [matrix_size, matrix_size].
    We'll also convert to a np.float32 and zero-center 1-scale the data.
    INPUTS:
    - path_img: (string) path to the data
    - name_img: (string) name of the image e.g. '123456.dcm'
    - matrix_size: (int) one dimension of the square image e.g. 224
    """
    # Setting up the filepaths and opening up the format.
    #filepath_temp = join(path_img, 'temp.dcm')
    filepath_img = join(path_img, name_img)
    # Reading/uncompressing/writing
    #if isfile(filepath_temp):
    #    remove(filepath_temp)
    #with gzip.open(filepath_img, 'rb') as f_gzip:
    #    file_content = f_gzip.read()
    #    with open(filepath_temp, 'w') as f_dcm:
    #        f_dcm.write(file_content)
    # Reading in dicom file to ndarray and processing
    dicom_content = dicom.read_file(filepath_img)
    img = dicom_content.pixel_array
    img = scipy.misc.imresize(img, (matrix_size, matrix_size), interp='cubic')
    img = img.astype(np.float32)
    img -= np.mean(img)
    img /= np.std(img)
    # Removing temporary file.
    #remove(filepath_temp)
    # Let's do some stochastic data augmentation.
    if not data_aug:
        return img
    if np.random.rand() > 0.5:                                #flip left-right
        img = np.fliplr(img)
    num_rot = np.random.choice(4)                             #rotate 90 randomly
    img = np.rot90(img, num_rot)
    up_bound = np.random.choice(174)                          #zero out square
    right_bound = np.random.choice(174)
    img[up_bound:(up_bound+50), right_bound:(right_bound+50)] = 0.0
    return img
    
def conv2d(l_input, filt_size, filt_num, stride=1, alpha=0.1, name="conv2d", norm="bn"):
    """
    A simple 2-dimensional convolution layer.
    Layer Architecture: 2d-convolution - bias-addition - batch_norm - reLU
    All weights are created with a (hopefully) unique scope.
    INPUTS:
    - l_input: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - filt_size: (int) size of the square filter to be made
    - filt_num: (int) number of filters to be made
    - stride: (int) stride of our convolution
    - alpha: (float) for the leaky ReLU.  Do 0.0 for ReLU.
    - name: (string) unique name for this convolution layer
    - norm: (string) to decide which normalization to use ("bn", "lrn", None)
    """
    # Creating and Doing the Convolution.
    input_size = l_input.get_shape().as_list()
    weight_shape = [filt_size, filt_size, input_size[3], filt_num]
    std = 0.01#np.sqrt(2.0 / (filt_size * filt_size * input_size[3]))
    with tf.variable_scope(name+"_conv_weights"):
        W = tf.get_variable("W", weight_shape, initializer=tf.random_normal_initializer(stddev=std))
    tf.add_to_collection("reg_variables", W)
    conv_layer = tf.nn.conv2d(l_input, W, strides=[1, stride, stride, 1], padding='SAME')
    # Normalization
    if norm=="bn":
        norm_layer = tflearn.layers.normalization.batch_normalization(conv_layer, name=(name+"_batch_norm"), decay=0.9)
    elif norm=="lrn":
        norm_layer = tflearn.layers.normalization.local_response_normalization(conv_layer)
    # ReLU
    relu_layer = tf.maximum(norm_layer, norm_layer*alpha)
    return relu_layer    

def max_pool(l_input, k=2, stride=None):
    """
    A simple 2-dimensional max pooling layer.
    Strides and size of max pool kernel is constrained to be the same.
    INPUTS:
    - l_input: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - k: (int) size of the max_filter to be made.  also size of stride.
    """
    if stride==None:
        stride=k
    # Doing the Max Pool
    max_layer = tf.nn.max_pool(l_input, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding='SAME')
    return max_layer

def incept(l_input, kSize=[16,16,16,16,16,16], name="incept", norm="bn"):
    """
    So, this is the classical incept layer.
    INPUTS:
    - l_input: (tensor.4d) input of size [batch_size, layer_width, layer_height, channels]
    - ksize: (array (6,)) [1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, poolproj]
    - name: (string) name of incept layer
    - norm: (string) to decide which normalization ("bn", "lrn", None)
    """
    layer_1x1 = conv2d(l_input, 1, kSize[0], name=(name+"_1x1"), norm=norm)
    layer_3x3a = conv2d(l_input, 1, kSize[1], name=(name+"_3x3a"), norm=norm)
    layer_3x3b = conv2d(layer_3x3a, 3, kSize[2], name=(name+"_3x3b"), norm=norm)
    layer_5x5a = conv2d(l_input, 1, kSize[3], name=(name+"_5x5a"), norm=norm)
    layer_5x5b = conv2d(layer_5x5a, 5, kSize[4], name=(name+"_5x5b"), norm=norm)
    layer_poola = max_pool(l_input, k=3, stride=1)
    layer_poolb = conv2d(layer_poola, 1, kSize[5], name=(name+"_poolb"), norm=norm)
    return tf.concat(3, [layer_1x1, layer_3x3b, layer_5x5b, layer_poolb])

def dense(l_input, hidden_size, keep_prob, alpha=0.1, name="dense"):
    """
    Dense (Fully Connected) layer.
    Architecture: reshape - Affine - batch_norm - dropout - relu
    WARNING: should not be the output layer.  Use "output" for that.
    INPUTS:
    - l_input: (tensor.2d or more) basically, of size [batch_size, etc...]
    - hidden_size: (int) Number of hidden neurons.
    - keep_prob: (float) Probability to keep neuron during dropout layer.
    - alpha: (float) Slope for leaky ReLU.  Set 0.0 for ReLU.
    - name: (string) unique name for layer.
    """
    # Flatten Input Layer
    input_size = l_input.get_shape().as_list()
    reshape_size = 1
    for iter_size in range(1, len(input_size)):
        reshape_size *= input_size[iter_size]
    reshape_layer = tf.reshape(l_input, [-1, reshape_size])
    # Creating and Doing Affine Transformation
    weight_shape = [reshape_layer.get_shape().as_list()[1], hidden_size]
    std = 0.01#np.sqrt(2.0 / reshape_layer.get_shape().as_list()[1])
    with tf.variable_scope(name+"_dense_weights"):
        W = tf.get_variable("W", weight_shape, initializer=tf.random_normal_initializer(stddev=std))
    tf.add_to_collection("reg_variables", W)
    affine_layer = tf.matmul(reshape_layer, W)
    # Batch Normalization
    norm_layer = tflearn.layers.normalization.batch_normalization(affine_layer, name=(name+"_batch_norm"), decay=0.9)
    # Dropout
    dropout_layer = tf.nn.dropout(norm_layer, keep_prob)
    # ReLU
    relu_layer = tf.maximum(dropout_layer, dropout_layer*alpha)
    return relu_layer

def output(l_input, output_size, name="output"):
    """
    Output layer.  Just a simple affine transformation.
    INPUTS:
    - l_input: (tensor.2d or more) basically, of size [batch_size, etc...]
    - output_size: (int) basically, number of classes we're predicting
    - name: (string) unique name for layer.
    """
    # Flatten Input Layer
    input_size = l_input.get_shape().as_list()
    reshape_size = 1
    for iter_size in range(1, len(input_size)):
        reshape_size *= input_size[iter_size]
    reshape_layer = tf.reshape(l_input, [-1, reshape_size])
    # Creating and Doing Affine Transformation
    weight_shape = [reshape_layer.get_shape().as_list()[1], output_size]
    std = 0.01#np.sqrt(2.0 / reshape_layer.get_shape().as_list()[1])
    with tf.variable_scope(name+"_output_weights"):
        W = tf.get_variable("W", weight_shape, initializer=tf.random_normal_initializer(stddev=std))
        b = tf.get_variable("b", output_size, initializer=tf.constant_initializer(0.0))
    tf.add_to_collection("reg_variables", W)
    affine_layer = tf.matmul(reshape_layer, W) + b
    return affine_layer

def get_L2_loss(reg_param, key="reg_variables"):
    """
    L2 Loss Layer. Usually will use "reg_variables" collection.
    INPUTS:
    - reg_param: (float) the lambda value for regularization.
    - key: (string) the key for the tf collection to get from.
    """
    L2_loss = 0.0
    for W in tf.get_collection(key):
        L2_loss += reg_param * tf.nn.l2_loss(W)
    return L2_loss

def get_CE_loss(logits, labels):
    """
    This calculates the cross entropy loss.
    Modular function made just because tf program name is long.
    INPUTS:
    - logits: (tensor.2d) logit probability values.
    - labels: (array of ints) basically, label \in {0,...,L-1}
    """
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

def get_accuracy(logits, labels):
    """
    Calculates accuracy of predictions.  Softmax based on largest.
    INPUTS:
    - logits: (tensor.2d) logit probability values.
    - labels: (array of ints) basically, label \in {0,...,L-1}
    """
    pred_labels = tf.argmax(logits,1)
    correct_pred = tf.equal(pred_labels, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def get_optimizer(cost, lr=0.001, decay=1.0, epoch_every=10):
    """
    Creates an optimizer based on learning rate and loss.
    We will use Adam optimizer.  This may have to change in the future.
    INPUTS:
    - cost: (tf value) usually sum of L2 loss and CE loss
    - lr: (float) the learning rate.
    - decay: (float) how much to decay each epoch.
    - epoch_every: (int) how many iterations is an epoch.
    """
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = float(lr)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   epoch_every, decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return optimizer

def Alex_conv(layer, b_name=""):
    """
    The convolution part of the classic Alex Net.
    Everything has been hardcoded to show example of use.
    INPUT:
    - layer: (tensor.4d) input tensor.
    - b_name: (string) branch name.  If not doing branch, doesn't matter.
    """
    conv1 = conv2d(layer, 11, 96, stride=4, name=b_name+"conv1")
    pool1 = max_pool(conv1, k=2)
    conv2 = conv2d(pool1, 11, 256, name=b_name+"conv2")
    pool2 = max_pool(conv2, k=2)
    conv3 = conv2d(pool2, 3, 384, name=b_name+"conv3")
    conv4 = conv2d(conv3, 3, 384, name=b_name+"conv4")
    conv5 = conv2d(conv3, 3, 256, name=b_name+"conv5")
    pool5 = max_pool(conv5, k=2)
    return pool5

def general_conv(layer, architecture_conv, b_name="", norm="bn"):
    """
    A generalized convolution block that takes an architecture.
    INPUTS:
    - layer: (tensor.4d) input tensor.
    - architecture_conv: (list of lists)
      [[filt_size, filt_num, stride], ..., [0, poolSize],
       [filt_size, filt_num, stride], ..., [0, poolSize],
       ...]
    - b_name: (string) branch name.  If not doing branch, doesn't matter.
    """
    for conv_iter, conv_numbers in enumerate(architecture_conv):
        if conv_numbers[0]==0:
            layer = max_pool(layer, k=conv_numbers[1])
        else:
            if len(conv_numbers)==2:
                conv_numbers.append(1)
            layer = conv2d(layer, conv_numbers[0], conv_numbers[1], stride=conv_numbers[2],
                           name=(b_name+"conv"+str(conv_iter)), norm=norm)
    return layer

def GoogLe_conv(layer, b_name="", norm="bn"):
    """
    This should be the convolution layers of the GoogLe net.
    We follow the v1 architecture as laid out by
    http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
    INPUTS:
    - layer: (tensor.4d) input tensor
    - b_name: (string) branch name, if necessary.
    - norm: (string) which normalization to use.
    """
    conv1 = conv2d(layer, 7, 64, stride=2, name=b_name+"conv1", norm=norm)
    pool1 = max_pool(conv1, k=3, stride=2)
    conv2a = conv2d(pool1, 1, 64, name=b_name+"conv2a", norm=norm)
    conv2b = conv2d(conv2a, 3, 192, name=b_name+"conv2b", norm=norm)
    pool2 = max_pool(conv2b, k=3, stride=2)
    incept3a = incept(pool2, kSize=[64,96,128,16,32,32], name=b_name+"incept3a", norm=norm)
    incept3b = incept(incept3a, kSize=[128,128,192,32,96,64], name=b_name+"incept3b", norm=norm)
    pool3 = max_pool(incept3b, k=3, stride=2)
    incept4a = incept(pool3, kSize=[192,96,208,16,48,64], name=b_name+"incept4a", norm=norm)
    incept4b = incept(incept4a, kSize=[160,112,224,24,64,64], name=b_name+"incept4b", norm=norm)
    incept4c = incept(incept4b, kSize=[128,128,256,24,64,64], name=b_name+"incept4c", norm=norm)
    incept4d = incept(incept4c, kSize=[112,144,288,32,64,64], name=b_name+"incept4d", norm=norm)
    incept4e = incept(incept4d, kSize=[256,160,320,32,128,128], name=b_name+"incept4e", norm=norm)
    pool4 = max_pool(incept4e, k=3, stride=2)
    incept5a = incept(pool4, kSize=[256,160,320,32,128,128], name=b_name+"incept5a", norm=norm)
    incept5b = incept(incept5a, kSize=[384,192,384,48,128,128], name=b_name+"incept5b", norm=norm)
    size_pool = incept5b.get_shape().as_list()[1]
    pool5 = tf.nn.avg_pool(incept5b, ksize=[1,size_pool,size_pool,1], strides=[1,1,1,1], padding='VALID')
    return pool5

def Le_Net(X, output_size, keep_prob=1.0, name=""):
    """
    Very Simple Lenet
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.  should be 0.4 at train.
    """
    layer = X
    conv1 = conv2d(layer, 5, 6, stride=1, name=name+"conv1", norm="bn")
    pool1 = max_pool(conv1, k=2, stride=2)
    conv2 = conv2d(pool1, 3, 16, stride=1, name=name+"conv2", norm="bn")
    pool2 = max_pool(conv2, k=2, stride=2)
    dense1 = dense(pool2, 120, keep_prob, name=name+"dense1")
    return output(dense1, output_size, name=name+"output")

def GoogLe_Net(X, output_size, keep_prob=1.0, name=""):
    """
    This is the famous GoogLeNet incarnation of the inception network.
    All the power is in the convs, so this is quite simple.
    INPUTS:
    - X: (tensor.4d) input tensor.
    - output_size: (int) number of classes we're predicting
    - keep_prob: (float) probability to keep during dropout.  should be 0.4 at train.
    """
    layer = GoogLe_conv(X, b_name=name)
    drop1 = tf.nn.dropout(layer, keep_prob)
    return output(layer, output_size, name=name+"output")

def Alex_Net(X, output_size, keep_prob=1.0, name=""):
    """
    The classic alex net architecture.
    INPUTS:
    - X: (tensor.4d) A tensor with dimensions (none, width, height, num_channels)
    - output_size: (int) The number of classes there are.
    - keep_prob: (float) Chance of keeping a neuron during dropout.
    """
    layer = X
    layer = Alex_conv(layer, b_name=name)
    dense1 = dense(layer, 4096, keep_prob, name=name+"dense1")
    dense2 = dense(dense1, 4096, keep_prob, name=name+"dense2")
    return output(dense2, output_size, name=name+"output")

def VGG16_Net(X, output_size, keep_prob=1.0):
    """
    The classic VGG16 net architecture.
    INPUTS:
    - X: (tensor.4d) A tensor with dimensions (none, width, height, num_channels)
    - output_size: (int) The number of classes there are.
    - keep_prob: (float) Chance of keeping a neuron during dropout.
    """
    architecture_conv=[[3, 64], [3, 64], [0, 2],
                       [3, 128], [3, 128], [0, 2],
                       [3, 256], [3, 256], [3, 256], [0, 2],
                       [3, 512], [3, 512], [3, 512], [0, 2],
                       [3, 512], [3, 512], [3, 512], [0, 2]]
    layer = general_conv(X, architecture_conv, b_name=name)
    layer = dense(layer, 4096, keep_prob, name=name+"dense1")
    layer = dense(layer, 4096, keep_prob, name=name+"dense2")
    return output(layer, output_size, name=name+"output")

def test_out(sess, list_dims, list_placeholders, list_operations, X_te, opts):
    """
    This code is to call a test on the validation set.
    INPUTS:
    - sess: (tf session) the session to run everything on
    - list_dim: (list of ints) list of dimensions
    - list_placeholders: (list of tensors) list of the placeholders for feed_dict
    - list_operations: (list of tensors) list of operations for graph access
    - X_tr: (list of strings) list of training sample names
    - opts: (parsed arguments)
    """
    # Let's unpack the lists
    matrix_size, num_channels = list_dims
    x, y, keep_prob = list_placeholders
    prob, pred, saver, L2_loss, CE_loss, cost, optimizer, accuracy, init = list_operations
    # Initializing what to put in.
    dataXX = np.zeros((1, matrix_size, matrix_size, num_channels), dtype=np.float32)
    # Running through the images.
    f = open(opts.outtxt, 'w')
    statement = 'subjectID' + '\t' + 'laterality' + '\t' + 'prediction'
    super_print(statement, f)
    for iter_data in range(len(X_te)):
        id_iter, lat_iter, img_iter = X_te[iter_data]
        dataXX[0, :, :, 0] = read_in_one_image(opts.path_data, img_iter, matrix_size)
        tflearn.is_training(False)
        pred_iter = sess.run(prob, feed_dict={x: dataXX, keep_prob: 1.0})
        statement = id_iter + '\t' + lat_iter + '\t' + str(pred_iter[0][1])
        super_print(statement, f)
        #left_img, right_img = X_te[iter_data]
        #dataXX[0, :, :, 0] = read_in_one_image(opts.path_data, left_img, matrix_size)
        #tflearn.is_training(False)
        #pred_left = sess.run(pred, feed_dict={x: dataXX, keep_prob: 1.0})
        #dataXX[0, :, :, 0] = read_in_one_image(opts.path_data, right_img, matrix_size)
        #pred_right = sess.run(pred, feed_dict={x: dataXX, keep_prob: 1.0})
        #statement = str(pred_left) + '\t' + str(pred_right)
        #super_print(statement, f)
    f.close()

def test_all(sess, list_dims, list_placeholders, list_operations, X_te, Y_te, opts):
    """
    This code is to call a test on the validation set.
    INPUTS:
    - sess: (tf session) the session to run everything on
    - list_dim: (list of ints) list of dimensions
    - list_placeholders: (list of tensors) list of the placeholders for feed_dict
    - list_operations: (list of tensors) list of operations for graph access
    - X_tr: (list of strings) list of training sample names
    - Y_tr: (list of ints) list of lables for training samples
    - opts: (parsed arguments)
    """
    # Let's unpack the lists.
    matrix_size, num_channels = list_dims
    x, y, keep_prob = list_placeholders
    prob, pred, saver, L2_loss, CE_loss, cost, optimizer, accuracy, init = list_operations
    # Initializing what to put in.
    loss_te = 0.0
    acc_te = 0.0
    dataXX = np.zeros((1, matrix_size, matrix_size, num_channels), dtype=np.float32)
    dataYY = np.zeros((1, ), dtype=np.int64)
    # Running through all test data points
    v_TP = 0.0
    v_FP = 0.0
    v_FN = 0.0
    v_TN = 0.0
    for iter_data in range(len(X_te)):
        # Reading in the data
        dataXX[0, :, :, 0] = read_in_one_image(opts.path_data, X_te[iter_data], matrix_size)
        dataYY[0] = Y_te[iter_data]
        tflearn.is_training(False)
        loss_iter, acc_iter = sess.run((cost, accuracy), feed_dict={x: dataXX, y: dataYY, keep_prob: 1.0})
        # Figuring out the ROC stuff
        if Y_te[iter_data] == 1:
            if acc_iter == 1:
                v_TP += 1.0 / len(X_te)
            else:
                v_FN += 1.0 /len(X_te)
        else:
            if acc_iter == 1:
                v_TN += 1.0 /len(X_te)
            else:
                v_FP += 1.0 /len(X_te)
        # Adding to total accuracy and loss
        loss_te += loss_iter / len(X_te)
        acc_te += acc_iter / len(X_te)
    return (loss_te, acc_te, [v_TP, v_FP, v_TN, v_FN])

def train_one_iteration(sess, list_dims, list_placeholders, list_operations, X_tr, Y_tr, opts):
    """
    Basically, run one iteration of the training.
    INPUTS:
    - sess: (tf session) the session to run everything on
    - list_dim: (list of ints) list of dimensions
    - list_placeholders: (list of tensors) list of the placeholders for feed_dict
    - list_operations: (list of tensors) list of operations for graph access
    - X_tr: (list of strings) list of training sample names
    - Y_tr: (list of ints) list of lables for training samples
    - opts: (parsed arguments)
    """
    # Let's unpack the lists.
    matrix_size, num_channels = list_dims
    x, y, keep_prob = list_placeholders
    prob, pred, saver, L2_loss, CE_loss, cost, optimizer, accuracy, init = list_operations
    # Initializing what to put in.
    dataXX = np.zeros((opts.bs, matrix_size, matrix_size, num_channels), dtype=np.float32)
    dataYY = np.zeros((opts.bs, ), dtype=np.int64)
    ind_list = np.random.choice(range(len(X_tr)), opts.bs, replace=False)
    # Fill in our dataXX and dataYY for training one batch.
    for iter_data,ind in enumerate(ind_list):
        dataXX[iter_data, :, :, 0] = read_in_one_image(opts.path_data, X_tr[ind], matrix_size, data_aug=False)
        dataYY[iter_data] = Y_tr[ind]
    tflearn.is_training(True)
    _, loss_iter, acc_iter = sess.run((optimizer, cost, accuracy), feed_dict={x: dataXX, y: dataYY, keep_prob: opts.dropout})
    return (loss_iter, acc_iter)

def train_net(X_tr, X_te, Y_tr, Y_te, opts, f):
    """
    Training of the net.  All we need is data names and parameters.
    INPUTS:
    - X_tr: (list of strings) training image names
    - X_te: (list of strings) validation image names
    - Y_tr: (list of ints) training labels
    - Y_te: (list of ints) validation labels
    - opts: parsed argument thing
    - f: (opened file) for output writing
    """
    # Setting the size and number of channels of input.
    matrix_size = opts.matrix_size
    num_channels = 1
    list_dims = [matrix_size, num_channels]
    # Finding out other constant values to be used.
    data_count = len(X_tr)
    iter_count = int(np.ceil(float(opts.epoch) * data_count / opts.bs))
    epoch_every = int(np.ceil(float(iter_count) / opts.epoch))
    print_every = min([100, epoch_every])
    max_val_acc = 0.0
    # Creating Placeholders
    x = tf.placeholder(tf.float32, [None, matrix_size, matrix_size, num_channels])
    y = tf.placeholder(tf.int64)
    keep_prob = tf.placeholder(tf.float32)
    list_placeholders = [x, y, keep_prob]
    # Create the network
    if opts.net == "Alex":
        pred = Alex_Net(x, 2, keep_prob=keep_prob)
    elif opts.net == "Le":
        pred = Le_Net(x, 2, keep_prob=keep_prob)
    elif opts.net == "VGG16":
        pred = VGG16_Net(x, 2, keep_prob=keep_prob)
    elif opts.net == "GoogLe":
        pred = GoogLe_Net(x, 2, keep_prob=keep_prob)
    else:
        statement = "Please specify valid network (e.g. Alex, VGG16, GoogLe)."
        super_print(statement, f)
        return 0
    # Define Operations in TF Graph
    saver = tf.train.Saver()
    L2_loss = get_L2_loss(opts.reg)
    CE_loss = get_CE_loss(pred, y)
    cost = L2_loss + CE_loss
    prob = tf.nn.softmax(pred)
    optimizer = get_optimizer(cost, lr=opts.lr, decay=opts.decay, epoch_every=epoch_every)
    accuracy = get_accuracy(pred, y)
    init = tf.initialize_all_variables()
    list_operations = [prob, pred, saver, L2_loss, CE_loss, cost, optimizer, accuracy, init]
    # Do the Training
    print "Training Started..."
    start_time = time.time()
    with tf.Session() as sess:
        sess.run(init)
        loss_tr = 0.0
        acc_tr = 0.0
        if opts.test:
            saver.restore(sess, opts.saver)
            test_out(sess, list_dims, list_placeholders, list_operations, X_te, opts)
            return 0
        for iter in range(iter_count):
            loss_temp, acc_temp = train_one_iteration(sess, list_dims, list_placeholders, list_operations, X_tr, Y_tr, opts)
            loss_tr += loss_temp / print_every
            acc_tr += acc_temp / print_every
            if ((iter)%print_every) == 0:
                current_time = time.time()
                loss_te, acc_te, ROC_values = test_all(sess, list_dims, list_placeholders, list_operations, X_te, Y_te, opts)
                # Printing out stuff
                statement = "    Iter"+str(iter+1)+": "+str((current_time - start_time)/60)
                statement += ", Acc_tr: "+str(acc_tr)
                statement += ", Acc_val: "+str(acc_te)
                statement += ", Loss_tr: "+str(loss_tr)
                statement += ", Loss_val: "+str(loss_te)
                super_print(statement, f)
                statement = "      True_Positive: "+str(ROC_values[0])
                statement += ", False_Positive: "+str(ROC_values[1])
                statement += ", True_Negative: "+str(ROC_values[2])
                statement += ", False_Negative: "+str(ROC_values[3])
                super_print(statement, f)
                loss_tr = 0.0
                acc_tr = 0.0
                if acc_te > max_val_acc:
                    max_val_acc = acc_te
                    saver.save(sess, opts.saver)
                if (current_time - start_time)/60 > opts.time:
                    break
    statement = "Best you could do: " + str(max_val_acc)
    super_print(statement, f)
    return 0

def main(args):
    """
    Main Function to do deep learning using tensorflow on pilot.
    INPUTS:
    - args: (list of strings) command line arguments
    """
    # Setting up reading of command line options, storing defaults if not provided.
    parser = argparse.ArgumentParser(description = "Do deep learning!")
    parser.add_argument("--pf", dest="path_data", type=str, default="/trainingData")
    parser.add_argument("--csv1", dest="csv1", type=str, default="/metadata/images_crosswalk.tsv")
    parser.add_argument("--csv2", dest="csv2", type=str, default="/metadata/exams_metadata.tsv")
    parser.add_argument("--csv3", dest="csv3", type=str, default="/scoringData/image_metadata.tsv")
    parser.add_argument("--net", dest="net", type=str, default="GoogLe")
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--reg", dest="reg", type=float, default=0.00001)
    parser.add_argument("--out", dest="output", type=str, default="/modelState/out_train.txt")
    parser.add_argument("--outtxt", dest="outtxt", type=str, default="/output/out.txt")
    parser.add_argument("--saver", dest="saver", type=str, default="/modelState/model.ckpt")
    parser.add_argument("--decay", dest="decay", type=float, default=1.0)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
    parser.add_argument("--bs", dest="bs", type=int, default=10)
    parser.add_argument("--epoch", dest="epoch", type=int, default=10)
    parser.add_argument("--test", dest="test", type=int, default=0)
    parser.add_argument("--ms", dest="matrix_size", type=int, default=224)
    parser.add_argument("--time", dest="time", type=float, default=1000000)
    opts = parser.parse_args(args[1:])
    # Setting up the output file.
    if isfile(opts.output):
        remove(opts.output)
    f = open(opts.output, 'w')
    # Finding list of data.
    statement = "Parsing the csv's."
    super_print(statement, f)
    path_csv_crosswalk = opts.csv1
    path_csv_metadata = opts.csv2
    path_csv_test = opts.csv1
    if opts.test:
        X_tr, X_te, Y_tr, Y_te = create_test_splits(path_csv_test)
    else:
        X_tr, X_te, Y_tr, Y_te = create_data_splits(path_csv_crosswalk, path_csv_metadata)
    # Train a network and print a bunch of information.
    statement = "Let's start the training!"
    super_print(statement, f)
    statement = "Network: " + opts.net + ", Dropout: " + str(opts.dropout) + ", Reg: " + str(opts.reg) + ", LR: " + str(opts.lr) + ", Decay: " + str(opts.decay)
    super_print(statement, f)
    train_net(X_tr, X_te, Y_tr, Y_te, opts, f)
    f.close()
    return 0

if __name__ == '__main__':
    main(sys.argv)
