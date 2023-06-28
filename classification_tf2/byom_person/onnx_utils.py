# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import tensorflow as tf
import numpy as np
import os

import onnxruntime as rt
from onnx import ModelProto
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import onnx
#====================== onnx evaluation =======================#
# parameters to be used for the normalization step if 'torch' is selected as preproc_mode
# these numbers are coming from 'imagenet' dataset
torch_means = [0.485,0.456,0.406]
torch_std = [0.224, 0.224, 0.224]

def get_model(onnx_model_path):
    '''
    get_model(onnx_model_path)
    function reads onnx model and returns it along with the onnxruntime inference session
        input:
            onnx_model_path : (str) full path to the onnx model "xyz/model.onnx"
        outputs:
            onx: onnx model
            sess: onnxruntime inference session object to perform the inference
    '''
    onx = ModelProto()
    with open(onnx_model_path, mode = 'rb') as f:
        content = f.read()
        onx.ParseFromString(content)
    sess = rt.InferenceSession(onnx_model_path)
    return onx, sess

def predict_onnx(sess, data):
    '''
    predict_onnx(sess, data)
    function runs the inference using the provided data and the onnxruntime inference session and returns the predictions
        inputs:
            sess: onnxruntime inference session obtained from the get_model(onnx_model_path) function
            data: input data to run the inference on (numpy array)
                    the data should have the same input shape as the model
        outputs:
            onx_pred: prediction results of the onnx model on the provided input data'''

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onx_pred = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
    return onx_pred

def plot_confusion_matrix(cm, class_labels, save_path = "float_model", test_accuracy = '_'):
    '''
    plot_confusion_matrix(cm, class_labels, save_path, test_accuracy)
    function prints the confusion_matrix, plots it, and save the plot as a png file

    inputs:
        cm: (nxn) shape confusion matrix, where n is num_classes
        class_labels: (an array of str) list of class labels
        save_path: (str) path where to save the resulting confusion matrix plot
        test_accuracy: (str) containing the accuracy to print on the confusion matrix title.
    '''
    print(f'confusion_matrix : \n{cm}')
    cm_normalized = [element/sum(row) for element, row in zip([row for row in cm], cm)]
    cm_normalized = np.array(cm_normalized)
    
    plt.figure(figsize = (4,4))
    
    disp = ConfusionMatrixDisplay(cm_normalized)
    disp.plot(cmap = "Blues")
    plt.title(f'Model_accuracy : {test_accuracy}', fontsize = 10)
    plt.tight_layout(pad=3)
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.xticks(np.arange(0, len(class_labels)), class_labels)
    plt.yticks(np.arange(0, len(class_labels)), class_labels)
    plt.rcParams.update({'font.size': 14})
    plt.show()
    plt.savefig(f'{save_path}_confusion-matrix.png')

def get_preprocessed_image(image_path, width = 128, height = 128, preproc_mode = 'tf', interpolation = 'bilinear'):
    '''
    get_preprocessed_image(image_path, width, height, preproc_mode, interpolation)
    function takes the path of the image as input and returns the preprocessed image in the form of a numpy array of size (nchw)
    
    inputs:
        image_path: (str) path to the image file to be loaded and preprocessed
        width: (int) width of the output image (if original width is different the image will be resized)
        height: (int) height of the output image (if original width is different the image will be resized)
        preproc_mode: (str) preprocessing mode, 'tf' or 'torch'
                      'tf' will bring the image in the range of [-1,+1], preproc_image_pixels = -1 + (image_pixels/127.5)
                      'torch' will apply double normalization based on imagenet data, 
                                _image_pixels = (image_pixels/255.)
                                preproc_image_pixels = (_image_pixels - torch_mean)/torch_std
        interpolation: (str) interpolation method, supported values are 'bilinear', 'nearest'
    '''
    img = tf.keras.utils.load_img(image_path, grayscale = False, color_mode = 'rgb',
     target_size = (width,height), interpolation=interpolation)
    img_array = np.array([tf.keras.utils.img_to_array(img)])
    if preproc_mode.lower() == 'tf':
        img_array = -1 + img_array / 127.5
    elif preproc_mode.lower() == 'torch':
        img_array = img_array / 255.0
        img_array = (img_array - torch_means)
        img_array = img_array/ torch_std
    else:
        raise Exception('Only \'tf\' or \'torch\' preprocessings are supported.')
    
    img_array = img_array.transpose((0,3,1,2))
    return img_array

def evaluate_onnx_model(onnx_model_path, test_dir, img_width = 128, img_height = 128, save_path='float_onnx', preproc_mode = 'tf', interpolation = 'bilinear'):
    '''
    evaluate_onnx_model(onnx_model_path, test_dir, img_width, img_height, save_path, preproc_mode, interpolation)
    function evaluates an onnx model at path onnx_model_path using the data available in test_dir, creates a confusion matrix and saves it at the save_path

    inputs:
        onnx_model_path: (str) path of the onnx model to evaluate
        test_dir: (str) path to the test dataset (one subdir per class with all images in them belonging to that class)
        img_widht: width of the image to be used for the preprocessing before passing to model
        img_height: height of the image to be used for the preprocessing before passing to model
        save_path: (str) path to save the confusion matrix (example: ./results_mobilenetv2/exports/model_name [without.png])
        preproc_mode: (str) preprocessing mode, supported values are 'tf', 'torch'
        interpolation: (str) interpolation type, supported types 'bilinear', 'nearest'
    outputs:
        test_acc: test accuracy (float number)
        test_cm: test confusion matrix (nxn matrix)

    '''
    _, sess = get_model(onnx_model_path)

    gt_labels = []
    prd_labels = np.empty((0))
    class_labels = sorted(os.listdir(test_dir))
    for i in range(len(class_labels)):
        class_label = class_labels[i]
        
        for file in os.listdir(os.path.join(test_dir, class_label)):
            gt_labels.append(i)
            image_path = os.path.join(test_dir,class_label,file)
            img = get_preprocessed_image(image_path, width = img_width, height = img_height, 
            preproc_mode = preproc_mode, interpolation = interpolation)
            
            # predicting the results on the batch
            pred = predict_onnx(sess, img).argmax(axis = 1)
            prd_labels = np.concatenate((prd_labels, pred))

    test_acc = round(accuracy_score(gt_labels, prd_labels), 6)
    print(f'Evaluation Top 1 accuracy : {test_acc}')
    test_cm = confusion_matrix(gt_labels,prd_labels)

    plot_confusion_matrix(test_cm, class_labels = class_labels, save_path = save_path, test_accuracy = test_acc)

    return test_acc, test_cm

# ============================   Onnx Quantization ============================== #
from datetime import datetime
import time
import onnxruntime.quantization as quantization
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, CalibrationDataReader


def preprocess_image_batch(images_folder: str, height: int, width: int, preproc_mode = 'tf', interpolation = 'bilinear', size_limit=0):
    """
    preprocess_image_batch(images_folder, height, width, preproc_mode, interpolation, size_limit)
    function loads a batch of images and preprocess them
    
    inputs:
        images_folder: (str) path to folder storing images
        height: (int) image height in pixels
        width: (int) image width in pixels
        preproc_mode: (str) preprocessing type, supported options are 'tf', 'torch'
        interpolation: (str) interpolation method, supported values are 'bilinear', or 'nearest'
        size_limit: (int) number of images to load. Default is 0 which means all images are picked.
    ouputs:
        bathc_data: a numpy array as a matrix characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        img = get_preprocessed_image(image_filepath, width = width, height = height, 
            preproc_mode = preproc_mode, interpolation = interpolation)
        unconcatenated_batch_data.append(img)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class ImageDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, preproc_mode: str, interpolation: str):
        '''
        ImageDataReader class creates a ImageDataReader object to pass to quantiz_static function for onnx model quantization
        inputs:
            calibration_image_folder: (str) a dataset to be used for performing the quantization (subset of the original training dataset)
            model_path: (str) path of the model to be evaluated
            preproc_mode: (str) preprocessing mode, supported 'tf' or 'torch'
            interpolation: (str) interpolation method, supported values are 'bilinear' and 'nearest'
        '''
        self.enum_data = None

        # Use inference session to get input shape.
        session = rt.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = preprocess_image_batch(
            calibration_image_folder, height, width, preproc_mode, interpolation, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def onnx_benchmark(model_path):
    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    (_, _, height, width) = session.get_inputs()[0].shape
    total = 0.0
    runs = 10
    input_data = np.zeros((1, 3, height, width), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        # print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")

def quantize_onnx_model(input_model, calibration_dataset_path, preproc_mode = 'tf', interpolation='bilinear'):
    '''
    quantize_onnx_model(input_model, calibration_dataset_path, preproc_mode, interpolation)
    function quantizes the input_model using the calibration_dataset_path using quantize-dequantize method

    inputs:
        input_model: (str) path of the onnx_model which is to be quantized using onnxruntime
        calibration_dataset_path: (str) path to the calibration dataset which will be used to quantize the model (a subset of the training dataset)
                                Contains n sub-directories, where n is number of classes and each sub-directory contains all the iamges for the class
        preproc_mode: (str) preprocessing mode, supported options are 'tf' or 'torch'
        interpolation: (str) interpolation method: supported options are 'bilinear', and 'nearest'

    '''

    if not input_model.endswith('.onnx'):
        raise Exception("Error! The model must be in onnx format")    
    
    # set the data reader pointing to the representative dataset
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time + ' - ' + 'Prepare the data reader for the representative dataset...')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time + ' - ' + 'Found a model to be quantized: {}'.format(os.path.basename(input_model)))
    
    # set the data reader pointing to the representative dataset
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + ' - ' + 'Prepare the data reader for the representative dataset...')
    
    image_datareader = ImageDataReader(calibration_dataset_path, input_model, preproc_mode, interpolation)       
    
    # prepare quantized onnx model filename
    quant_model = os.path.splitext(input_model)
    if not calibration_dataset_path is None:
        quant_model = quant_model[0] + '_QDQ_quant' + quant_model[1]
    else:
        quant_model = quant_model[0] + '_QDQ_fakequant' + quant_model[1]
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time + ' - ' + 'Quantize the model {}, please wait...'.format(os.path.basename(input_model)))

    # prepare quantized onnx model filename
    quant_model = os.path.splitext(input_model)
    if not calibration_dataset_path is None:
        quant_model = quant_model[0] + '_QDQ_quant' + quant_model[1]
    else:
        quant_model = quant_model[0] + '_QDQ_fakequant' + quant_model[1]
    # Calibrate and quantize model
    # Turn off model optimization during quantization
    infer_model = os.path.splitext(input_model)
    infer_model = infer_model[0] + '_infer' + infer_model[1]
    quantization.quant_pre_process(input_model_path=input_model, output_model_path=infer_model)
    quantize_static(
        infer_model,
        quant_model,
        image_datareader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type = QuantType.QInt8,
        optimize_model=True,
        reduce_range=True
    )
    print(current_time + ' - ' + '{} model has been created'.format(os.path.basename(quant_model)))

    print("benchmarking fp32 model...")
    onnx_benchmark(input_model)

    print("benchmarking int8 model...")
    onnx_benchmark(quant_model)
    
    # delete the temp files
    os.remove(infer_model)

def change_opset(model_path):
    '''
    change_opset(model_path)
    function updates the opset of the exported onnx model in a way that this can be used in STM32Cube.AI

    inputs:
        model_path: (str) onnx model path
                    the updated model over-writes the input model
    '''
    m = onnx.load(model_path)
    onnx_opset = -1
    for entry in m.opset_import:
        if entry.domain == 'ai.onnx' or entry.domain == '':  # empty string == onnx domain
            onnx_opset = entry.version
    for entry in m.opset_import:
        if entry.domain == 'com.ms.internal.nhwc':
            entry.version = onnx_opset
    onnx.save(m, model_path)