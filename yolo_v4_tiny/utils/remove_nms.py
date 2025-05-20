# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import onnx_graphsurgeon as gs
import numpy as np
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


input_model = '../export/yolov4_cspdarknet_tiny_epoch_015.onnx'
input_shape = (1,3,256,256)
target_opset = 17

# load the ONNX model
model = onnx.load(input_model)
graph = gs.import_onnx(model)

# get tensors from the graph
tensors = graph.tensors()

# set the correct input shape
graph.inputs = [tensors["Input"].to_variable(dtype=np.float32, shape=input_shape)]

# set outputs before BatchedNMS
graph.outputs = [tensors["box"].to_variable(dtype=np.float32), # bounding boxes
                 tensors["cls"].to_variable(dtype=np.float32)] # class probabilities

# # fix the dynamic shapes
# graph.inp
graph.cleanup()


# create a model from the new graph
model_wo_pp = gs.export_onnx(graph)

# set the intermediate representation version to 9
model_wo_pp.ir_version = 8

# running symbolic inference to have the inference shapes assigned to nodes
model_wo_pp = SymbolicShapeInference.infer_shapes(
                                            model_wo_pp,
                                            2**31-1,
                                            False,
                                            False,
                                            0)

# sanitize onnx opset imports
del model_wo_pp.opset_import[:]
opset = model_wo_pp.opset_import.add()
opset.domain = ''
opset.version = target_opset

# saving the model 
onnx.save_model(model_wo_pp, f'{input_model[:-5]}_no_nms.onnx')