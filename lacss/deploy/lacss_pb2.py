# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lacss.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0blacss.proto\x12\x0ftrackmate.lacss\"l\n\x05Image\x12\x0e\n\x06height\x18\x01 \x01(\x04\x12\r\n\x05width\x18\x02 \x01(\x04\x12\x0f\n\x07\x63hannel\x18\x03 \x01(\x04\x12%\n\x05\x64type\x18\x04 \x01(\x0e\x32\x16.trackmate.lacss.DType\x12\x0c\n\x04\x64\x61ta\x18\x05 \x01(\x0c\"\x9d\x01\n\x08Settings\x12\x15\n\rmin_cell_area\x18\x01 \x01(\x02\x12\x1b\n\x13remove_out_of_bound\x18\x02 \x01(\x08\x12\x0f\n\x07scaling\x18\x03 \x01(\x02\x12\x0f\n\x07nms_iou\x18\x04 \x01(\x02\x12\x1b\n\x13\x64\x65tection_threshold\x18\x05 \x01(\x02\x12\x1e\n\x16segmentation_threshold\x18\x06 \x01(\x02\"[\n\x05Input\x12+\n\x08settings\x18\x01 \x01(\x0b\x32\x19.trackmate.lacss.Settings\x12%\n\x05image\x18\x02 \x01(\x0b\x32\x16.trackmate.lacss.Image\"4\n\x05Label\x12\x0e\n\x06height\x18\x01 \x01(\x04\x12\r\n\x05width\x18\x02 \x01(\x04\x12\x0c\n\x04\x64\x61ta\x18\x06 \x01(\x0c\"V\n\x06Result\x12%\n\x05score\x18\x01 \x01(\x0b\x32\x16.trackmate.lacss.Label\x12%\n\x05label\x18\x02 \x01(\x0b\x32\x16.trackmate.lacss.Label*\x14\n\x05\x44Type\x12\x0b\n\x07\x46LOAT32\x10\x00\x42\'\n\x1b\x66iji.plugin.trackmate.lacssB\x08LacssMsgb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'lacss_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\n\033fiji.plugin.trackmate.lacssB\010LacssMsg'
  _globals['_DTYPE']._serialized_start=537
  _globals['_DTYPE']._serialized_end=557
  _globals['_IMAGE']._serialized_start=32
  _globals['_IMAGE']._serialized_end=140
  _globals['_SETTINGS']._serialized_start=143
  _globals['_SETTINGS']._serialized_end=300
  _globals['_INPUT']._serialized_start=302
  _globals['_INPUT']._serialized_end=393
  _globals['_LABEL']._serialized_start=395
  _globals['_LABEL']._serialized_end=447
  _globals['_RESULT']._serialized_start=449
  _globals['_RESULT']._serialized_end=535
# @@protoc_insertion_point(module_scope)