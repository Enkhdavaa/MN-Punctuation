from concurrent import futures

import sys
import os

sys.path.append(os.path.dirname(__file__))

import model_service_pb2
import model_service_pb2_grpc

from punctuation import PunctuationModel

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self):
        self.model = PunctuationModel()

    def RestorePunctuation(self, request, context):
        text = request.text
        punctuated_text = self.model.run_model(text)
        return model_service_pb2.PunctuationResponse(restored_text=punctuated_text) # type: ignore