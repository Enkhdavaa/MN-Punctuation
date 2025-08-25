from concurrent import futures

# Import the generated classes
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../proto'))

import model_service_pb2
import model_service_pb2_grpc

from run_puntuation_model.punctuationmode import PunctuationModel

class ModelServiceServicer(model_service_pb2_grpc.ModelServiceServicer):

    def __init__(self):
        self.model = PunctuationModel()

    def RestorePunctuation(self, request, context):
        text = request.text
        restored_text = self.model.restore_punctuation(text)
        return model_service_pb2.PunctuationResponse (restored_text=restored_text)