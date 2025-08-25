python -m grpc_tools.protoc \
    -I./grpc-model-service/proto \
    --python_out=./grpc-model-service/src \
    --grpc_python_out=./grpc-model-service/src \
    ./grpc-model-service/proto/model_service.proto