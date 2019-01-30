# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANS = 3
HEATMAP_HEIGHT = 63
HEATMAP_WIDTH = 63
DATA_TYPE = "float32"

# High model path
ModelPath_point_change = "savemodel/V3_high-height_point_change_model_v5_0.9423_model.h5"
ModelPath_all_count = "savemodel/ V3_high-height_all_count_model_v12_0.9821.h5"


# initialize constants used for server queuing
GPU_COUNT = 1
QUEUE_GPUS_R1 = ["gpu" + str(i) + "_r1" for i in range(GPU_COUNT)]


# batch size and sleep time
BATCH_SIZE = 5
SERVER_SLEEP = 0.15
CLIENT_SLEEP = 0.15
GET_MAX = 100

# threshold of model
Threshold_point_change_one = 0.5
Threshold_point_change_min = 0.65
Threshold_all_count = 0.65
