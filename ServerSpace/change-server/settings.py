# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
IMAGE_CHANS = 3
DATA_TYPE = "float32"

# V2 model path
# ModelPath_changedetect = "savemodel/V2_change-model-v9_0.9984_wudi.h5"

# V3 model path 三代机矮层模型
# ModelPath_changedetect = "savemodel/V3_low-height_change_model-v9_0.9969_wudi.h5"

# V3 model path 三代机高层模型
ModelPath_changedetect = "savemodel/V3_high-height_change_model-v9_0.9961_wudi.h5"


# initialize constants used for server queuing
GPU_COUNT = 2
QUEUE_GPUS_CD = ["gpu" + str(i) + "_r1" for i in range(GPU_COUNT)]

# batch size and sleep time
BATCH_SIZE = 5
SERVER_SLEEP = 0.05
CLIENT_SLEEP = 0.05
GET_MAX = 100

# threshold of model
Threshold_Change_Detect = 0.75
