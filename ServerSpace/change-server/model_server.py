# import normal packages
import os
import json
import time
import redis
import numpy as np
import pickle
# import my packages
import settings
# choose one of the gpus
with open("gpu_setting.json", 'r') as load_f:
    load_dict = json.load(load_f)
    gpu_index = load_dict["gpu_now"]
    if gpu_index >= settings.GPU_COUNT - 1:
        load_dict["gpu_now"] = 0
    else:
        load_dict["gpu_now"] = gpu_index + 1
with open("gpu_setting.json", 'w') as dump_f:
    json.dump(load_dict, dump_f)
# gpu visible setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
# keras
from keras.models import load_model
# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)
# other
image_shape = (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, settings.IMAGE_CHANS)
data_type = settings.DATA_TYPE

def predict_process():
    # load the pre-trained model
    print("* Loading models...")
    model_changedetect = load_model(settings.ModelPath_changedetect)
    print("* change_detect model loaded")
    # continually pool for new request
    while True:
        # time
        loop_start_time = time.time()
        # queue of take_goods
        queue_cd = db.lrange(settings.QUEUE_GPUS_CD[gpu_index], 0, settings.BATCH_SIZE - 1)
        batch_size = len(queue_cd)
        # check to see if the batch list is None
        if batch_size > 0:
            # time
            start_time = time.time()
            requestIDs = []
            batch_a = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            batch_b = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            # loop over the queue
            for i, q in enumerate(queue_cd):
                q = pickle.loads(q)
                image_a = db.get(q["image_a_key"])
                if image_a is not None:
                    batch_a[i] = pickle.loads(image_a)
                image_b = db.get(q["image_b_key"])
                if image_b is not None:
                    batch_b[i] = pickle.loads(image_b)
                # update the list of image IDs
                requestIDs.append(q["request_id_diy"])
            preds_changedetect = model_changedetect.predict_on_batch([batch_a, batch_b])
            # loop over the request IDs and their corresponding set of
            # results from our model
            for i in range(batch_size):
                out = {"pred_change": preds_changedetect[i]}
                # single count result
                db.set(requestIDs[i], pickle.dumps(out))
            # remove the set of images from our queue
            db.ltrim(settings.QUEUE_GPUS_CD[gpu_index], len(requestIDs), -1)
            # time
            run_time = time.time() - start_time
            print("* change_detect, Batch size: {0}, use time: {1:.3f}".format(batch_size, run_time))
        # time
        loop_time = time.time() - loop_start_time
        # sleep for rest time
        if loop_time < settings.SERVER_SLEEP:
            time.sleep(settings.SERVER_SLEEP - loop_time)


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    predict_process()
