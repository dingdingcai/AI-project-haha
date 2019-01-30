# import normal packages
import os
import json
import time
import redis
import numpy as np
import pickle
# import my packages
import settings
from DepthwiseConv2D import DepthwiseConv2D
from switchnorm import SwitchNormalization

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
    # load models
    print("* Loading models...")
    model_point_change = load_model(settings.ModelPath_point_change,
                                    custom_objects={'DepthwiseConv2D': DepthwiseConv2D,
                                                    'SwitchNormalization': SwitchNormalization}, compile=False)
    print("1 point_change model loaded from {}".format(settings.ModelPath_point_change))

    model_all_count = load_model(settings.ModelPath_all_count,
                                 custom_objects={'DepthwiseConv2D': DepthwiseConv2D,
                                                 'SwitchNormalization': SwitchNormalization}, compile=False)
    print("2 all_count model loaded from {}".format(settings.ModelPath_all_count))
    print("Load Model Complete!")
    # continually pool for new request
    while True:
        # time
        loop_start_time = time.time()
        # Queue of take_goods
        queue_r1 = db.lrange(settings.QUEUE_GPUS_R1[gpu_index], 0, settings.BATCH_SIZE - 1)
        batch_size = len(queue_r1)
        # check to see if the batch list is None
        if batch_size > 0:
            # time
            start_time = time.time()
            requestIDs = []
            batch_a = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            batch_b = np.zeros(((batch_size,) + image_shape), dtype=data_type)
            batch_b2 = np.zeros(((batch_size,) + (480, 640, 3)), dtype=data_type)
            batch_mask_a_all = np.zeros((batch_size, settings.HEATMAP_HEIGHT, settings.HEATMAP_WIDTH, 1), dtype=np.uint8)
            # loop over the queue
            for i, q in enumerate(queue_r1):
                q = pickle.loads(q)
                # deserialize the object and obtain the input image
                image_a = db.get(q["image_c_key"])
                if image_a is not None:
                    batch_a[i] = pickle.loads(image_a)
                image_b = db.get(q["image_b_key"])
                if image_b is not None:
                    batch_b[i] = pickle.loads(image_b)
                # temp for all count
                image_b2 = db.get(q["image_b2_key"])
                if image_b2 is not None:
                    batch_b2[i] = pickle.loads(image_b2)
                mask_a = db.get(q["pointmap_c_key"])
                if mask_a is not None:
                    batch_mask_a_all[i] = pickle.loads(mask_a)
                # update the list of image IDs
                requestIDs.append(q["request_id_diy"])
            # do things of single_abnormal detect model
            time_0 = time.time()
            # normal = 0, abnormal = 1
            preds_point_change = model_point_change.predict_on_batch([batch_a, batch_b, batch_mask_a_all, batch_mask_a_all])
            time_1 = time.time()
            # do things of single count model twice
            preds_all_count = model_all_count.predict_on_batch(batch_b2)
            time_2 = time.time()
            # loop over the request IDs and their corresponding set of
            # results from our model
            for i in range(batch_size):
                out = {"point_change": preds_point_change[i],
                       "all_count": preds_all_count[i]}
                db.set(requestIDs[i], pickle.dumps(out))
            # remove the set of images from our queue
            db.ltrim(settings.QUEUE_GPUS_R1[gpu_index], len(requestIDs), -1)
            # time
            run_time = time.time() - start_time
            print("* take_goods, Batch size: {0}, use time: {1:.3f}, point_change time : {2:.3f}, "
                  "all_count time : {3:.3f}"
                  .format(batch_size, run_time, time_1 - time_0, time_2 - time_1))

        # time
        loop_time = time.time() - loop_start_time
        # sleep for rest time
        if loop_time < settings.SERVER_SLEEP:
            time.sleep(settings.SERVER_SLEEP - loop_time)


# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
    predict_process()
