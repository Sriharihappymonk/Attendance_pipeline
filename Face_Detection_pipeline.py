# gstreamer
import sys
from io import BytesIO
import io
import os
from dotenv import load_dotenv
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, GLib
import ast
import shutil
#multi treading 
import asyncio
import nats
import os
import json
import numpy as np 
from PIL import Image
import cv2
import glob
from nanoid import generate
from multiprocessing import Process, Queue
import torch
import torchvision.transforms as T
from general import (check_requirements_pipeline)
import logging 
import threading
import gc
import fnmatch
import subprocess as sp
import time
from pytz import timezone 
from datetime import datetime
import ipfsApi

# Detection
from track import run
from track import lmdb_known
from track import lmdb_unknown

from nats.aio.client import Client as NATS

nc_client = NATS()

path1 = "./Nats_output"

if os.path.exists(path1) is False:
    os.mkdir(path1)
    
# Multi-threading
MODEL = 'cnn'
count_person =0
known_whitelist_faces = []
known_whitelist_id = []
known_blacklist_faces = []
known_blacklist_id = []
face_did_encoding_store = dict()
track_type = []
dict_frame = {}
frame = []
count_frame ={}
count = 0
processes = []
devicesUnique = []
activity_list = []
detect_count = []
person_count = []
vehicle_count = []
avg_Batchcount_person =[]
avg_Batchcount_vehicel = []
activity_list= []
geo_locations = []
track_person = []
track_vehicle = []
batch_person_id = []
timestamp = []
timestamp = ''
iterator = 1

#ipfs
# api = ipfsApi.Client('216.48.181.154', 5001)

start_flag = True
# gstreamer
# Initializes Gstreamer, it's variables, paths
Gst.init(sys.argv)
image_arr = None
device_types = ['', 'h.264', 'h.264', 'h.264', 'h.265', 'h.265', 'h.265']
load_dotenv()

first_dev_id = []


async def json_publish(primary):    
    nc = await nats.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    js = nc.jetstream()
    JSONEncoder = json.dumps(primary)
    json_encoded = JSONEncoder.encode()
    Subject = "service.attendance"
    Stream_name = "service"
    #await js.add_stream(name= Stream_name, subjects=[Subject])
    print(json_encoded)
    ack = await js.publish(Subject, json_encoded)
    print(f'Ack: stream={ack.stream}, sequence={ack.seq}')
    print("Activity is getting published")

async def Video_creating(file_id, device_data):
    global  first_dev_id
    print("check")
    device_id = device_data[0]
    device_urn = device_data[1]
    timestampp = device_data[2]
    if (len(first_dev_id) <= 1) and (device_id not in first_dev_id):
        first_dev_id.append(device_id)
    #first_dev_id = list(set(first_dev_id))
    
    print(first_dev_id)
    if device_id == first_dev_id[0]:
        
        #print(device_urn)
        queue1 = Queue()

        global avg_Batchcount_person, avg_Batchcount_vehicel,track_person,track_vehicle,detect_count
        
        file_id_str = str(file_id)
        video_name1 = path1 + '/' + str(device_id) +'/Nats_video'+str(device_id)+'-'+file_id_str+'.mp4'
        print(video_name1, [timestampp,device_id],queue1)
        # det = Process(target= run(video_name1, [timestampp,device_id],queue1))
        # det.start()
        
        run(video_name1, [timestampp,device_id],queue1)
            # rtsp_add_flag = False

        batch_output = queue1.get()

        track_type = batch_output["track_type"]
        track_person = batch_output["track_person"]
        batch_person_id = batch_output["batch_person_id"]
        detect_count = batch_output["detect_count"] 
        avg_Batchcount_person = batch_output["avg_Batchcount_person"]
        file_path = batch_output["save_dir"]
        batchId = batch_output["batchid"]

        #ipfs
        # #print("################################")
        # file_path = save_dir
        # #print(file_path)
        for path, dirs, files in os.walk(os.path.abspath(file_path)):
            #print(path, dirs, files)
            for filename in fnmatch.filter(files, "*.mp4"):
                src_file = os.path.join(file_path, filename)
                ##print(file_path)
                # res = api.add(src_file)
                # #print(res)
                # res_cid = res[0]['Hash']
                # #print(res_cid)
                command = 'ipfs --api=/ip4/216.48.181.154/tcp/5001 add {file_path} -Q'.format(file_path=src_file)
                #print(src_file)
                output = sp.getoutput(command)
                person_cid = output
                #print(person_cid)
            shutil.rmtree(path)
        torch.cuda.empty_cache()
                # video_cid.append(res_cid)
        # #print("################################")
        ct =  datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f')
        metapeople ={
                        "type":ast.literal_eval(track_type),
                        "track":ast.literal_eval(track_person),
                        "id":ast.literal_eval(batch_person_id),
                        "activity":None ,
                        "detect_time":str(timestamp)
                        }

        metaVehicle = {
                        "type":None,
                        "track":None,
                        "id":None,
                        "activity":None
        }
        metaElephant = {
            "track":None,
            "count":None
        }
        metaObj = {
                    "people":metapeople,
                    "vehicle":metaVehicle,
                    "elephant":metaElephant
                }
        
        metaBatch = {
            "detect": ast.literal_eval(detect_count),
            "count": {"people_count":avg_Batchcount_person,
                        "vehicle_count":None} ,
                    "object":metaObj,
                    "cid": person_cid
        }
        
        primary = { "pipeline":"attendance",
                    "deviceid":str(device_urn),
                    "batchid":batchId, 
                    
                    "timestamp":str(ct), 
                    "geo": {"latitude":'12.913632983105556',
                            "longitude":'77.58994246818435'},
                    "metaData": metaBatch}
        print("***************************************************************************************")
        print("***************************************************************************************")
        print("***************************************************************************************")

        print(primary)

        print("***************************************************************************************")
        print("***************************************************************************************")
        print("***************************************************************************************")

        await json_publish(primary=primary)


    #await asyncio.sleep(1)
async def gst_data(file_id , device_data):
    
    global count 
    count = count + 1
    print("**************************")
    print("**************************")
    print("**************************")
    print(count)
    print("**************************")
    print("**************************")
    print("**************************")
    sem = asyncio.Semaphore(1)
    device_id = device_data[0]
    device_urn = device_data[1]
    timestampp = device_data[2]
    await sem.acquire()

    if device_id not in devicesUnique:
        t = Process(target= await Video_creating(file_id=file_id, device_data=device_data))
        t.start()
        processes.append(t)
        #await Video_creating(file_id=file_id, device_data=device_data)
        devicesUnique.append(device_id)
    else:
        ind = devicesUnique.index(device_id)
        t = processes[ind]
        Process(name = t.name, target= await Video_creating(file_id=file_id , device_data=device_data))
        #await Video_creating(file_id=file_id , device_data=device_data)



    logging.basicConfig(filename="log_20.txt", level=logging.DEBUG)
    logging.debug("Debug logging test...")
    logging.info("Program is working as expected")
    logging.warning("Warning, the program may not function properly")
    logging.error("The program encountered an error")
    logging.critical("The program crashed")

async def gst_stream(gst_stream_data):

    device_id = gst_stream_data[0]
    urn = gst_stream_data[1]
    location = gst_stream_data[2]
    device_type = gst_stream_data[3]

    def format_location_callback(mux, file_id, data):

        device_data = []
        global start_flag
        device_data.append(data)
        device_data.append(urn)
        device_data.append(timestampp)
        if file_id == 0:
            if start_flag:
                pass
            else:
                file_id = 4
                asyncio.run(gst_data(file_id , device_data))
        else:
            file_id = file_id - 1
            start_flag = False
            asyncio.run(gst_data(file_id , device_data))

        #print(file_id, '-------', data)

    def frame_timestamp(identity, buffer, data):
        global timestampp
        timestampp =  datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f')

    try:
        # rtspsrc location='rtsp://happymonk:admin123@streams.ckdr.co.in:3554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif' protocols="tcp" ! rtph264depay ! h264parse ! splitmuxsink location=file-%03d.mp4 max-size-time=60000000000
        # pipeline = Gst.parse_launch('filesrc location={location} name={device_id} ! decodebin name=decode-{device_id} ! videoconvert name=convert-{device_id} ! videoscale name=scale-{device_id} ! video/x-raw, format=GRAY8, width = 1080, height = 1080 ! appsink name=sink-{device_id}'.format(location=location, device_id=device_id))
        video_name = path1 + '/' + str(device_id)
        #print(video_name)
        if not os.path.exists(video_name):
            os.makedirs(video_name, exist_ok=True)
        video_name = path1 + '/' + str(device_id) + '/Nats_video'+str(device_id)
        #print(video_name)
        if(device_type.lower() == "h264"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! identity name=ident-{device_id} ! rtph264depay name=depay-{device_id} ! h264parse name=parse-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-size-time=20000000000 max-files=5 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))
        elif(device_type.lower() == "h265"):
            pipeline = Gst.parse_launch('rtspsrc location={location} protocols="tcp" name={device_id} ! identity name=ident-{device_id} ! rtph265depay name=depay-{device_id} ! h265parse name=parse-{device_id} ! splitmuxsink location={path}-%01d.mp4 max-size-time=20000000000 max-files=5 name=sink-{device_id}'.format(location=location, path=video_name, device_id = device_id))

        sink = pipeline.get_by_name('sink-{device_id}'.format(device_id=device_id))
        identity = pipeline.get_by_name('ident-{device_id}'.format(device_id=device_id))
    
        if not pipeline:
            print("Not all elements could be created.")

        
        identity.connect("handoff", frame_timestamp, device_id)
        sink.connect("format-location", format_location_callback, device_id)
        
        # Start playing
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Unable to set the pipeline to the playing state.")

    except TypeError as e:
        print(TypeError," gstreamer streaming error >> ", e)

#
def on_message(bus: Gst.Bus, message: Gst.Message, loop: GLib.MainLoop):
    mtype = message.type
    """
        Gstreamer Message Types and how to parse
        https://lazka.github.io/pgi-docs/Gst-1.0/flags.html#Gst.MessageType
    """

    if mtype == Gst.MessageType.EOS:
        #print("End of stream")
        loop.quit()

    elif mtype == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(("Error received from element %s: %s" % (
            message.src.get_name(), err)))
        print(("Debugging information: %s" % debug))
        loop.quit()

    elif mtype == Gst.MessageType.STATE_CHANGED:
        if isinstance(message.src, Gst.Pipeline):
            old_state, new_state, pending_state = message.parse_state_changed()
            print(("Pipeline state changed from %s to %s." %
            (old_state.value_nick, new_state.value_nick)))

    elif mtype == Gst.MessageType.ELEMENT:
        print(message.src)
    return True

async def cb(msg):
    try :
        print("entered callback")
        data = (msg.data)
        data  = data.decode()
        data = json.loads(data)
        if "urn" not in data:
            data["urn"] = "uuid:eaadf637-a191-4ae7-8156-07433934718b"
        gst_stream_data = [data["deviceId"], data["urn"], data["rtsp"], data["videoEncodingInformation"]]
        if (data):
            p2 = Process(target = await gst_stream(gst_stream_data))
            p2.start()
            
            # p3 = Process(target = await hls_stream(data))
            # p3.start()

        subject = msg.subject
        reply = msg.reply
        await nc_client.publish(msg.reply,b'Received!')
        print("Received a message on '{subject} {reply}': {data}".format(
            subject=subject, reply=reply, data=data))
        
    except TypeError as e:
        print(TypeError," Nats msg callback error >> ", e)

async def main():
    lmdb_known()
    lmdb_unknown()

    
    pipeline = Gst.parse_launch('fakesrc ! queue ! fakesink')

    # # Init GObject loop to handle Gstreamer Bus Events
    # loop = GLib.MainLoop()

    # bus = pipeline.get_bus()
    # # allow bus to emit messages to main thread
    # bus.add_signal_watch()

    # # Add handler to specific signal
    # bus.connect("message", on_message, loop)
    await nc_client.connect(servers=["nats://216.48.181.154:5222"])
    print("Nats Connected")
    
    await nc_client.subscribe("service.device_discovery", cb=cb)
    print("subscribed")

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # for i in range(3,4):
    
    #     device_urn = 'uuid:eaadf637-a191-4ae7-8156-07433934718b'
    #     stream_url = os.getenv('RTSP_URL_{id}'.format(id=i))
    #     target= await gst_stream(device_id=i, urn=device_urn, location=stream_url, device_type=device_types[i])
    #     time.sleep(5)

    
    # try:
    #     loop.run()
    # except Exception:
    #     traceback.print_exc()
    #     loop.quit()

    # # Stop Pipeline
    pipeline.set_state(Gst.State.NULL)
    del pipeline

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try :
        loop.run_until_complete(main())
        loop.run_forever()
    except RuntimeError as e:
        print("error ", e)
        print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")
        
"""
Json Object For a Batch Video 

JsonObjectBatch= {ID , TimeStamp , {Data} } 
Data = {
    "person" : [ Device Id , [Re-Id] , [Frame TimeStamp] , [Lat , Lon], [Person_count] ,[Activity] ]
}  

"""

"""
metapeople ={
                    "type":{" 00: known whitelist, 01: known blacklist, 10: unknown first time, 11: unknown repeat"},
                    "track":{" 0: tracking OFF, 1: tracking ON"},
                    "id":"face_id",
                    "activity":{"Null"}  
                    }
    
    metaVehicel = {
                    "type":{"Null"},
                    "track":{"Null"},
                    "id":"Null",
                    "activity":"Null"
    }
    metaObj = {
                 "people":metapeople,
                 "vehicle":"Null"
               }
    
    metaBatch = {
        "Detect": "0: detection NO, 1: detection YES",
        "Count": {"people_count":str(avg_Batchcount),
                  "vehicle_count":"Null" ,
        "Object":metaObj
        
    }
    
    primary = { "deviceid":str(Device_id),
                "batchid":str(BatchId), 
                "timestamp":str(frame_timestamp), 
                "geo":str(Geo_location),
                "metaData": metaBatch}
    #print(primary)
    
"""
