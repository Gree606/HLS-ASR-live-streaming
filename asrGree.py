'''
ASR API V2
Developed by Arun Kumar A(CS20S013) - October 2022
Nginx HLS Streaming , VAD, socket emitting the output text: Author- Greeshma Susan John
'''
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, request
from flask import render_template
import torch
import librosa
import os
import traceback
import base64
import wave
from collections import defaultdict
from pathlib import Path
import yaml
import time
import numpy as np
import datetime
import logging
from silero_vad import get_silero_vad_model, get_speech_timestamps
from datetime import timedelta
import kaldiio
from webvtt import WebVTT, Caption
import subprocess
from speechbrain.pretrained import EncoderClassifier
from transcribe_speech import ASR
from itertools import repeat, chain
from werkzeug.datastructures import FileStorage
import wget
from whisper_transcribe_speech import Whisper_ASR
from transformers import WhisperProcessor, WhisperForConditionalGeneration


###### imports for streaming
import auditok
import os
from pydub import AudioSegment
import requests
from flask_socketio import SocketIO
import re
import json
'''
MODELS_DIR is the root directory of all models. This will contain a directory for each language.
Inside each language directory, there can be multiple directories, one for each espnet model.
Each model directory has to contain all the necessary files like checkpoint, config, tokenizer model required to load an espnet model.
'''
MODELS_DIR="/home/tenet/gree/ASR_IITM/asr_models"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"

DEVICE = "cpu"
#DEVICE="cuda"
if DEVICE == "cuda":
    MAX_WORKERS=2
    N_GPUS=2
else:
    MAX_WORKERS=8
BEAM_SIZE=20
CTC_WEIGHT=0.3   
FILE_UPLOAD_DIR="/home/tenet/gree/ASR_IITM/ASR_OUT/uploads/" 
ENABLE_TIMESTAMP=True
SAMPLING_RATE=16000
ENABLE_VAD=False
MAX_SPEECH_DURATION=5
FBANK_MODELS = set(["tamil/ulca_tamil_transformer"])
#FBANK_MODELS = set()
DEFAULT_MODEL = {
     "english" : "ulca_780h_conformer",
     "tamil" : "ulca_tamil_transformer",
    #  "tamil" : "tamil_3000h_raw_transformer",
     "hindi" : "hindi_ulca_2400h_conformer"
}

SPEECH_TO_TEXT_DEFAULT_CONFIG = {
    "maxlenratio": 0.0,
    "minlenratio": 0.0,
    "beam_size": BEAM_SIZE,
    "ctc_weight": CTC_WEIGHT,
    "lm_weight": 0.0,
    "penalty": 0.0,
    "nbest": 1,
    "device": DEVICE,
}

SUPPORTED_WHISPER_LANGUAGES = {}

logger = logging.getLogger("ASR_API")
logger.setLevel(logging.INFO)

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
socketio = SocketIO(app,aync_mode='gevent', monkey_patch=True)

# initialise inference model
import sys
import espnet
from espnet2.bin.asr_inference import Speech2Text
import argparse
import numpy as np
import wave
from scipy.io.wavfile import read, write
import scipy.signal as sps

streamOn=False

def get_model(config):  
    print("============================")  
    print(config)
    print("#############################")
    speech2text = Speech2Text(
        **config
    )
    
    return speech2text

model_map = defaultdict(lambda: {})
model_description = defaultdict(lambda: {})
whisper_processor = defaultdict(lambda: {})
whisper_device = defaultdict(lambda: {})
worker_pool = None
whisper_worker_pool = None

vad_model = None
lid_model = None

FRONT_END_LANG_MAPS = {'sanskrit': 'संस्कृत', 'english': 'English', 'hindi': 'हिन्दी', 
                        'telugu': 'తెలుగు', 'bengali': 'বাংলা', 'tamil': 'தமிழ்', 
                        'kannada': 'ಕನ್ನಡ', 'marathi': 'मराठी', 'gujarati': 'ગુજરાતી',
                        'malayalam': 'മലയാളം', 'urdu': 'اردو', 'odia': 'ଓଡିଆ', 
                        'assamese': 'অসমীয়া', 'maithili': 'मैथिली', 'kashmiri': 'कश्मिरि', 
                        'bhojpuri': 'भोजपुरी', 'awadhi': 'अवधी', 'chattisgarhi': 'छत्तीसगढ़ी',
                        'punjabi': 'ਪੰਜਾਬੀ'
                    }

lang_map = {"hi":"hindi","en":"english","ta":"tamil","te":"telugu","ml":"malayalam","mr":"marathi","raj":"rajasthani","bn":"bengali","kn":"kannada","or":"odia","gu":"gujarati","pa":"punjabi"}

def decode_speech(input_file, request_data):
    st = time.time()
    tmp_dir = None
    with app.app_context():
        timestamp=""
        if ENABLE_TIMESTAMP:
            timestamp=str(datetime.datetime.now())
            for spl_char in ['-', ' ', ':', '.']:
                timestamp = timestamp.replace(spl_char, '_')
        filename, ext = None, None
        if "." in input_file.filename:
            filename, ext = input_file.filename.split(".")
        else:
            filename = input_file.filename
        tmp_dir=f"tmp/data/data_{timestamp}"
        print("tmp_dir" , tmp_dir)
        os.makedirs(tmp_dir, exist_ok = True)
        file_save_path = os.path.join(FILE_UPLOAD_DIR, filename + "_" + timestamp + "." + ext)
        input_file.save(file_save_path)
        print ("File saved at",file_save_path)
        logger.info(f"File saved at {file_save_path}")
        data, samplerate = librosa.load(file_save_path, sr=SAMPLING_RATE)
        print("file_save_path_afetr_blibrosa",file_save_path)
        print("data=",data)
        language = request_data["language"] if "language" in request_data else "english"
        model_variant = request_data["model"] if "model" in request_data else "default"
        # language = request.values.get("language") if "language" in request.values else "english"
        # model_variant = request.values.get("model") if "model" in request.values else "default"

        if model_variant == "default":
            logger.info(f"Selecting default model {DEFAULT_MODEL[language]}")
            model_variant = DEFAULT_MODEL[language]

        if language not in model_map:
            return jsonify(status='failure', reason=f"The selected language {language} is not supported"), (tmp_dir, file_save_path)

        if model_variant not in model_map[language]:
            return jsonify(status='failure', reason=f"The selected model variant {model_variant} is not available in language {language}"), (tmp_dir, file_save_path)
        
        selected_model = model_map[language][model_variant]
        
        # Apply VAD and get speech segments
        speech_segments = []
        speech_segments_timestamps = []
        if ENABLE_VAD:
            vad_start = time.time()
            speech_timestamps = get_speech_timestamps(data, vad_model, sampling_rate=SAMPLING_RATE,return_seconds=True, speech_pad_ms=400, min_speech_duration_ms=100)
            running_segment = []
            running_segment_timestamp = []
            running_segment_dur = 0.0
            for speech_timestamp in speech_timestamps:
                current_segment = data[speech_timestamp['start']:speech_timestamp['end']]
                current_segment_dur = speech_timestamp['end_in_sec'] - speech_timestamp['start_in_sec']
                if running_segment_dur + current_segment_dur < MAX_SPEECH_DURATION:
                    running_segment.append(current_segment)
                    running_segment_timestamp.append(speech_timestamp)
                    running_segment_dur += current_segment_dur
                elif running_segment_dur + current_segment_dur == MAX_SPEECH_DURATION:
                    running_segment.append(current_segment)
                    running_segment_timestamp.append(speech_timestamp)
                    speech_segments.append(np.concatenate(running_segment))
                    speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
                    running_segment = []
                    running_segment_timestamp = []
                    running_segment_dur = 0.0
                else:
                    if len(running_segment) == 0:
                        speech_segments.append(current_segment)
                        speech_segments_timestamps.append((speech_timestamp['start_in_sec'], speech_timestamp['end_in_sec']))
                        running_segment = []
                        running_segment_timestamp = []
                        running_segment_dur = 0.0
                    else:
                        speech_segments.append(np.concatenate(running_segment))
                        speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
                        running_segment = [current_segment]
                        running_segment_timestamp = [speech_timestamp]
                        running_segment_dur = current_segment_dur
            if len(running_segment) > 0:
                # append the residual speech segment
                speech_segments.append(np.concatenate(running_segment))
                speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
            vad_end = time.time()
            vad_elapsed = "{:.2f}".format((vad_end-vad_start))
            logger.info(f"Time taken for VAD {vad_elapsed}")
        else:
            speech_segments.append(data)

        n_segments = len(speech_segments)
        logger.info(f"number of speech segments: {n_segments}")

        if n_segments == 0:
            return jsonify(status='failure', reason="No speech detected"), (tmp_dir, file_save_path)
        
        if f"{language}/{model_variant}" in FBANK_MODELS:
            # create tmp directory
            os.makedirs(tmp_dir, exist_ok = True)
            # if model is fbank_pitch
            # create wav.scp and segments files
            # resave audio file as wav with one channel and 16k sample rate
            logger.info("Extracting fbank pitch features using kaldi...")
            ffmpeg_cmd = f"ffmpeg -y -i {file_save_path} -ar 16000 -ac 1 {tmp_dir}/input.wav"
            os.system(ffmpeg_cmd)
            input_file_path = os.path.abspath(f"{tmp_dir}/input.wav")
            with open(f"{tmp_dir}/wav.scp", "w") as wav_scp:
                wav_scp.write(f"input {input_file_path}\n")
                wav_scp.close()
            with open(f"{tmp_dir}/segments", "w") as segments, open(f"{tmp_dir}/spk2utt", "w") as spk2utt, open(f"{tmp_dir}/utt2spk", "w") as utt2spk:
                for start, end in speech_segments_timestamps:
                    start_id = str((f"{start:.1f}").zfill(8)).replace(".", "_")
                    end_id = str((f"{end:.1f}").zfill(8)).replace(".", "_")
                    segments.write(f"input_{start_id}_{end_id} input {start} {end}\n")
                    spk2utt.write(f"input_{start_id}_{end_id} input_{start_id}_{end_id}\n")
                    utt2spk.write(f"input_{start_id}_{end_id} input_{start_id}_{end_id}\n")
            # use espnet utils for generating fbank pitch features
            _nj = min(n_segments, 32)
            data_dir = os.path.abspath(tmp_dir)
            fbank_cmd = f"bash fbank_pitch_wrapper.sh {data_dir} {_nj}"
            out = os.system(fbank_cmd)
            logger.info(out)

            # load speech segments in memory from feats.scp
            feats_scp = kaldiio.load_scp_sequential(f'{tmp_dir}/feats.scp')
            print("**********")
            print(tmp_dir)
            print("!!!!!!!!!!!!!!!!")
            speech_segments = []
            for key, np_array in feats_scp:
                speech_segments.append(np_array)
        else:
            # if model is raw
            logger.info("Using raw features")
            pass

        # move data to DEVICE
        # speech_segments = [torch.tensor(t, device=selected_model.device) for t in speech_segments]
        
        # chunksize = max(1, int(n_segments/MAX_WORKERS))
        # outputs = None
        splits = min(len(speech_segments), MAX_WORKERS)
        if model_variant == "Fine-tuned_Whisper":
            outputs = whisper_worker_pool.starmap(Whisper_ASR.transcribe, zip(repeat(selected_model), repeat(whisper_processor[language]), repeat(whisper_device[language]), repeat(language), np.array_split(np.array(speech_segments), splits)))
        else:
            outputs = worker_pool.starmap(ASR.transcribe, zip(repeat(selected_model), np.array_split(np.array(speech_segments), splits)))

        transcriptions = list(chain(*outputs))
        # results = selected_model(speech=data)
        # outputs = [selected_model(data)]
        # text = []
        # for results in outputs:
        #     if results is not None and len(results) > 0:
        #         nbests = [text for text, token, token_int, hyp in results]
        #         text.append(nbests[0] if nbests is not None and len(nbests) > 0 else "")
        #     # else:
        #     #     return jsonify(status='failure', reason=f"Could not transcribe the given speech")
        
        transcription = " ".join(transcriptions)

        # create VTT file
        vtt = None
        if ENABLE_VAD and "vtt" in request_data and request_data["vtt"] == "true":
            logger.info("VTT requested. Generating VTT...")
            vtt = WebVTT()
            for timestamp, cap in zip(speech_segments_timestamps, transcriptions):
                start_delta = timedelta(seconds=timestamp[0])
                end_delta = timedelta(seconds=timestamp[1])
                start_milli = int(start_delta.microseconds/1000)
                end_milli = int(end_delta.microseconds/1000)
                start_time = get_formatted_time(start_delta.seconds) + "." + f"{start_milli:03d}"
                end_time = get_formatted_time(end_delta.seconds) + "." + f"{end_milli:03d}"
                caption = Caption(
                    start_time,
                    end_time,
                    cap
                )
                vtt.captions.append(caption)
                    
        et = time.time()
        elapsed_time = "{:.2f}".format((et-st))
        # write output and captions to a file
        if vtt is not None:
            vtt.save('tmp/data/captions.vtt')
        with open('tmp/data/output.txt', 'w') as f:
            f.write(transcription)
            f.write("\n")
        if vtt is None:
            return jsonify(status="success", transcript=transcription, time_taken=elapsed_time), (tmp_dir, file_save_path), transcription
        else:
            return jsonify(status="success", transcript=transcription, time_taken=elapsed_time, vtt=vtt.content), (tmp_dir, file_save_path), transcription

def ulca_decode_speech(url, lang, urlFlag):
    st = time.time()
    tmp_dir = None
    with app.app_context():
        timestamp=""
        if ENABLE_TIMESTAMP:
            timestamp=str(datetime.datetime.now())
            for spl_char in ['-', ' ', ':', '.']:
                timestamp = timestamp.replace(spl_char, '_')
        tmp_dir=f"tmp/data/data_{timestamp}"
        os.makedirs(tmp_dir, exist_ok = True)
        if urlFlag:
            file_save_path = wget.download(url, tmp_dir)
        else:
            file_save_path = tmp_dir+"/input_" + timestamp + ".wav"
            # print(file_save_path)
            wav_file_id = open(file_save_path,'wb')
            decode_string = base64.b64decode(url)
            wav_file_id.write(decode_string)
            wav_file_id.close()
        if "." in file_save_path:
            filename, ext = file_save_path.split(".")
        else:
            filename = file_save_path
        logger.info(f"File saved at {file_save_path}")
        data, samplerate = librosa.load(file_save_path, sr=SAMPLING_RATE)
        language = lang
        model_variant = "default"
        # language = request.values.get("language") if "language" in request.values else "english"
        # model_variant = request.values.get("model") if "model" in request.values else "default"

        if model_variant == "default":
            logger.info(f"Selecting default model {DEFAULT_MODEL[language]}")
            model_variant = DEFAULT_MODEL[language]

        if language not in model_map:
            return jsonify(status='failure', reason=f"The selected language {language} is not supported"), (tmp_dir, file_save_path)

        if model_variant not in model_map[language]:
            return jsonify(status='failure', reason=f"The selected model variant {model_variant} is not available in language {language}"), (tmp_dir, file_save_path)
        
        selected_model = model_map[language][model_variant]
        
        # Apply VAD and get speech segments
        speech_segments = []
        speech_segments_timestamps = []
        if ENABLE_VAD:
            vad_start = time.time()
            speech_timestamps = get_speech_timestamps(data, vad_model, sampling_rate=SAMPLING_RATE,return_seconds=True, speech_pad_ms=400, min_speech_duration_ms=100)
            running_segment = []
            running_segment_timestamp = []
            running_segment_dur = 0.0
            for speech_timestamp in speech_timestamps:
                current_segment = data[speech_timestamp['start']:speech_timestamp['end']]
                current_segment_dur = speech_timestamp['end_in_sec'] - speech_timestamp['start_in_sec']
                if running_segment_dur + current_segment_dur < MAX_SPEECH_DURATION:
                    running_segment.append(current_segment)
                    running_segment_timestamp.append(speech_timestamp)
                    running_segment_dur += current_segment_dur
                elif running_segment_dur + current_segment_dur == MAX_SPEECH_DURATION:
                    running_segment.append(current_segment)
                    running_segment_timestamp.append(speech_timestamp)
                    speech_segments.append(np.concatenate(running_segment))
                    speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
                    running_segment = []
                    running_segment_timestamp = []
                    running_segment_dur = 0.0
                else:
                    if len(running_segment) == 0:
                        speech_segments.append(current_segment)
                        speech_segments_timestamps.append((speech_timestamp['start_in_sec'], speech_timestamp['end_in_sec']))
                        running_segment = []
                        running_segment_timestamp = []
                        running_segment_dur = 0.0
                    else:
                        speech_segments.append(np.concatenate(running_segment))
                        speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
                        running_segment = [current_segment]
                        running_segment_timestamp = [speech_timestamp]
                        running_segment_dur = current_segment_dur
            if len(running_segment) > 0:
                # append the residual speech segment
                speech_segments.append(np.concatenate(running_segment))
                speech_segments_timestamps.append((running_segment_timestamp[0]['start_in_sec'], running_segment_timestamp[-1]['end_in_sec']))
            vad_end = time.time()
            vad_elapsed = "{:.2f}".format((vad_end-vad_start))
            logger.info(f"Time taken for VAD {vad_elapsed}")
        else:
            speech_segments.append(data)

        n_segments = len(speech_segments)
        logger.info(f"number of speech segments: {n_segments}")

        if n_segments == 0:
            return jsonify(status='failure', reason="No speech detected"), (tmp_dir, file_save_path)
        
        if f"{language}/{model_variant}" in FBANK_MODELS:
            # create tmp directory
            os.makedirs(tmp_dir, exist_ok = True)
            # if model is fbank_pitch
            # create wav.scp and segments files
            # resave audio file as wav with one channel and 16k sample rate
            logger.info("Extracting fbank pitch features using kaldi...")
            ffmpeg_cmd = f"ffmpeg -y -i {file_save_path} -ar 16000 -ac 1 {tmp_dir}/input.wav"
            os.system(ffmpeg_cmd)
            input_file_path = os.path.abspath(f"{tmp_dir}/input.wav")
            with open(f"{tmp_dir}/wav.scp", "w") as wav_scp:
                wav_scp.write(f"input {input_file_path}\n")
                wav_scp.close()
            with open(f"{tmp_dir}/segments", "w") as segments, open(f"{tmp_dir}/spk2utt", "w") as spk2utt, open(f"{tmp_dir}/utt2spk", "w") as utt2spk:
                for start, end in speech_segments_timestamps:
                    start_id = str((f"{start:.1f}").zfill(8)).replace(".", "_")
                    end_id = str((f"{end:.1f}").zfill(8)).replace(".", "_")
                    segments.write(f"input_{start_id}_{end_id} input {start} {end}\n")
                    spk2utt.write(f"input_{start_id}_{end_id} input_{start_id}_{end_id}\n")
                    utt2spk.write(f"input_{start_id}_{end_id} input_{start_id}_{end_id}\n")
            # use espnet utils for generating fbank pitch features
            _nj = min(n_segments, 32)
            data_dir = os.path.abspath(tmp_dir)
            fbank_cmd = f"bash fbank_pitch_wrapper.sh {data_dir} {_nj}"
            out = os.system(fbank_cmd)
            logger.info(out)

            # load speech segments in memory from feats.scp
            feats_scp = kaldiio.load_scp_sequential(f'{tmp_dir}/feats.scp')
            speech_segments = []
            for key, np_array in feats_scp:
                speech_segments.append(np_array)
        else:
            # if model is raw
            logger.info("Using raw features")
            pass

        # move data to DEVICE
        # speech_segments = [torch.tensor(t, device=selected_model.device) for t in speech_segments]
        
        # chunksize = max(1, int(n_segments/MAX_WORKERS))
        # outputs = None
        splits = min(len(speech_segments), MAX_WORKERS)
        outputs = worker_pool.starmap(ASR.transcribe, zip(repeat(selected_model), np.array_split(np.array(speech_segments), splits)))

        transcriptions = list(chain(*outputs))
        # results = selected_model(speech=data)
        # outputs = [selected_model(data)]
        # text = []
        # for results in outputs:
        #     if results is not None and len(results) > 0:
        #         nbests = [text for text, token, token_int, hyp in results]
        #         text.append(nbests[0] if nbests is not None and len(nbests) > 0 else "")
        #     # else:
        #     #     return jsonify(status='failure', reason=f"Could not transcribe the given speech")
        
        et = time.time()
        elapsed_time = "{:.2f}".format((et-st))

        transcription = " ".join(transcriptions)

        # create VTT file
        return jsonify(status="success", transcript=transcription, time_taken=elapsed_time), (tmp_dir, file_save_path)
        
def make_dummy_calls():
    # dummy calls as a hack to initalize cuda processes so that requests are served fast
    logger.info("Making dummy calls...")
    try:
        # have a dummy wav file for all languages and use it
        with open("sample_speech", "r") as f:
            sample_speech = {}
            for line in f.readlines():
                line = line.strip()
                lang, wav_file_path = line.split(" ", 1)
                sample_speech[lang] = wav_file_path
            for lang in sample_speech:
                if lang in model_description:
                    for model in model_description[lang]["models"]:
                        logger.info(f"Making dummy call for {lang}/{model}")
                        json_data = {
                            "language": lang,
                            "model": model
                        }
                        input_file = None
                        wav_file_path = sample_speech[lang]
                        with open(wav_file_path, 'rb') as f:
                            input_file = FileStorage(f)
                            filename = os.path.basename(wav_file_path)
                            input_file.filename = filename
                            out, tmp_files = decode_speech(input_file, json_data)
                            logger.info(f"Output: {out}")
                            # clear temp files in a background process
                            for tmp_file in tmp_files:
                                if tmp_file is not None:
                                    subprocess.Popen(["rm","-rf",tmp_file])
            logger.info("All dummy calls has been made")
    except:
        traceback.print_exc()

def setup_app():
    global model_map, model_description, worker_pool, whisper_worker_pool, vad_model, lid_model
    model_i = 0
    for language in os.listdir(MODELS_DIR):
        language_dir = os.path.join(MODELS_DIR, language)
        if os.path.isdir(language_dir):
            for model_name in os.listdir(language_dir):
                model_dir_path = os.path.join(language_dir, model_name)
                if model_name.startswith(".") or not os.path.isdir(model_dir_path):
                    continue
                config_file = Path(os.path.join(model_dir_path, "config.yaml"))
                if config_file.exists():
                    with config_file.open("r", encoding="utf-8") as f:
                        args = yaml.safe_load(f)
                    args = argparse.Namespace(**args)
                    config = SPEECH_TO_TEXT_DEFAULT_CONFIG.copy()
                    config['bpemodel'] = os.path.join(model_dir_path, args.bpemodel)
                    config['asr_train_config'] = config_file
                    config['asr_model_file'] = os.path.join(model_dir_path, "model.pth")

                    device = DEVICE
                    if device == "cuda":
                        device = device + ":" + str((model_i)%N_GPUS)

                    config['device'] = device

                    decode_config_file = Path(os.path.join(model_dir_path, "decode_config.yaml"))
                    if decode_config_file.exists():
                        with decode_config_file.open("r", encoding="utf-8") as f:
                            decode_args = yaml.safe_load(f)
                        decode_args = argparse.Namespace(**decode_args)
                        # overriding with model specific decode arguments
                        config['beam_size'] = decode_args.beam_size
                        config['ctc_weight'] = decode_args.ctc_weight
                    model = get_model(config)
                    model_map[language][model_name] = model
                    logger.info(f"{language}/{model_name} loaded in device {config['device']}")
                    model_i += 1

                    # load description of the model
                    if not bool(model_description[language]):
                        caption = None
                        try:                            
                            with open(os.path.join(language_dir, "caption"), "r") as f:
                                caption = f.read().strip()
                        except Exception as e:
                            logger.error(e)
                        model_description[language]["caption"] = caption if caption is not None else language
                    if "models" not in model_description[language]:
                        model_description[language]["models"] = {}
                    description = ""
                    try:                            
                        with open(os.path.join(model_dir_path, "description"), "r") as f:
                            description = f.read().strip()
                    except:
                        pass
                    model_description[language]["models"][model_name] = {"description": description}
                else:
                    logger.warning(f"Warning! config.yaml is missing in {model_dir_path}. Ignoring this model.")

    for language in SUPPORTED_WHISPER_LANGUAGES.keys():
        model_name = "Fine-tuned_Whisper"
        if "models" not in model_description[language]:
            model_description[language]["models"] = {}
        # model_map[language][model_name] = SUPPORTED_WHISPER_LANGUAGES[language]
        model_description[language]["models"][model_name] = {"description": "whisper-small-model"}
        model_description[language]["caption"] = FRONT_END_LANG_MAPS[language]
        device = DEVICE
        if device == "cuda":
            device = device + ":" + str((model_i)%N_GPUS)
            model_i += 1
        model_map[language][model_name] = WhisperForConditionalGeneration.from_pretrained(SUPPORTED_WHISPER_LANGUAGES[language]).to(device)
        whisper_processor[language] = WhisperProcessor.from_pretrained(SUPPORTED_WHISPER_LANGUAGES[language])
        whisper_device[language] = device
        logger.info(f"{language}/{model_name} loaded in device {device}")
    
    for language in model_map.keys():
        # choose default model for languages for which DEFAULT_MODEL is not set
        if language not in DEFAULT_MODEL:
            DEFAULT_MODEL[language] = list(model_map[language].keys())[0]

    for lang, model_name in DEFAULT_MODEL.items():
        if lang in model_description:
            model_description[lang]["models"][model_name]["default"] = True

    logger.info("Default models:")
    logger.info(DEFAULT_MODEL)
    logger.info(f"The models of following languages have been loaded: {list(model_map.keys())}" )

    vad_model = get_silero_vad_model()
    logger.info("VAD model loaded")

    if DEVICE == "cuda":
        mp = torch.multiprocessing.get_context("spawn")
    else:
        import multiprocessing as mp
    # import multiprocessing as mp
    worker_pool = mp.Pool(MAX_WORKERS)
    whisper_worker_pool = mp.Pool(MAX_WORKERS)
   
    make_dummy_calls()
    logger.info("Server initialized")
    

setup_app()

def get_formatted_time(time_in_seconds):
    m, s = divmod(time_in_seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def getLangId(file_save_path, tmp_dir):
    ffmpeg_cmd = f"ffmpeg -y -i {file_save_path} -ar 16000 -ac 1 {tmp_dir}/input.wav"
    os.system(ffmpeg_cmd)
    input_file_path = os.path.abspath(f"{tmp_dir}/input.wav")
    x = lid_model.load_audio(input_file_path, overwrite=True)
    out = lid_model.classify_batch(x)
    logger.info(out)
    return out[3][0]

@app.route('/', strict_slashes=False)
def home():
    return render_template('index1.html')

@app.route('/langs', methods=['GET', 'POST'], strict_slashes=False)
def list_supported_languages():
    return jsonify(model_description)

@app.route('/decode_single_segment', methods=['GET', 'POST'], strict_slashes=False)
def decode_single_segment():
    language = request.values.get("language") if "language" in request.values else "english"
    model_variant = request.values.get("model") if "model" in request.values else "default"
    if model_variant == "default":
        logger.info(f"Selecting default model {DEFAULT_MODEL[language]}")
        model_variant = DEFAULT_MODEL[language]

    if language not in model_map:
        return jsonify(status='failure', reason=f"The selected language {language} is not supported")

    if model_variant not in model_map[language]:
        return jsonify(status='failure', reason=f"The selected model variant {model_variant} is not available in language {language}")
    
    selected_model = model_map[language][model_variant]
    files = request.files
    input_file = files.get('file')
    file_save_path = os.path.join(FILE_UPLOAD_DIR, input_file.filename)
    input_file.save(file_save_path)
    # logger.info(f"File saved at {file_save_path}")
    data, samplerate = librosa.load(file_save_path, sr=SAMPLING_RATE)
    
    outputs = worker_pool.starmap(ASR.transcribe, [(selected_model, [data])])

    transcriptions = list(chain(*outputs))

    transcription = " ".join(transcriptions)
    return jsonify(transcript=transcription)    

##########################################Streaming & VAD - Gree################################################

@app.route('/display')
def index():
    return render_template("index1.html")

@app.route('/stopStream', methods=['GET', 'POST'], strict_slashes=False)
def stream_stop():
    streamOn=False
    return jsonify(status="success", reason="Stream turned off")


@app.route('/stream', methods=['GET', 'POST'], strict_slashes=False)
def asr_stream():
    request_data = request.get_json(force=True, silent=True)
    # print("Logs")
    # print(request_data)
    if request_data is None:
        request_data = request.values
        # print(request_data)
    logger.debug("Request received")
    files=os.listdir('/nginx/hls')
    files.sort()
    num=0
    streamOn=True

    while(streamOn):
        # count=0
        # for file1 in files:
        #     if os.path.isfile(os.path.join('/nginx/hls', file1)):
        #         count += 1
        num_value=-1
        files=os.listdir('/nginx/hls')
        files.sort()
        if len(files) >= 2:
            # Extract the second last file name
            second_last_file_name = files[-2]
            # print("Second last file name:", second_last_file_name)
            num_value=int(re.search(r'\d+',second_last_file_name).group())
            # print("Num Value", num_value)

        # print(files)
        j=0
        # print("num and num_value",num, num_value)
        while (num<=num_value):
            print("num and num_value",num, num_value)
            tmp_files = []
            ts_file='/nginx/hls/obs-'+str(num)+'.ts'
            sampling_rate=16000
            audio = auditok.AudioRegion.load(ts_file, sampling_rate=sampling_rate)
            # raw_audio = audio.raw_data
            regions = auditok.split(audio, min_dur=0.25, max_dur=5, max_silence=0.3, energy_threshold=55)
            
            num_regions = sum(1 for _ in regions)
            regions = auditok.split(audio, min_dur=0.5, max_dur=100.0, max_silence=0.3, energy_threshold=40, sampling_rate=16000)
            # print("regions",regions)
            for i, region in enumerate(regions):
                print("j", j)
                file1 = region.save("sepAudio/samp{j}.wav".format(i=i, j=j))
                # input_file = None
                try:
                    file_path="sepAudio/samp{j}.wav".format(j=j)
                    ################################direct decode segment###########################
                    if os.path.isfile(file_path):
                        input_file = FileStorage(stream=open(file_path, "rb"), filename=file_path)
                    out, tmp_files, trans_val = decode_speech(input_file, request_data)
                    ###############################gpu decode segment###############################
                    # if os.path.isfile(file_path):
                    #     input_file = ('file', open(file_path, "rb"))
                    #     url = "http://projects.respark.iitm.ac.in:1241/decode_single_segment"
                    #     payload = {'vtt': 'true','language': 'english'}
                    #     headers = {}
                    #     response = requests.request("POST", url, headers=headers, data=payload, files=[input_file])
                    #     response_data=json.loads(response.text)
                    #     print(response_data.get('transcript'))
                        # socketio.emit("asrOut",trans_val)
                    #################################################################################
                    # return out
                    # socketio.emit("asrOut",trans_val)
                except Exception as err:
                    logger.error(err)
                    return jsonify(status="failure", reason=str(err))
                finally:
                    # clear temp files in a background process
                    for tmp_file in tmp_files:
                        if tmp_file is not None:
                            subprocess.Popen(["rm","-rf",tmp_file])
                j=j+1
            num=num+1
        # return jsonify(status="failure", reason="No valid response found")

###############################################################################
        

@app.route('/decode', methods=['GET', 'POST'], strict_slashes=False)
def asr_inference():
    tmp_files = []
    request_data = request.get_json(force=True, silent=True)
    # print("Logs")
    # print(request_data)
    if request_data is None:
        request_data = request.values
        # print(request_data)
    logger.debug("Request received")
    files = request.files
    # print(files)
    input_file = None
    try:
        input_file = files.get('file')
        # print(input_file)
        # print("End")
    except Exception as err:
        logger.error(err)
        return jsonify(status='failure', reason=f"Unsupported input. {err}")
    try:
        # out, tmp_files = decode_speech(input_file, request_data)
        return out
    except Exception as err:
        logger.error(err)
        return jsonify(status="failure", reason=str(err))
    finally:
        # clear temp files in a background process
        for tmp_file in tmp_files:
            if tmp_file is not None:
                subprocess.Popen(["rm","-rf",tmp_file])

@app.route('/ulca_api', methods=['GET', 'POST'], strict_slashes=False)
def ulca_api():
    tmp_files = []
    request_data = request.get_json(force=True, silent=True)
    # print("Logs")
    # print(request_data)
    if request_data is None:
        request_data = request.values
        # print(request_data)
    # print(request_data)
    # print(request_data.keys())
    logger.debug("Request received")
    nlines = len(request_data['audio'])
    url_list = [None] * nlines
    for idx in range(0,nlines):
        if "audioContent" in request_data['audio'][idx]:
            url_list[idx] = request_data['audio'][idx]['audioContent']
            urlFlag = False
        else:
            url_list[idx] = request_data['audio'][idx]['audioUri']
            urlFlag = True
    lang = lang_map[request_data["config"]["language"]["sourceLanguage"]]

    # try:
    #     outputs = worker_pool.starmap(ulca_decode_speech, zip(url_list, repeat(lang)))
    #     transcriptions = list(chain(*outputs))
    #     return transcriptions
    # except Exception as err:
    #     logger.error(err)
    #     return jsonify(status="failure", reason=str(err))
    # finally:
    #     # clear temp files in a background process
    #     for tmp_file in tmp_files:
    #         if tmp_file is not None:
    #             subprocess.Popen(["rm","-rf",tmp_file])

    try:
        out_list = []
        for url in url_list:
            out, tmp_files = ulca_decode_speech(url, lang, urlFlag)
            output = out.get_json()
            # print(output)
            # print(output.keys())
            out_list.append({"source":output['transcript']})
        return jsonify(status="SUCCESS", output=out_list)
    except Exception as err:
        logger.error(err)
        return jsonify(status="failure", reason=str(err))
    finally:
        # clear temp files in a background process
        for tmp_file in tmp_files:
            if tmp_file is not None:
                subprocess.Popen(["rm","-rf",tmp_file])

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    # app.run(host='0.0.0.0', port=1241, certfile='/home/monica/certificate.pem', keyfile='/home/monica/key.pem')
    socketio.run(app, host='0.0.0.0', port=8000)
