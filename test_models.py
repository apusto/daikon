import os
import subprocess


def training_run(model_name):
    cuda_devices = 6
    epochs = 6
    log_dir = "logs/" + model_name
    model_dir = "model/" + model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    command = "CUDA_VISIBLE_DEVICES=" + str(cuda_devices) + " daikon train --source ../bpe150k.codes.de " + \
              "--target ../bpe150k.codes.en --epochs=" + str(epochs) + " --log_to=" + log_dir + " " + \
              "--model_name=" + model_name + " --save_to=" + model_dir

    print(command)
    #subprocess.call(command, shell=True)


os.chdir(".")

for model in (#"original",
              #"original-dropout-keep-prob-0.2",
              #"original-dropout-keep-prob-0.5",
              #"original-dropout-keep-prob-0.8",
              ["bidirectional-no-dout-supress-unk"]):
              #"bidirectional-dropout-keep-prob-0.2",
              #"bidirectional-dropout-keep-prob-0.5"):
              #"bidirectional-dropout-keep-prob-0.8"):
    training_run(model)
