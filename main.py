# coding=utf-8

from utils import run_lib_flowgrad
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Rectified Flow Model configuration.", lock_config=True)
flags.DEFINE_enum("mode", 'flowgrad-edit', ["flowgrad-edit"], "Running mode.")
flags.DEFINE_string("text_prompt", None, "text prompt for editing")
flags.DEFINE_float("alpha", 0.7, "The coefficient to balance the edit loss and the reconstruction loss.")
flags.DEFINE_string("model_path", None, "Path to pre-trained model checkpoint.")
flags.DEFINE_string("image_path", None, "The path to the image that will be edited")
flags.DEFINE_string("output_folder", "output", "The folder name for storing output")
flags.mark_flags_as_required(["model_path", "text_prompt", "alpha", "image_path"])


text_prompts = ['A photo of an old face.','A photo of a sad face.','A photo of a smiling face.','A photo of an angry face.','A photo of a face with curly hair.']
image_paths = ['examples/original/00004.jpg',
 'examples/original/00008.jpg',
 'examples/original/00021.jpg',
 'examples/original/00037.jpg',
 'examples/original/00039.jpg',
 'examples/original/00070.jpg',
 'examples/original/00072.jpg',
 'examples/original/00078.jpg',
 'examples/original/00097.jpg',
 'examples/original/00098.jpg',
 'examples/original/00103.jpg',
 'examples/original/00105.jpg',
 'examples/original/00107.jpg',
 'examples/original/00117.jpg',
 'examples/original/00133.jpg',
 'examples/original/00134.jpg',
 'examples/original/00182.jpg',
 'examples/original/00185.jpg',
 'examples/original/00186.jpg']
model_path = './checkpoint_10.pth'

def main(argv):
  # if FLAGS.mode == "flowgrad-edit":
  #   run_lib_flowgrad.flowgrad_edit(FLAGS.config, FLAGS.text_prompt, FLAGS.alpha, FLAGS.model_path, FLAGS.image_path, FLAGS.output_folder)
  # else:
  #   raise ValueError(f"Mode {FLAGS.mode} not recognized.")

  output_dirs = ['ocfm/old', 'ocfm/sad', 'ocfm/smile', 'ocfm/angry', 'ocfm/curly']

  prompt = text_prompts[0]
  output_dir = output_dirs[0]

  metrics = run_lib_flowgrad.flowgrad_edit_batch(
       FLAGS.config, 
       model_path, 
       image_paths, 
       prompt, 
       output_dir
    )
  
  print(metrics)


if __name__ == "__main__":
  app.run(main)
