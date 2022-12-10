#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import torch  # import torch and sklearn at the very beginning to avoid incompatible errors
import sklearn

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

# from demo_net import demo
# from test_net import test  # action recognition
from test_gaze_net import test  # change to gaze estimation
# from train_net import train  # action recognition
from train_gaze_net import train  # change to gaze estimation
# from visualization import visualize


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (cfg.TENSORBOARD.MODEL_VIS.ENABLE or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    # if cfg.DEMO.ENABLE:
    #     demo(cfg)


if __name__ == "__main__":
    main()
