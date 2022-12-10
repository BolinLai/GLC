#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.
    _C.DATA.GAUSSIAN_KERNEL = 15

    # If True, use global embed in the network
    _C.MVIT.GLOBAL_EMBED_ON = False

    pass
