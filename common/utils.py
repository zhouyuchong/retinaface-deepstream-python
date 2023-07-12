'''
Author: zhouyuchong
Date: 2023-07-12 09:50:16
Description: 
LastEditors: zhouyuchong
LastEditTime: 2023-07-12 10:37:57
'''
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import ctypes
import sys
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

def long_to_uint64(l):
    value = ctypes.c_uint64(l & 0xffffffffffffffff).value
    return value

def cal_ratio(input_height, input_width, output_height, output_width):
    return max(output_height/input_height, output_width/input_width)