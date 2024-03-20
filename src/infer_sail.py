
import time
import numpy as np
import cv2
import sophon.sail as sail
import logging


class TimeAverager(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._cnt = 0
        self._total_time = 0
        self._total_samples = 0

    def record(self, usetime, num_samples=None):
        self._cnt += 1
        self._total_time += usetime
        if num_samples:
            self._total_samples += num_samples

    def get_average(self):
        if self._cnt == 0:
            return 0
        return self._total_time / float(self._cnt)

    def get_ips_average(self):
        if not self._total_samples or self._cnt == 0:
            return 0
        return float(self._total_samples) / self._total_time


def nearest_interpolate(input_array, new_shape):
    # 计算缩放因子
    input_shape = np.array(input_array.shape)
    output_shape = np.array(new_shape)
    scales = output_shape / input_shape

    # 生成新的索引网格
    new_indices = [np.arange(size) for size in new_shape]
    new_indices = np.meshgrid(*new_indices, indexing='ij')

    # 应用缩放因子到索引
    new_indices = [np.floor(indices / scale).astype(int) for indices, scale in zip(new_indices, scales)]

    # 使用最近邻插值
    return input_array[tuple(new_indices)]

class Predictor:
    def __init__(self, args):
        self.args = args
        if self.args.test_speed:
            self.cost_averager = TimeAverager()

        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1,1,3)

        if args.use_optic_flow:
            self.disflow = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
            width, height = self.net_w, self.net_h
            self.prev_gray = np.zeros((height, width), np.uint8)
            self.prev_cfd = np.zeros((height, width), np.float32)
            self.is_first_frame = True

    def prepare_input(self, image):
        input_image = cv2.resize(image, dsize=(self.net_w, self.net_h))
        input_image = (input_image.astype(np.float32) / 255.0 - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def run(self, img, bg):

        input_data = self.prepare_input(img)
        input_data = {self.input_name: input_data}

        if self.args.test_speed:
            start = time.time()

        output = self.net.process(self.graph_name, input_data)

        if self.args.test_speed:
            self.cost_averager.record(time.time() - start)

        output = output[self.output_names[0]]

        return self.postprocess(output, img, bg)

    def postprocess(self, pred_img, origin_img, bg):
        score_map = pred_img[0, 1, :, :]

        # post process
        if self.args.use_post_process:
            mask_original = score_map.copy()
            mask_original = (mask_original * 255).astype("uint8")
            _, mask_thr = cv2.threshold(mask_original, 240, 1,
                                        cv2.THRESH_BINARY)
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
            mask_erode = cv2.erode(mask_thr, kernel_erode)
            mask_dilate = cv2.dilate(mask_erode, kernel_dilate)
            score_map *= mask_dilate

        # optical flow
        if self.args.use_optic_flow:
            score_map = 255 * score_map
            cur_gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
            cur_gray = cv2.resize(cur_gray,
                                  (pred_img.shape[-1], pred_img.shape[-2]))
            optflow_map = optic_flow_process(cur_gray, score_map, self.prev_gray, self.prev_cfd, \
                    self.disflow, self.is_first_frame)
            self.prev_gray = cur_gray.copy()
            self.prev_cfd = optflow_map.copy()
            self.is_first_frame = False
            score_map = optflow_map / 255.

        score_map = score_map[np.newaxis, np.newaxis, ...]

        h, w = origin_img.shape[:2]
        score_map = nearest_interpolate(score_map, (1, 1, h, w))
        alpha = np.transpose(score_map.squeeze(1), [1, 2, 0])

        bg = cv2.resize(bg, (w, h))
        if bg.ndim == 2:
            bg = bg[..., np.newaxis]

        out = (alpha * origin_img + (1 - alpha) * bg).astype(np.uint8)
        return out


def human_seg_tracking(pre_gray, cur_gray, prev_cfd, dl_weights, disflow):
    """计算光流跟踪匹配点和光流图
    输入参数:
        pre_gray: 上一帧灰度图
        cur_gray: 当前帧灰度图
        prev_cfd: 上一帧光流图
        dl_weights: 融合权重图
        disflow: 光流数据结构
    返回值:
        is_track: 光流点跟踪二值图，即是否具有光流点匹配
        track_cfd: 光流跟踪图
    """
    check_thres = 8
    h, w = pre_gray.shape[:2]
    track_cfd = np.zeros_like(prev_cfd)
    is_track = np.zeros_like(pre_gray)
    flow_fw = disflow.calc(pre_gray, cur_gray, None)
    flow_bw = disflow.calc(cur_gray, pre_gray, None)
    flow_fw = np.round(flow_fw).astype(np.int)
    flow_bw = np.round(flow_bw).astype(np.int)
    y_list = np.array(range(h))
    x_list = np.array(range(w))
    yv, xv = np.meshgrid(y_list, x_list)
    yv, xv = yv.T, xv.T
    cur_x = xv + flow_fw[:, :, 0]
    cur_y = yv + flow_fw[:, :, 1]

    # 超出边界不跟踪
    not_track = (cur_x < 0) + (cur_x >= w) + (cur_y < 0) + (cur_y >= h)
    flow_bw[~not_track] = flow_bw[cur_y[~not_track], cur_x[~not_track]]
    not_track += (np.square(flow_fw[:, :, 0] + flow_bw[:, :, 0]) +
                  np.square(flow_fw[:, :, 1] + flow_bw[:, :, 1])) >= check_thres
    track_cfd[cur_y[~not_track], cur_x[~not_track]] = prev_cfd[~not_track]

    is_track[cur_y[~not_track], cur_x[~not_track]] = 1

    not_flow = np.all(np.abs(flow_fw) == 0,
                      axis=-1) * np.all(np.abs(flow_bw) == 0, axis=-1)
    dl_weights[cur_y[not_flow], cur_x[not_flow]] = 0.05
    return track_cfd, is_track, dl_weights


def human_seg_track_fuse(track_cfd, dl_cfd, dl_weights, is_track):
    """光流追踪图和人像分割结构融合
    输入参数:
        track_cfd: 光流追踪图
        dl_cfd: 当前帧分割结果
        dl_weights: 融合权重图
        is_track: 光流点匹配二值图
    返回
        cur_cfd: 光流跟踪图和人像分割结果融合图
    """
    fusion_cfd = dl_cfd.copy()
    is_track = is_track.astype(np.bool)
    fusion_cfd[is_track] = dl_weights[is_track] * dl_cfd[is_track] + (
        1 - dl_weights[is_track]) * track_cfd[is_track]
    # 确定区域
    index_certain = ((dl_cfd > 0.9) + (dl_cfd < 0.1)) * is_track
    index_less01 = (dl_weights < 0.1) * index_certain
    fusion_cfd[index_less01] = 0.3 * dl_cfd[index_less01] + 0.7 * track_cfd[
        index_less01]
    index_larger09 = (dl_weights >= 0.1) * index_certain
    fusion_cfd[index_larger09] = 0.4 * dl_cfd[index_larger09] + 0.6 * track_cfd[
        index_larger09]
    return fusion_cfd


def threshold_mask(img, thresh_bg, thresh_fg):
    dst = (img / 255.0 - thresh_bg) / (thresh_fg - thresh_bg)
    dst[np.where(dst > 1)] = 1
    dst[np.where(dst < 0)] = 0
    return dst.astype(np.float32)


def optic_flow_process(cur_gray, scoremap, prev_gray, pre_cfd, disflow,
                       is_init):
    """光流优化
    Args:
        cur_gray : 当前帧灰度图
        pre_gray : 前一帧灰度图
        pre_cfd  ：前一帧融合结果
        scoremap : 当前帧分割结果
        difflow  : 光流
        is_init : 是否第一帧
    Returns:
        fusion_cfd : 光流追踪图和预测结果融合图
    """
    h, w = scoremap.shape
    cur_cfd = scoremap.copy()

    if is_init:
        if h <= 64 or w <= 64:
            disflow.setFinestScale(1)
        elif h <= 160 or w <= 160:
            disflow.setFinestScale(2)
        else:
            disflow.setFinestScale(3)
        fusion_cfd = cur_cfd
    else:
        weights = np.ones((h, w), np.float32) * 0.3
        track_cfd, is_track, weights = human_seg_tracking(
            prev_gray, cur_gray, pre_cfd, weights, disflow)
        fusion_cfd = human_seg_track_fuse(track_cfd, cur_cfd, weights, is_track)

    return fusion_cfd
