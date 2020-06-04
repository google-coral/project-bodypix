# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import io
import sys
import termios
import threading
import time
import queue

import numpy as np
from PIL import Image

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstGL', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, GstBase, GstGL, GstVideo, Gtk

from OpenGL.arrays.arraydatatype import ArrayDatatype
from OpenGL.GLES3 import (
    glActiveTexture, glBindBuffer, glBindTexture, glBindVertexArray, glBlendEquation, glBlendFunc,
    glBufferData, glClear, glClearColor, glDeleteBuffers, glDeleteTextures, glDeleteVertexArrays,
    glDisable, glDrawElements, glEnable, glEnableVertexAttribArray, glGenBuffers, glGenTextures,
    glGenVertexArrays, glPixelStorei, glTexImage2D, glTexParameteri, glTexSubImage2D,
    glVertexAttribPointer)
from OpenGL.GLES3 import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER, GL_FUNC_ADD, GL_LINEAR, GL_NEAREST,
    GL_ONE_MINUS_SRC_ALPHA, GL_R16F, GL_R32F, GL_RED, GL_RGB, GL_RGBA16F, GL_RGBA, GL_SRC_ALPHA, GL_STATIC_DRAW, GL_TEXTURE0,
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TRIANGLES, GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_VERTEX_SHADER)

import ctypes
from ctypes import pythonapi
from ctypes.util import find_library

from pose_engine import PoseEngine, EDGES, BODYPIX_PARTS

# Color mapping for bodyparts
RED_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "right" in v]
GREEN_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "hand" in v or "torso" in v]
BLUE_BODYPARTS = [k for k,v in BODYPIX_PARTS.items() if "leg" in v or "arm" in v or "face" in v or "hand" in v]

Gst.init(None)

# ctypes imports for missing or broken introspection APIs.
libgstgl = ctypes.CDLL(find_library('gstgl-1.0'))
libgstgl.gst_gl_memory_get_texture_id.argtypes = [ctypes.c_void_p]
libgstgl.gst_gl_memory_get_texture_id.restype = ctypes.c_uint
GstGLFramebufferFunc = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.py_object)
libgstgl.gst_gl_framebuffer_draw_to_texture.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        GstGLFramebufferFunc, ctypes.py_object]
libgstgl.gst_gl_framebuffer_draw_to_texture.restype = ctypes.c_bool
libgstgl.gst_is_gl_memory_egl.argtypes = [ctypes.c_void_p]
libgstgl.gst_is_gl_memory_egl.restype = ctypes.c_bool

def get_gl_texture_id(buf):
    memory = buf.peek_memory(0)
    assert GstGL.is_gl_memory(memory)
    return libgstgl.gst_gl_memory_get_texture_id(hash(memory))

def is_egl_image(buf):
    memory = buf.peek_memory(0)
    assert GstGL.is_gl_memory(memory)
    return libgstgl.gst_is_gl_memory_egl(hash(memory))


POSITIONS = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ], dtype=np.float32)

TEXCOORDS = np.array([
         0.0, 0.0,
         1.0, 0.0,
         1.0, 1.0,
         0.0, 1.0,
    ], dtype=np.float32)

INDICES = np.array([
         0, 1, 2, 0, 2, 3
    ], dtype=np.uint16)

FRAGMENT_SHADER_SRC = '''
    precision mediump float;
    varying vec2 v_texcoord;
    uniform sampler2D image_tex, hm_tex, bg_tex;
    uniform int stage;
    uniform float ratio;
    uniform float heatmap_mul;

    // Clamps heatmap between [v0, v1] and rescales to [0,1]
    float sample_heatmap(float v0, float v1)
    {
        float value = texture2D(hm_tex, v_texcoord).a;
        float a = v0 / (v0 - v1);
        float b = 1.0 / (v1 - v0);
        return clamp(a + b * value, 0.0, 1.0);
    }

    vec4 stage0_background()
    {
        float heatmap = sample_heatmap(-2.3, -0.6);
        vec4 bg = texture2D(bg_tex, v_texcoord);
        vec4 image = texture2D(image_tex, v_texcoord);
        vec4 estimate = (bg * heatmap + image * (1.0 - heatmap));
        vec4 new_bg = bg * (1.0 - ratio) + ratio * estimate;
        return new_bg;
    }

    vec4 stage1_anon_background()
    {
        return texture2D(bg_tex, v_texcoord);
    }

    vec4 stage2_overlays()
    {
      float heatmap = sample_heatmap(-1.0, 1.0)*heatmap_mul;
      vec4 body_outline = vec4(texture2D(hm_tex, v_texcoord).xyz, 0.7*heatmap);
      return body_outline;
    }

    void main()
    {
        if (stage == 0) {
            gl_FragColor = stage0_background();
        } else if (stage == 1) {
            gl_FragColor = stage1_anon_background();
        } else if (stage == 2) {
            gl_FragColor = stage2_overlays();
        } else {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        }
    }
'''

KEYPOINT_SCORE_THRESHOLD = 0.2

SVG_HEADER     = '<svg width="{w}" height="{h}" version="1.1" >'
SVG_STYLES     = '''
    <style>
        .counter {{ font-size: {counter_size}px; font-family: sans-serif; }}
        .text_bg {{ fill: black; }}
        .text_fg {{ fill: white; }}
        .bbox {{ stroke: white; stroke-width: 2; fill: none;}}
        .kpcirc {{ fill: cyan; stroke: blue; }}
        .kpline {{ stroke: blue; stroke-width: 2; }}
        .whiteline {{ stroke: white; stroke-width: 2; }}
    </style>
    '''
SVG_BB_RECT    = ' <rect x="{x}" y="{y}" width="{w}" height="{h}" class="bbox" />'
SVG_KP_CIRC    = ' <circle cx="{cx}" cy="{cy}" r="5" class="kpcirc" />'
SVG_KP_LINE    = ' <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="kpline" />'
SVG_WHITE_LINE    = ' <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="whiteline" />'
SVG_TEXT       = '''
    <text x="{x}" y="{y}" dx="0.05em" dy="0.05em" class="{clazz} text_bg">{text}</text>
    <text x="{x}" y="{y}" class="{clazz} text_fg">{text}</text>
    '''
SVG_FOOTER     = '</svg>'

TOGGLE_SKELETONS = 's'
TOGGLE_BBOXES = 'b'
TOGGLE_ANON = 'a'
TOGGLE_HEATMAP = 'h'
TOGGLE_BODYPARTS = 'p'
TOGGLE_RESET = 'r'

class Callback:
    def __init__(self, engine, src_size, save_every_n_frames=-1, print_stats=False):
        self.engine = engine
        self.src_size = src_size
        self.save_every_n_frames = save_every_n_frames
        self.print_stats = print_stats
        self.inf_q = queue.SimpleQueue()
        self.trash = queue.SimpleQueue()
        self.trash_lock = threading.RLock()
        self.vinfo = GstVideo.VideoInfo()
        self.glcontext = None
        self.pool = None
        self.fbo = None
        self.default_shader = None
        self.hm_shader = None
        self.hm_tex_id = 0     # Instantaneous heatmap
        self.vao_id = 0
        self.positions_buffer = 0
        self.texcoords_buffer = 0
        self.vbo_indices_buffer = 0
        self.frames = 0
        self.reset_display_toggles()
        self.inf_times = collections.deque(maxlen=100)
        self.agg_times = collections.deque(maxlen=100)
        self.frame_times = collections.deque(maxlen=100)
        self.running = True
        self.gc_thread = threading.Thread(target=self.gc_loop)
        self.gc_thread.start()
        self.last_frame_time = time.monotonic()

    def reset_display_toggles(self):
        self.skeletons = True
        self.bboxes = True
        self.anon = False
        self.hm = True
        self.bodyparts = True

    def gc_loop(self):
        while self.running:
            try:
                buf = self.trash.get(timeout=0.1)
                self.trash.put(buf)
                buf = None
                self.empty_trash()
            except queue.Empty:
                pass

    # gl thread
    def empty_trash_gl(self, glcontext):
        while True:
            try:
                buf = self.trash.get(block=False)
                # Anyone trashing buffers must hold trash_lock to ensure
                # the last ref is dropped in this thread!
                with self.trash_lock:
                    buf = None
            except queue.Empty:
                break

    def empty_trash(self):
        self.glcontext.thread_add(self.empty_trash_gl)

    # Caller must hold trash_lock until its final ref to bufs is dropped.
    def trash_buffer(self, buf):
        self.trash.put(buf)

    # gl thread
    def init_gl(self, glcontext):
        #TODO deinit at some point
        assert not self.glcontext

        vert_stage = GstGL.GLSLStage.new_default_vertex(glcontext)
        frag_stage = GstGL.GLSLStage.new_with_string(glcontext,
            GL_FRAGMENT_SHADER,
            GstGL.GLSLVersion.NONE,
            GstGL.GLSLProfile.COMPATIBILITY | GstGL.GLSLProfile.ES,
            FRAGMENT_SHADER_SRC)
        self.hm_shader = GstGL.GLShader.new(glcontext)
        self.hm_shader.compile_attach_stage(vert_stage)
        self.hm_shader.compile_attach_stage(frag_stage)
        self.hm_shader.link()

        self.default_shader = GstGL.GLShader.new_default(glcontext)
        a_position = self.default_shader.get_attribute_location('a_position')
        a_texcoord = self.default_shader.get_attribute_location('a_texcoord')

        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)

        self.positions_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(POSITIONS), POSITIONS, GL_STATIC_DRAW)

        self.texcoords_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(TEXCOORDS), TEXCOORDS, GL_STATIC_DRAW)

        self.vbo_indices_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_indices_buffer)
        glBufferData(GL_ARRAY_BUFFER, ArrayDatatype.arrayByteCount(INDICES), INDICES, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_indices_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, self.positions_buffer);
        glVertexAttribPointer.baseFunction(a_position, 2, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.texcoords_buffer);
        glVertexAttribPointer.baseFunction(a_texcoord, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(a_position)
        glEnableVertexAttribArray(a_texcoord)

        glBindVertexArray(0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        hm_w, hm_h = self.get_heatmap_texture_size()

        texture_ids = glGenTextures(1)
        self.hm_tex_id = texture_ids

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.hm_tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, hm_w, hm_h, 0, GL_RGBA, GL_FLOAT, None)

        self.glcontext = glcontext

    # gl thread
    def init_fbo(self, glcontext, width, height):
        #TODO deinit at some point
        self.fbo = GstGL.GLFramebuffer.new_with_default_depth(self.glcontext, width, height)

    # gl thread
    def render_single_texture(self, tex):
        glBindVertexArray(self.vao_id)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, get_gl_texture_id(tex))
        self.default_shader.use()
        self.default_shader.set_uniform_1i('image_tex', 0)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)
        return True

    # gl thread
    def update_heatmap(self, glcontext, heatmap):
        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.hm_tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        heatmap.shape[1], heatmap.shape[0],
                        GL_RGBA, GL_FLOAT, heatmap)

    # Since the aspect ratio of the input image is not necessarily the
    # same as the inference input size and we rescale while maintaining
    # aspect ratio, we need to trim the heatmap to match the image
    # aspect ratio.
    def get_heatmap_texture_size(self):
        src_ratio = self.src_size[0]/self.src_size[1]
        inf_ratio = self.engine.image_width/self.engine.image_height
        result = [
          int(self.engine.heatmap_size[0]*min(1.0, src_ratio/inf_ratio)),
          int(self.engine.heatmap_size[1]*min(1.0, inf_ratio/src_ratio)),
        ]
        return result

    # gl thread
    def setup_scene(self, image_tex, bind_hm=True, bind_bg=True):
        glBindVertexArray(self.vao_id)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, image_tex)

        glActiveTexture(GL_TEXTURE0 + 1)
        glBindTexture(GL_TEXTURE_2D, self.hm_tex_id if bind_hm else 0)

        glActiveTexture(GL_TEXTURE0 + 2)
        glBindTexture(GL_TEXTURE_2D, get_gl_texture_id(self.bg_buf) if bind_bg else 0)

        self.hm_shader.use()
        self.hm_shader.set_uniform_1i('image_tex', 0)
        self.hm_shader.set_uniform_1i('hm_tex', 1)
        self.hm_shader.set_uniform_1i('bg_tex', 2)


    # gl thread
    def render_background(self, args):
        vid_buf, new_bg_buf = args

        # This is the mixing ratio of the instantaneous background estimate and
        # the current aggregated background estimate.
        ratio = max(0.001, 1.0 / max(1.0, self.frames / 2.0))
        self.setup_scene(get_gl_texture_id(vid_buf))
        self.hm_shader.set_uniform_1i('stage', 0)
        self.hm_shader.set_uniform_1f('ratio', ratio)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)

        # Drop the ref to old background here in the GL thread.
        self.bg_buf = new_bg_buf
        return True

    # gl thread
    def render_anon_background(self, arg):
        self.setup_scene(0, bind_hm=False)
        self.hm_shader.set_uniform_1i('stage', 1)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)
        return True

    # gl thread
    def render_overlays(self, arg):
        self.setup_scene(0, bind_bg=False, bind_hm=True)
        self.hm_shader.set_uniform_1i('stage', 2)
        self.hm_shader.set_uniform_1f('heatmap_mul', 1.0 if self.hm else 0.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBlendEquation(GL_FUNC_ADD)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)
        glDisable(GL_BLEND)
        return True

    # gl thread
    def render_to_texture_gl(self, glcontext, dst, render_func, render_arg):
        libgstgl.gst_gl_framebuffer_draw_to_texture(
            hash(self.fbo),
            hash(dst.peek_memory(0)),
            GstGLFramebufferFunc(render_func),
            render_arg)
        meta = GstGL.buffer_add_gl_sync_meta(glcontext, dst)
        meta.set_sync_point(glcontext)

    def render_to_texture(self, dst, render_func, render_arg):
        self.glcontext.thread_add(self.render_to_texture_gl, dst, render_func, render_arg)

    def ensure_buffers_setup(self, vid_caps):
        assert self.glcontext
        if self.pool:
            return

        self.vinfo.from_caps(vid_caps)

        self.glcontext.thread_add(self.init_fbo, self.vinfo.width, self.vinfo.height)

        self.pool = GstGL.GLBufferPool.new(self.glcontext)
        config = self.pool.get_config()
        Gst.BufferPool.config_set_params(config, vid_caps, self.vinfo.size, 0, 0)
        Gst.BufferPool.config_add_option(config, GstVideo.BUFFER_POOL_OPTION_VIDEO_META)
        self.pool.set_config(config)
        #TODO set inactive at some point
        self.pool.set_active(True)

        self.bg_buf = self.acquire_pooled_buffer()
        def clear_texture(args):
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
        self.render_to_texture(self.bg_buf, clear_texture, None)

    def acquire_pooled_buffer(self):
        assert self.pool
        res, buf = self.pool.acquire_buffer()
        assert res == Gst.FlowReturn.OK
        return buf

    def get_output_buffer(self, vid_caps, pts=Gst.CLOCK_TIME_NONE):
        self.ensure_buffers_setup(vid_caps)
        buf = self.acquire_pooled_buffer()
        buf.pts = pts
        return buf

    def generate_svg(self, poses, inference_box, text):
        box_x, box_y, box_w, box_h = inference_box
        scale_x, scale_y = self.vinfo.width / box_w, self.vinfo.height / box_h

        svg = io.StringIO()
        svg.write(SVG_HEADER.format(w=self.vinfo.width , h=self.vinfo.height))
        svg.write(SVG_STYLES.format(counter_size=int(3 * self.vinfo.height / 100)))

        pose_count = 0
        # Iterate over poses and keypoints just once.
        for pose in poses:
            xys = {}
            bbox = [sys.maxsize, sys.maxsize, 0, 0]
            good_poses = 0
            for label, keypoint in pose.keypoints.items():
                if keypoint.score < KEYPOINT_SCORE_THRESHOLD:
                    continue
                good_poses += 1

                # Offset and scale to source coordinate space.
                kp_y = int((keypoint.yx[0] - box_y) * scale_y)
                kp_x = int((keypoint.yx[1] - box_x) * scale_x)

                bbox[0] = int(min(bbox[0], kp_y))
                bbox[1] = int(min(bbox[1], kp_x))
                bbox[2] = int(max(bbox[2], kp_y))
                bbox[3] = int(max(bbox[3], kp_x))

                xys[label] = (kp_x, kp_y)
                # Keypoint.
                if self.skeletons:
                    svg.write(SVG_KP_CIRC.format(cx=kp_x, cy=kp_y))

            y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            # Bounding box.
            if good_poses:
                if self.bboxes:
                    svg.write(SVG_BB_RECT.format(x=x, y=y, w=w, h=h))
                pose_count += 1

            for a, b in EDGES:
                if a not in xys or b not in xys:
                    continue
                ax, ay = xys[a]
                bx, by = xys[b]
                # Skeleton.
                if self.skeletons:
                    svg.write(SVG_KP_LINE.format(x1=ax, y1=ay, x2=bx, y2=by))

        svg.write(SVG_TEXT.format(x=0, y='1em', clazz='counter', text=text))

        svg.write(SVG_FOOTER)
        return svg.getvalue()

    # run_inference and aggregate_buffers runs in separate threads
    # to allow parallelization between TPU and CPU processing.
    def run_inference(self, inf_buf, inf_caps):
        start = time.monotonic()
        inference_time, data = self.engine.run_inference(inf_buf)

        # Underlying output tensor is owned by engine and if we want to
        # keep the data around while running another inference we have
        # to make our own copy.
        self.inf_q.put(data.copy())

        if self.save_every_n_frames > 0 and self.frames % self.save_every_n_frames == 0:
            meta = GstVideo.buffer_get_video_meta(inf_buf)
            result, mapinfo = inf_buf.map(Gst.MapFlags.READ)
            image = Image.frombytes('RGB', (meta.width, meta.height), mapinfo.data)
            image.save('inf_{:05d}.png'.format(self.frames))
            inf_buf.unmap(mapinfo)
        elapsed = time.monotonic() - start
        self.inf_times.append(elapsed)

    # Called on GStreamer streaming thread. Returns
    # (svg, out_buf) tuple. Caller guarantees out_buf is freed on
    # the gl thread or there's risk of deadlock.
    def aggregate_buffers(self, inf_buf, inf_caps, vid_buf, vid_caps, box):
        self.frames += 1
        start = time.monotonic()
        data = self.inf_q.get()
        poses, heatmap, bodyparts = self.engine.ParseOutputs(data)

        # Clip heatmaps according to aspect ratio difference between camera
        # and inference input size
        hm_crop_size = self.get_heatmap_texture_size()
        hbox_topleft = [
          (self.engine.heatmap_size[1]-hm_crop_size[1])//2,
          (self.engine.heatmap_size[0]-hm_crop_size[0])//2,
        ]
        heatmap = heatmap[
          hbox_topleft[0]:hbox_topleft[0]+hm_crop_size[1],
          hbox_topleft[1]:hbox_topleft[1]+hm_crop_size[0]
        ]
        bodyparts = bodyparts[
          hbox_topleft[0]:hbox_topleft[0]+hm_crop_size[1],
          hbox_topleft[1]:hbox_topleft[1]+hm_crop_size[0]
        ]

        if self.bodyparts:
          # Turn bodyparts into different hues, overall heatmap
          # acts as opacity mask (alpha channel)
          rgba_heatmap = np.dstack([
            (np.sum(bodyparts[:,:,RED_BODYPARTS], axis=2)-0.5)*100,
            (np.sum(bodyparts[:,:,GREEN_BODYPARTS], axis=2)-0.5)*100,
            (np.sum(bodyparts[:,:,BLUE_BODYPARTS], axis=2)-0.5)*100,
            heatmap,
          ])
        else:
          rgba_heatmap = np.dstack([
            np.ones_like(heatmap),
            np.zeros_like(heatmap),
            np.zeros_like(heatmap),
            heatmap])

        # Upload heatmap
        self.glcontext.thread_add(self.update_heatmap, rgba_heatmap)

        # Render new background.
        new_bg_buf = self.get_output_buffer(vid_caps)
        self.render_to_texture(new_bg_buf, self.render_background, (vid_buf, new_bg_buf))

        # Render output image.
        if self.anon:
            out_buf = self.get_output_buffer(vid_caps, vid_buf.pts)
            self.render_to_texture(out_buf, self.render_anon_background, None)
        else:
            # NXP has an optimization where camera frames in dmabufs are wrapped
            # as EGLImages to be used as textures without copies or extra draws.
            # This works as source for drawing, but they can't be drawn to. So
            # if we get an EGLImage here we need to allocate our own output buffer
            # so that we can draw to it. When glvideoflip flips the video this is
            # already done there, in which case we can draw straight to vid_buf.
            if is_egl_image(vid_buf):
                out_buf = self.get_output_buffer(vid_caps, vid_buf.pts)
                self.render_to_texture(out_buf, self.render_single_texture, vid_buf)
            else:
                out_buf = vid_buf

        self.render_to_texture(out_buf, self.render_overlays, None)

        # Useful for debugging. Simply call this with any GL buffer as the last
        # parameter to draw that texture to the first parameter (e.g. out_buf).
        # self.render_to_texture(out_buf, self.render_single_texture, self.bg_buf)

        now_time = time.monotonic()

        self.agg_times.append(now_time - start)
        self.frame_times.append(now_time - self.last_frame_time)
        self.last_frame_time = now_time

        inf_time = 1000 * sum(self.inf_times) / len(self.inf_times)
        frame_time = 1000 * sum(self.frame_times) / len(self.frame_times)
        with self.trash_lock:
            self.trash_buffer(vid_buf)
            vid_buf = None
        text = 'Inference: {:.2f} ms  Total frame time: {:.2f} ms ({:.2f} FPS) Current occupancy: {:d}'.format(
                    inf_time, frame_time, 1000 / frame_time, len(poses))
        if self.print_stats and (self.frames % 100) == 0: print(text)

        # Generate SVG overlay.
        svg = self.generate_svg(poses, box, text)

        return svg, out_buf

    def handle_stdin_char(self, char):
        if char == TOGGLE_SKELETONS:
            self.skeletons = not self.skeletons
        elif char == TOGGLE_BBOXES:
            self.bboxes = not self.bboxes
        elif char == TOGGLE_ANON:
            self.anon = not self.anon
        elif char == TOGGLE_HEATMAP:
            self.hm = not self.hm
        elif char == TOGGLE_BODYPARTS:
            if not self.hm and self.bodyparts:
              self.hm = not self.hm
            else:
              self.bodyparts = not self.bodyparts
              self.hm = self.hm or self.bodyparts
        elif char == TOGGLE_RESET:
          self.frames = 1


class GstPipeline:
    def __init__(self, pipeline, callback):
        self.callback = callback

        self.pipeline = Gst.parse_launch(pipeline)
        self.overlaysink = self.pipeline.get_by_name('overlaysink')

        # We're high latency on higher resolutions, don't drop our late frames.
        # TODO: Maybe make this dynamic?
        sinkelement = self.overlaysink.get_by_interface(GstVideo.VideoOverlay)
        sinkelement.set_property('sync', False)
        sinkelement.set_property('qos', False)

        inference = self.pipeline.get_by_name('inf')
        inference.callback = callback.run_inference

        aggregator = self.pipeline.get_by_name('agg')
        aggregator.buffers_aggregated_callback = self.on_buffers_aggregated
        aggregator.trash_lock = self.callback.trash_lock
        aggregator.trash_buffer_callback = self.callback.trash_buffer

        # Set up a pipeline bus watch to catch errors.
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        self.box = None
        self.setup_window()

    def run(self):
        # Set to READY and wait, get OpenGL context.
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)

        assert self.overlaysink.glcontext
        self.overlaysink.glcontext.thread_add(self.callback.init_gl)

        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            old_mode = termios.tcgetattr(fd)
            new_mode = termios.tcgetattr(fd)
            new_mode[3] = new_mode[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new_mode)
            GLib.io_add_watch(sys.stdin, GLib.IO_IN, self.on_stdin)

        try:
            # Run pipeline.
            self.pipeline.set_state(Gst.State.PLAYING)
            Gtk.main()
        except:
            pass
        finally:
            if sys.stdin.isatty():
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_mode)

        # Clean up.
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
        self.callback.empty_trash()
        self.callback.running = False
        self.pipeline.set_state(Gst.State.NULL)
        while GLib.MainContext.default().iteration(False):
            pass

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            # Try to seek back to the beginning. If pipeline
            # isn't seekable we shouldn't get here in the first
            # place so if seek fails just quit.
            if not self.pipeline.seek_simple(Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                    0):
                Gtk.main_quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write('Warning: %s: %s\n' % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write('Error: %s: %s\n' % (err, debug))
            Gtk.main_quit()
        return True

    def on_buffers_aggregated(self, inf_buf, inf_caps, vid_buf, vid_caps):
        svg, out_buf = self.callback.aggregate_buffers(inf_buf, inf_caps,
                vid_buf, vid_caps, self.get_box())
        if svg:
            self.overlaysink.set_property('svg', svg)
        return out_buf

    # Returns a cached representation of the inference scaling box.
    def get_box(self):
        if not self.box:
            glbox = self.pipeline.get_by_name('glbox')
            assert glbox
            glbox = glbox.get_by_name('filter')
            self.box = (glbox.get_property('x'), glbox.get_property('y'),
                        glbox.get_property('width'), glbox.get_property('height'))
        return self.box

    # stdin is ready for reading
    def on_stdin(self, file, cond):
        char = file.read(1)
        if len(char) == 1:
            self.callback.handle_stdin_char(char)
            return True
        return False

    def setup_window(self):
        # Only set up our own window if we have Coral overlay sink in the pipeline.
        if not self.overlaysink:
            return

        gi.require_version('GstGL', '1.0')
        gi.require_version('GstVideo', '1.0')
        from gi.repository import GstGL, GstVideo

        # Needed to commit the wayland sub-surface.
        def on_gl_draw(sink, widget):
            widget.queue_draw()

        # Needed to account for window chrome etc.
        def on_widget_configure(widget, event, overlaysink):
            allocation = widget.get_allocation()
            overlaysink.set_render_rectangle(allocation.x, allocation.y,
                    allocation.width, allocation.height)
            return False

        window = Gtk.Window()
        window.fullscreen()

        drawing_area = Gtk.DrawingArea()
        window.add(drawing_area)
        drawing_area.realize()

        self.overlaysink.connect('drawn', on_gl_draw, drawing_area)

        # Wayland window handle.
        wl_handle = self.overlaysink.get_wayland_window_handle(drawing_area)
        self.overlaysink.set_window_handle(wl_handle)

        # Wayland display context wrapped as a GStreamer context.
        wl_display = self.overlaysink.get_default_wayland_display_context()
        self.overlaysink.set_context(wl_display)

        drawing_area.connect('configure-event', on_widget_configure, self.overlaysink)
        window.connect('delete-event', Gtk.main_quit)
        window.show_all()

        # The appsink pipeline branch must use the same GL display as the screen
        # rendering so they get the same GL context. This isn't automatically handled
        # by GStreamer as we're the ones setting an external display handle.
        def on_bus_message_sync(bus, message, overlaysink):
            if message.type == Gst.MessageType.NEED_CONTEXT:
                _, context_type = message.parse_context_type()
                if context_type == GstGL.GL_DISPLAY_CONTEXT_TYPE:
                    sinkelement = overlaysink.get_by_interface(GstVideo.VideoOverlay)
                    gl_context = sinkelement.get_property('context')
                    if gl_context:
                        display_context = Gst.Context.new(GstGL.GL_DISPLAY_CONTEXT_TYPE, True)
                        GstGL.context_set_gl_display(display_context, gl_context.get_display())
                        message.src.set_context(display_context)
            return Gst.BusSyncReply.PASS

        bus = self.pipeline.get_bus()
        bus.set_sync_handler(on_bus_message_sync, self.overlaysink)

class Aggregator(GstBase.Aggregator):
    SINK_CAPS = 'video/x-raw(memory:GLMemory),format=RGBA,width=[1,{max_int}],height=[1,{max_int}],texture-target=2D'
    SINK_CAPS += '; video/x-raw,format=RGB,width=[1,{max_int}],height=[1,{max_int}]'
    SINK_CAPS = Gst.Caps.from_string(SINK_CAPS.format(max_int=GLib.MAXINT))
    SRC_CAPS = 'video/x-raw(memory:GLMemory),format=RGBA,width=[1,{max_int}],height=[1,{max_int}],texture-target=2D'
    SRC_CAPS = Gst.Caps.from_string(SRC_CAPS.format(max_int=GLib.MAXINT))
    __gstmetadata__ = ('<longname>', '<class>', '<description>', '<author>')
    __gsttemplates__ = (
            Gst.PadTemplate.new_with_gtype("sink_%u",
                                Gst.PadDirection.SINK,
                                Gst.PadPresence.REQUEST,
                                SINK_CAPS,
                                GstBase.AggregatorPad.__gtype__),
            Gst.PadTemplate.new_with_gtype("src",
                                Gst.PadDirection.SRC,
                                Gst.PadPresence.ALWAYS,
                                SRC_CAPS,
                                GstBase.AggregatorPad.__gtype__))

    # TODO: report actual latency

    def __init__(self):
        self.vid_pad = None
        self.inf_pad = None
        self.vid_caps = None
        self.inf_caps = None
        self.buffers_aggregated_callback = None
        self.trash_buffer_callback = None

    # TODO: gracefully handle errors.
    def ensure_pads_found(self):
        if self.vid_pad and self.inf_pad:
            return

        for pad in self.sinkpads:
            caps = pad.get_current_caps()
            feature = caps.get_features(0).get_nth(0)
            struct = caps.get_structure(0)
            if feature == Gst.CAPS_FEATURE_MEMORY_SYSTEM_MEMORY:
                self.inf_pad = pad
                self.inf_caps = caps
            elif feature == GstGL.CAPS_FEATURE_MEMORY_GL_MEMORY:
                self.vid_pad = pad
                self.vid_caps = caps
        assert self.vid_pad and self.inf_pad

    def do_aggregate(self, timeout):
        self.ensure_pads_found()
        assert self.buffers_aggregated_callback
        assert self.trash_lock
        assert self.trash_buffer_callback

        vid_buf = self.vid_pad.pop_buffer()
        inf_buf = self.inf_pad.pop_buffer()

        # If either input is empty we're EOS (end of stream).
        if not vid_buf or not inf_buf:
            with self.trash_lock:
                self.trash_buffer_callback(vid_buf)
                vid_buf = None
            return Gst.FlowReturn.EOS

        # Get the output buffer to push downstream.
        out_buf = self.buffers_aggregated_callback(inf_buf, self.inf_caps, vid_buf, self.vid_caps)

        # Unref the inputs ASAP. Drop the final video ref in the GL thread
        # or there's deadlock between the GL lock and the Python GIL.
        with self.trash_lock:
            self.trash_buffer_callback(vid_buf)
            vid_buf = None
        inf_buf = None

        # Push buffer downstream.
        ret = self.finish_buffer(out_buf)

        # Finally drop the ref to the output buffer, again in the GL thread.
        with self.trash_lock:
            self.trash_buffer_callback(out_buf)
            out_buf = None

        return ret

    def do_fixate_src_caps (self, caps):
        self.ensure_pads_found()
        return self.vid_caps

class Inference(GstBase.BaseTransform):
    CAPS = 'video/x-raw,format=RGB,width=[1,{max_int}],height=[1,{max_int}]'
    CAPS = Gst.Caps.from_string(CAPS.format(max_int=GLib.MAXINT))
    __gstmetadata__ = ('<longname>', '<class>', '<description>', '<author>')
    __gsttemplates__ = (Gst.PadTemplate.new('sink',
                            Gst.PadDirection.SINK,
                            Gst.PadPresence.ALWAYS,
                            CAPS),
                        Gst.PadTemplate.new('src',
                            Gst.PadDirection.SRC,
                            Gst.PadPresence.ALWAYS,
                            CAPS)
                        )

    # TODO: report actual latency

    def __init__(self):
        self.caps = None
        self.set_passthrough(False)
        self.callback = None

    def do_set_caps(self, in_caps, out_caps):
        assert in_caps.is_equal(out_caps)
        self.caps = in_caps
        return True

    def do_transform_ip(self, buf):
        assert self.callback
        self.callback(buf, self.caps)
        return Gst.FlowReturn.OK


def register_elements(plugin):
    gtype = GObject.type_register(Aggregator)
    Gst.Element.register(plugin, 'aggregator', 0, gtype)
    gtype = GObject.type_register(Inference)
    Gst.Element.register(plugin, 'inference', 0, gtype)
    return True

Gst.Plugin.register_static(
    Gst.version()[0], Gst.version()[1], # GStreamer version
    '',                                 # name
    '',                                 # description
    register_elements,                  # init_func
    '',                                 # version
    'unknown',                          # license
    '',                                 # source
    '',                                 # package
    ''                                  # origin
)

def run_pipeline(cb, src_size, inference_size, video_src,
            h264=False, jpeg=False, mirror=False):
    SRC_CAPS = '{format},width={width},height={height},framerate=30/1'
    INF_CAPS = 'video/x-raw,format=RGB,width={width},height={height}'
    direction = 'horiz' if mirror else 'identity'
    PIPELINE = 'aggregator name=agg ! glsvgoverlaysink name=overlaysink \n'
    if video_src.startswith('/dev/video'):
        PIPELINE += 'v4l2src device={video_src} ! {src_caps}\n ! {lq} ! {decoder}'
        if jpeg:
            src_format = 'image/jpeg'
            decoder = 'decodebin'
        elif h264:
            src_format = 'video/x-h264'
            decoder = 'decodebin'
        else:
            src_format = 'video/x-raw'
            decoder = 'identity'
    else:
        PIPELINE += 'filesrc location={video_src} ! {decoder}'
        src_format = ''
        decoder = 'decodebin'
    PIPELINE += """ ! glupload ! glvideoflip video-direction={direction} ! tee name=t
           t. ! {q} ! glfilterbin filter=glbox name=glbox ! {inf_caps} !
                {q} ! inference name=inf ! agg.
           t. ! {q} ! agg.
        """

    src_caps = SRC_CAPS.format(format=src_format, width=src_size[0], height=src_size[1])
    inf_caps = INF_CAPS.format(width=inference_size[0], height=inference_size[1])
    q = 'queue max-size-buffers=1'
    lq = 'queue max-size-buffers=1 leaky=downstream'
    pipeline = PIPELINE.format(src_caps=src_caps, inf_caps=inf_caps, direction=direction,
            lq=lq, q=q, video_src=video_src, decoder=decoder)

    print('\nGstreamer pipeline:\n')
    print(pipeline)
    pipeline = GstPipeline(pipeline, cb)
    pipeline.run()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--width', help='Source width', default='1920')
    parser.add_argument('--height', help='Source height', default='1080')
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use video/jpeg input', action='store_true')
    args = parser.parse_args()

    if args.h264 and args.jpeg:
        print('Error: both mutually exclusive options h264 and jpeg set')
        sys.exit(1)

    default_model = 'models/bodypix_mobilenet_v1_075_1024_768_16_quant_edgetpu_decoder.tflite'
    model = args.model if args.model else default_model
    print('Model: {}'.format(model))

    engine = PoseEngine(model)
    inference_size = (engine.image_width, engine.image_height)
    print('Inference size: {}'.format(inference_size))

    src_size = (int(args.width), int(args.height))
    if args.videosrc.startswith('/dev/video'):
        print('Source size: {}'.format(src_size))

    print('Toggle mode keys:')
    print(' Toggle skeletons: ', TOGGLE_SKELETONS)
    print(' Toggle bounding boxes: ', TOGGLE_BBOXES)
    print(' Toggle anonymizer mode: ', TOGGLE_ANON)
    print(' Toggle heatmaps: ', TOGGLE_HEATMAP)
    print(' Toggle bodyparts: ', TOGGLE_BODYPARTS)
    run_pipeline(Callback(engine, src_size, save_every_n_frames=-1, print_stats=True),
      src_size, inference_size, video_src=args.videosrc, h264=args.h264, jpeg=args.jpeg,
            mirror=args.mirror)

if __name__== "__main__":
    main()
