#!/usr/bin/env python3
"""
CARLA 0.9.15 –  manual‑driving client **with real‑time speed‑limit sign detection**
-------------------------------------------------------------------------------
► Steering / driving controls (identical to official manual_control.py)
    W / S        throttle / brake
    A / D        steer left / right
    Q            toggle reverse gear
    Space        hand‑brake
    P            toggle autopilot
    ESC / Ctrl+Q quit


► Traffic‑sign recognition
    * RGB camera mounted on hood (1280 × 720)
    *  YOLOv3‑tiny custom model (3 classes: 30, 60, 90 km/h)
    * Detection runs every 5 video frames using OpenCV DNN (CPU by default).

Model files are loaded from models/ relative to the repository root.

Requires:  `pip install pygame opencv‑python numpy carla==0.9.*`
"""

# ===========================================================================
#  Imports
# ===========================================================================
import argparse
import glob
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import cv2
import weakref             # to avoid circular refs in camera callback
from math import tan, radians
import carla
from typing import Optional
from carla import ColorConverter as cc
from pygame.locals import (
    K_a, K_d, K_w, K_s, K_q, K_p, K_SPACE,
    K_ESCAPE, KMOD_CTRL, K_r, K_UP, K_DOWN, K_LEFT, K_RIGHT,
    K_n
)

# ===========================================================================
#  Paths to the custom YOLO‑tiny files
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
CFG_PATH = PROJECT_ROOT / "models" / "yolov3-tiny-obj.cfg"
WEIGHTS_PATH = PROJECT_ROOT / "models" / "yolov3-tiny-obj_5000.weights"

# class‑id -> speed‑limit (km/h)
SIGN2KMH = {0: 30, 1: 60, 2: 90}
CONF_THRESH = 0.80
RUN_EVERY_N = 5          # detect every N frames

# Window / camera resolution
WIN_WIDTH, WIN_HEIGHT = 1280, 720

# ===========================================================================
#  Helper – simple HUD overlay using pygame
# ===========================================================================
class HUD:
    def __init__(self, display):
        self.display = display
        self.font_big = pygame.font.Font(pygame.font.get_default_font(), 72)
        self.font_med = pygame.font.Font(pygame.font.get_default_font(), 48)

    def draw(self, limit_icon):
        if limit_icon:      # center-top
            self.display.blit(
                limit_icon,
                (WIN_WIDTH // 2 - limit_icon.get_width() // 2, 10)
            )

# ======================================================================
#  Camera callback with on-frame YOLO + overlay (no more flicker)
# ======================================================================
class CameraProcessor:
    def __init__(self, developer:bool = False):
        self.developer = developer
        self.last_conf = 0.0     # confidence of last detection
        self.last_limit = ""
        self.last_detect_time = 0.0
        self.surface = None
        self.frame_id = 0
        self.raw_conf  = 0.0      # highest-conf sign in *this* frame
        self.raw_cls   = None
        self.raw_time  = 0.0      # when that frame was processed
        if not (CFG_PATH.exists() and WEIGHTS_PATH.exists()):
            raise FileNotFoundError('YOLO cfg/weights not found in models/')
        # tiny yolo
        self._net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        ln = self._net.getUnconnectedOutLayers()
        self._layers = [self._net.getLayerNames()[i - 1] for i in ln]

        # store last set of boxes+labels so we can redraw each frame
        self._boxes = []  # list of (x1, y1, x2, y2, label, confidence)

    # ------------------------------------------------------------------
    def _detect(self, bgr):
        """
        Run tiny-YOLO once and update self._boxes with all detections.
        Each entry is (x1, y1, x2, y2, label_text, confidence).
        """
        # --- reset “raw” info BEFORE running detection -----------------
        self.raw_conf = 0.0          # <-- add
        self.raw_cls  = None         # <-- add
        # ---------------------------------------------------------------
        self._boxes.clear()
        H, W = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1/255.0, (608, 608), swapRB=True, crop=False)
        self._net.setInput(blob)
        outs = self._net.forward(self._layers)

        best_conf_any,  best_cls_any  = 0.0, None   # 〈– NEW
        best_conf_strong, best_cls_strong = 0.0, None
        for out in outs:
            for det in out:
                scores = det[5:]
                cls_id = int(np.argmax(scores))
                conf = float(scores[cls_id])
                if cls_id not in SIGN2KMH:
                    continue
                # ------------- bookkeeping for ANY confidence ------------
                if conf > best_conf_any:
                    best_conf_any, best_cls_any = conf, cls_id
                # ---------------------------------------------------------
                # keep only strong ones for drawing
                if conf < CONF_THRESH:
                    continue
                # bounding box (center_x, center_y, width, height) scaled to original
                cx, cy, w, h = det[0:4] * np.array([W, H, W, H])
                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                x2, y2 = int(cx + w / 2), int(cy + h / 2)

                label_text = f"{SIGN2KMH[cls_id]} km/h {conf*100:.0f}%"
                self._boxes.append((x1, y1, x2, y2, label_text, conf))

                # remember highest-confidence sign ⇒ HUD
                if conf > best_conf_strong:
                    best_conf_strong, best_cls_strong = conf, cls_id
        
        # If we found at least one valid sign, update last_limit + timestamp
        # ---- expose “raw” info so the main loop can see low-conf frames ---
        if best_conf_any > 0.0:
            self.raw_conf = best_conf_any
            self.raw_cls  = best_cls_any
            self.raw_time = time.time()

        # ---- update the *strong* (≥ threshold) HUD value ------------------
        if best_conf_strong > 0.0:
            self.last_conf      = best_conf_strong
            self.last_limit     = str(SIGN2KMH[best_cls_strong])
            self.last_detect_time = self.raw_time
    def __call__(self, image: carla.Image):
        # 1) Convert CARLA raw → H×W×4 → take only first three channels (BGR in memory)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]

        # 2) Every N frames, run detection on a copy
        self.frame_id += 1
        if self.frame_id % RUN_EVERY_N == 0:
            # make a writable BGR copy for detection, but keep original arr unchanged
            bgr_for_detect = arr[:, :, ::-1].copy()
            self._detect(bgr_for_detect)

        # 3) Now draw boxes (from self._boxes) on a fresh BGR copy every frame
        bgr_draw = arr[:, :, ::-1].copy()
        if self.developer:
            for (x1, y1, x2, y2, label_text, _) in self._boxes:
                color = (0, 255, 0)
                cv2.rectangle(bgr_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    bgr_draw,
                    label_text,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # 4) Convert annotated BGR back to RGB for pygame
        rgb = bgr_draw[:, :, ::-1]

        # 5) Create a pygame Surface (note: pygame expects (width, height) order)
        self.surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

# ===========================================================================
#  Keyboard -> VehicleControl helper
# ===========================================================================
class KeyboardControl:
    def __init__(self):
        self.control = carla.VehicleControl()
        self.reverse = False

    def parse(self):
        keys = pygame.key.get_pressed()
        ctl = self.control
        # reset each tick
        ctl.throttle = ctl.brake = 0.0
        ctl.steer = 0.0
        ctl.hand_brake = False

        if keys[K_a] or keys[K_LEFT]:
            ctl.steer = -1.0
        if keys[K_d] or keys[K_RIGHT]:
            ctl.steer = 1.0
        if keys[K_w] or keys[K_UP]:
            ctl.throttle = 1.0
        if keys[K_s] or keys[K_DOWN]:
            ctl.brake = 1.0
        if keys[K_SPACE]:
            ctl.hand_brake = True
        if keys[K_q]:
            self.reverse = not self.reverse
        ctl.reverse = self.reverse
        return ctl

# ===========================================================================
#  Main game loop
# ===========================================================================
# helper that works on every CARLA version
def _autopilot_enabled(vh):
    attr = getattr(vh, "is_autopilot_enabled", None)
    return attr() if callable(attr) else False




def main() -> None:
    # ── argument parsing & basic setup ───────────────────────────────
    parser = argparse.ArgumentParser(description="CARLA manual sign-aware client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("-a", "--autopilot", action="store_true")
    #developer overlay switch
    parser.add_argument("--developer", action="store_true",help="Show bounding boxes & diagnostic texts")
    args = parser.parse_args()

    pygame.init()
    pygame.mixer.init()
    display = pygame.display.set_mode(
        (WIN_WIDTH, WIN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    clock = pygame.time.Clock()
    # -------------------------------------------------------------------
    #  Sign icons (PNG), loaded from assets/ and resized to 64×64.
    # -------------------------------------------------------------------
    global SIGN_ICONS
    SIGN_ICONS = {}
    for kmh in (30, 60, 90):
        try:
            img = pygame.image.load(str(ASSETS_DIR / f"sign_{kmh}.png")).convert_alpha()
            SIGN_ICONS[kmh] = pygame.transform.smoothscale(img, (64, 64))
        except (pygame.error, FileNotFoundError):
            SIGN_ICONS[kmh] = None      # fallback to text if image missing

    for kmh, icon in SIGN_ICONS.items():
        if icon is None:
            print(f"[WARN] sign_{kmh}.png not found or failed to load.")
        else:
            print(f"[OK]   sign_{kmh}.png loaded.")

    # ── connect to simulator ─────────────────────────────────────────
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    # ── spawn a specific hero vehicle (Mercedes coupe) ───────────────
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.mercedes.coupe")
    if vehicle_bp is None:
        raise RuntimeError("Vehicle blueprint not found!")

    vehicle_bp.set_attribute("role_name", "hero")
    # optional: set color, driver id, etc.
    # vehicle_bp.set_attribute("color", "255,0,0")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Map has no spawn points!")

    spawn_point = random.choice(spawn_points)   # or pick index 0, etc.
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    if vehicle is None:
        raise RuntimeError("Failed to spawn the Mercedes coupe.")

    vehicle.set_autopilot(args.autopilot)
    print("Spawned hero:", vehicle.type_id)
    
    spectator = world.get_spectator()

    def _update_spectator() -> None:
        tf = vehicle.get_transform()
        tf.location += carla.Location(x=-8, z=3)
        tf.rotation.pitch = -10
        spectator.set_transform(tf)

    _update_spectator()

    # ── hood camera --------------------------------------------------
    #bbox   = vehicle.bounding_box
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(WIN_WIDTH))
    cam_bp.set_attribute("image_size_y", str(WIN_HEIGHT))
    #cam_tf = carla.Transform(
        #carla.Location(x=1.2 if bbox.extent.z < 1.5 else 2.0,
                       #z=max(1.4, min(2.8, bbox.extent.z * 1.6)))
    #)
    #cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_tf  = carla.Transform(
                carla.Location(x=0.30, y=0.0, z=1.30),   # same offsets
                carla.Rotation())                        # (0,0,0)
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

    #camera   = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    #cam_proc = CameraProcessor()
    cam_proc = CameraProcessor(developer=args.developer)
    camera.listen(cam_proc)
    detection_ran = (cam_proc.frame_id % RUN_EVERY_N == 0)        ### NEW
    # Warn Audio:
    BEEP = pygame.mixer.Sound(str(ASSETS_DIR / "beep.mp3"))   # load once, reuse forever
    BEEP.set_volume(0.7)                    # tweak loudness to taste



    # ── helpers & HUD ────────────────────────────────────────────────
    hud        = HUD(display)
    kb_control = KeyboardControl()
    debug_font = pygame.font.Font(pygame.font.get_default_font(), 24)
    tsr_active = True
    low_conf_streak = 0
    TSR_MISS_LIMIT = 10
    last_seen_ts     = 0.0
    tsr_warning_text = ""
    active_limit_kmh  = None          # final limit we obey
    active_conf      = 0.0
    source_txt        = "--"          # CAM, GPS, GPS>CAM …
    last_no_data_time = time.time()   # alert cooldown
    beep_channel = None
    beep_enabled = True 
    frames_without_detection = 0      # how many *consecutive frames* had NO box
    last_sign_was_weak       = False  # “weak” means raw_conf < 0.80
    SIGN_GONE_SEC      = 4.0       # how long the view must be blank
    last_seen_time     = 0.0       # unix-time of the *last* box (any conf)
    sign_open        = False      # “closed” means we have already handled
    sign_had_strong = False        # did this sign ever hit ≥0.80 ?

    # ───────────────── MAIN LOOP ─────────────────────────────────────
    try:
        while True:
            clock.tick_busy_loop(30)
            debug_message = ""

            # ── 1) handle GUI events + chase-cam ───────────────────────
            for e in pygame.event.get():
                _update_spectator()
                if e.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if e.type == pygame.KEYDOWN and (
                    e.key == K_ESCAPE or (e.key == K_q and pygame.key.get_mods() & KMOD_CTRL)
                ):
                    raise KeyboardInterrupt
                if e.type == pygame.KEYDOWN and e.key == K_p:
                    vehicle.set_autopilot(not _autopilot_enabled(vehicle))
                if e.type == pygame.KEYDOWN and e.key == K_n:
                    beep_enabled = not beep_enabled
                    if not beep_enabled and beep_channel and beep_channel.get_busy():
                        beep_channel.stop(); beep_channel = None
                

            # ── ISA decision: ≥ 80 % only──────────────
            cam_limit_kmh, cam_conf = None, 0.0
            if detection_ran:
                now = time.time()

                # a) YOLO found at least one box this pass
                if cam_proc.raw_conf > 0:
                    last_seen_time = now

                    if not sign_open:                    # new sign just appeared
                        sign_open       = True
                        sign_had_strong = False          # reset for this sign

                    if cam_proc.raw_conf >= CONF_THRESH:
                        sign_had_strong = True           # remember we had a strong hit
                        low_conf_streak = 0              # strong ⇒ wipe global streak

                # b) YOLO found no box – check how long view has been blank
                else:
                    blank_for = now - last_seen_time
                    if sign_open and blank_for >= SIGN_GONE_SEC:
                        # sign definitely left FOV
                        if not sign_had_strong:          # whole sign was weak
                            low_conf_streak += 1
                        sign_open = False                # ready for next sign

            # ------------------------------------------------------------------
            #  2) Disable TSR if too many consecutive weak signs
            # ------------------------------------------------------------------
            if low_conf_streak >= TSR_MISS_LIMIT and tsr_active:
                tsr_active       = False
                active_limit_kmh = None
                tsr_warning_text = (
                    f"TRAFFIC-SIGN RECOGNITION DISABLED — "
                    f"{low_conf_streak} consecutive low-confidence signs"
                )

            # ------------------------------------------------------------------
            #  3) Update current speed limit (only strong signs may do this)
            # ------------------------------------------------------------------
            if cam_proc.last_conf >= CONF_THRESH and tsr_active:
                active_limit_kmh = int(cam_proc.last_limit)
                active_conf      = cam_proc.last_conf
            # ───────────────────────────────────────────────────────────────────

            # ── 3) manual-drive speed guard ───────────────────────────
            if not _autopilot_enabled(vehicle):
                ctl = kb_control.parse()
                v   = vehicle.get_velocity()
                spd = math.sqrt(v.x**2 + v.y**2 + v.z**2)  # m/s

                overspeed = ( 
                    active_limit_kmh is not None and
                    spd*3.6 > active_limit_kmh
                )
                if overspeed and beep_enabled:
                    if beep_channel is None or not beep_channel.get_busy():
                        beep_channel = BEEP.play(loops=-1)
                else:
                    if beep_channel is not None and beep_channel.get_busy():
                        beep_channel.stop()
                        beep_channel = None
                
                vehicle.apply_control(ctl)
                debug_message = f"Speed {spd*3.6:.1f} km/h"
            else:
                debug_message = "Autopilot engaged"

            # ── 4) draw camera frame   ──────────────────────────────────
            if cam_proc.surface:
                display.blit(cam_proc.surface, (0, 0))

            # ── 5) HUD (speed & limit) ────────────────────────────────
            v = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
            icon_surf = SIGN_ICONS.get(active_limit_kmh) if tsr_active else None
            hud.draw(icon_surf)
            if tsr_active and active_limit_kmh is not None and icon_surf is None:
                label = hud.font_med.render(f"{active_limit_kmh} km/h", True, (255, 255, 255))
                display.blit(label, (WIN_WIDTH // 2 - label.get_width() // 2, 10))

            # ── 6) left status panel (no TTL, no GPS) ─────────────────
            if args.developer: 
                conf_var_pct = (active_conf - CONF_THRESH) * 100 if active_limit_kmh else 0
                left_lines = [
                    f"CONF : {active_conf*100:.1f}% (+{conf_var_pct:.1f}%)"
                    if active_limit_kmh else "CONF : --",

                ]
                # ____ Drawing the new req draw icon 
                text_y = 10 # fallback: no icon

                for i, ln in enumerate(left_lines):
                    display.blit(
                        debug_font.render(ln, True, (173, 216, 230)),
                        (10, text_y + 30 * i)
                    )
                status_txt = "BEEP: ON" if beep_enabled else "BEEP: OFF"
                display.blit(
                    debug_font.render(status_txt, True, (250, 250, 210)),
                    (10, WIN_HEIGHT - 30)
                )
            # ── 8) centre-screen TSR shut-down warning ─────────────
            if tsr_warning_text:
                warn = debug_font.render(tsr_warning_text, True, (255, 0, 0))
                display.blit(
                    warn,
                    (WIN_WIDTH // 2 - warn.get_width() // 2,
                    WIN_HEIGHT // 2 - warn.get_height() // 2)
                )
             # ── 7) bottom-right debug line Speedometer ────────────────────────────
            if debug_message:
                dbg = debug_font.render(debug_message, True, (255, 255, 255))
                display.blit(
                    dbg,
                    (WIN_WIDTH - dbg.get_width() - 10,
                    WIN_HEIGHT - dbg.get_height() - 10)
                )
            
            # ── 9) diagnostic overlay ─────────────────────────────────────────
            if args.developer:
                diag_lines = [
                    f"TSR active: {tsr_active}",
                    f"raw_conf          : {cam_proc.raw_conf:.2f}",   # 〈– new — shows every sign
                    f"strong_conf (≥80%): {cam_proc.last_conf:.2f}",  # optional rename; same as before
                    f"low-conf streak : {low_conf_streak}/{TSR_MISS_LIMIT}",
                    ]
                for i, txt in enumerate(diag_lines):
                    display.blit(
                    debug_font.render(txt, True, (200, 200, 200)),
                    (WIN_WIDTH - 320, 10 + 25 * i)      # right-top corner
                    )
            pygame.display.flip()





    except KeyboardInterrupt:
        pass
    finally:
        camera.stop(); camera.destroy()
        vehicle.destroy()
        pygame.quit()

if __name__ == "__main__":
    main()