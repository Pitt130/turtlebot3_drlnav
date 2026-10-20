"""Top-down recording of the Isaac runtime, for watching episodes afterwards.

A near-orthographic camera looks straight down on the arena (image up = +y).
Frames are rendered with SimulationContext.render(), which disables physics
for that app update, so recording does not change the simulation. Each frame
gets the lidar rays and a status line drawn on top and is written as a JPEG on
a worker thread; make_video() turns the frames into an MP4.
"""
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

CAMERA_PATH = "/World/record_camera"
CAMERA_HEIGHT = 30.0
HORIZONTAL_APERTURE = 20.955           # mm, the USD camera default


class TopDownRecorder:
    def __init__(self, world, out_dir, half_width, size=720, center=(0.0, 0.0)):
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.world, self.size, self.center, self.half_width = world, size, center, half_width
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        camera = UsdGeom.Camera.Define(world.stage, CAMERA_PATH)
        camera.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], CAMERA_HEIGHT))
        camera.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
        camera.CreateVerticalApertureAttr(HORIZONTAL_APERTURE)
        camera.CreateFocalLengthAttr(HORIZONTAL_APERTURE * CAMERA_HEIGHT / (2 * half_width))
        camera.CreateClippingRangeAttr(Gf.Vec2f(CAMERA_HEIGHT - 2.0, CAMERA_HEIGHT + 1.0))
        self.render_product = rep.create.render_product(CAMERA_PATH, (size, size))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product])
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.frames = 0

    def warm_up(self, updates=30):
        for _ in range(updates):
            self.world.render()

    def pixel(self, x, y):
        scale = self.size / (2 * self.half_width)
        return (self.size / 2 + (x - self.center[0]) * scale, self.size / 2 - (y - self.center[1]) * scale)

    def capture(self, text, scan_origin=None, scan_yaw=0.0, ranges=(), angles=()):
        """Render the current state; scan_origin/yaw place the lidar rays in the world."""
        self.world.render()
        rgba = self.annotator.get_data()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return
        rays, robot = [], None
        if scan_origin is not None:
            ox, oy = scan_origin
            for r, a in zip(ranges, angles):
                hit = math.isfinite(r)
                d = r if hit else 3.5
                rays.append((self.pixel(ox, oy), self.pixel(ox + d * math.cos(scan_yaw + a),
                                                            oy + d * math.sin(scan_yaw + a)), hit))
            c, s = math.cos(scan_yaw), math.sin(scan_yaw)
            robot = [self.pixel(ox + dx * c - dy * s, oy + dx * s + dy * c)
                     for dx, dy in ((0.20, 0.0), (-0.06, 0.09), (-0.06, -0.09))]
        path = self.out_dir / f"{self.frames:06d}.jpg"
        self.frames += 1
        self.pool.submit(_write_frame, rgba[..., :3].copy(), path, text, rays, robot)

    def close(self):
        self.pool.shutdown(wait=True)
        self.annotator.detach()


def _write_frame(rgb, path, text, rays, robot):
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for start, end, hit in rays:
        draw.line([start, end], fill=(230, 40, 40) if hit else (240, 170, 170), width=1)
        if hit:
            draw.ellipse([end[0] - 2, end[1] - 2, end[0] + 2, end[1] + 2], fill=(230, 40, 40))
    if robot:
        draw.polygon(robot, fill=(255, 210, 0), outline=(0, 0, 0))
    draw.rectangle([0, 0, image.width, 24], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    image.save(path, quality=90)


def make_video(frame_dir, out_file, fps=10):
    import subprocess
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(Path(frame_dir) / "%06d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "23", str(out_file)], check=True)


if __name__ == "__main__":
    import sys
    make_video(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
