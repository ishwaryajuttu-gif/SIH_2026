"""Level-2 dataset collection: record frames of YOUR mock BAS set-up from the webcam.

    python tools/record_dataset.py --name session1            # SPACE = save frame, A = auto mode, Q = quit
    python tools/record_dataset.py --name session1 --auto 0.5 # save one frame every 0.5 s automatically

Tips for a good dataset (aim for 300-800 images per session):
  * vary lighting (lamp on/off, window), camera angle and distance
  * include hands touching, holding, moving every object, AND frames with no hands
  * include look-alike distractors so the model learns what NOT to detect
  * record some "unsafe" clips: hand in the restricted area, touching the restricted tool
"""
import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=time.strftime("session_%Y%m%d_%H%M%S"))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto", type=float, default=0.0, help="seconds between automatic saves (0 = manual)")
    args = ap.parse_args()

    out = ROOT / "dataset" / "raw" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    auto = args.auto > 0
    last = 0.0
    n = len(list(out.glob("*.jpg")))
    print(f"Saving to {out}  (SPACE=save  A=toggle auto  Q=quit)")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        save = key == ord(" ") or (auto and now - last >= (args.auto or 0.5))
        if key == ord("a"):
            auto = not auto
        if key == ord("q"):
            break
        if save:
            n += 1
            cv2.imwrite(str(out / f"{args.name}_{n:05d}.jpg"), frame)
            last = now
        view = frame.copy()
        cv2.putText(view, f"saved: {n}  auto: {'ON' if auto else 'off'}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if auto else (255, 255, 255), 2)
        cv2.imshow("BAS dataset recorder", view)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done - {n} images in {out}")


if __name__ == "__main__":
    main()
