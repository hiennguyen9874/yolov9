import argparse
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.experimental_seg import End2End, End2EndRoialign
from models.yolo import (
    ClassificationModel,
    DDetect,
    Detect,
    DetectionModel,
    DualDDetect,
    DualDetect,
    SegmentationModel,
)
from utils.general import (
    check_img_size,
    check_requirements,
    colorstr,
    file_size,
    get_default_args,
    LOGGER,
    print_args,
    Profile,
    url2file,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment


def export_formats():
    # YOLO export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    # YOLO export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(
                f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)"
            )
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


# @try_export
def export_onnx(
    model,
    im,
    file,
    opset,
    dynamic,
    dynamic_batch,
    simplify,
    end2end,
    trt,
    topk_all,
    device,
    iou_thres,
    score_thres,
    mask_resolution,
    pooler_scale,
    sampling_ratio,
    image_size,
    cleanup,
    roi_align,
    roi_align_type,
    prefix=colorstr("ONNX:"),
):
    # YOLO ONNX export
    check_requirements("onnx")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output", "proto"]
    dynamic_axes = None

    if dynamic:
        raise NotImplementedError

    if end2end:
        if trt:
            output_names = [
                "num_dets",
                "det_boxes",
                "det_scores",
                "det_classes",
                "det_masks",
            ]
        else:
            output_names = ["output"]

    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}}  # shape(1,3,640,640)
        output_axes = {
            "output": {0: "batch"},
            "proto": {0: "batch"},
        }

        if end2end:
            if trt:
                output_axes = {
                    "num_dets": {0: "batch"},
                    "det_boxes": {0: "batch"},
                    "det_scores": {0: "batch"},
                    "det_classes": {0: "batch"},
                    "det_masks": {0: "batch"},
                }
            else:
                output_axes = {
                    "output": {0: "num_dets"},
                }
        dynamic_axes.update(output_axes)

    if end2end:
        if roi_align:
            model = End2EndRoialign(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                mask_resolution=mask_resolution,
                pooler_scale=pooler_scale,
                sampling_ratio=sampling_ratio,
                device=device,
                trt=trt,
                max_wh=max(image_size),
                roi_align_type=roi_align_type,
            )
        else:
            model = End2End(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                pooler_scale=pooler_scale,
                device=device,
                trt=trt,
                max_wh=max(image_size),
            )

    torch.onnx.export(
        (model.cpu() if (dynamic or dynamic_batch) else model),
        im.cpu() if (dynamic or dynamic_batch) else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes if (dynamic or dynamic_batch) else None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {
        "stride": int(max(model.model.stride if end2end else model.stride)),
        "names": model.model.names if end2end else model.names,
    }
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(
                ("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1")
            )
            import onnxsim

            LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")

    if cleanup:
        try:
            LOGGER.info("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"Cleanup failure: {e}")

    return f, model_onnx


@smart_inference_mode()
def run(
    data=ROOT / "data/coco.yaml",  # 'dataset.yaml path'
    weights=ROOT / "yolo.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLO Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
    mask_resolution=56,
    pooler_scale=0.25,
    sampling_ratio=0,
    dynamic_batch=False,
    end2end=False,
    trt=False,
    cleanup=False,
    roi_align=False,
    roi_align_type=0,
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    (onnx,) = flags  # export booleans
    file = Path(
        url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights
    )  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert (
            device.type != "cpu" or coreml
        ), "--half only compatible with GPU export, i.e. use --device 0"
        assert (
            not dynamic
        ), "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert (
            device.type == "cpu"
        ), "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, DDetect, DualDetect, DualDDetect)):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)  # model output shape
    metadata = {
        "stride": int(max(model.stride)),
        "names": model.names,
    }  # model metadata
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"
    )

    # Exports
    f = [""] * len(fmts)  # exported filenames
    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )  # suppress TracerWarning
    if onnx:  # OpenVINO requires ONNX
        f[0], _ = export_onnx(
            model=model,
            im=im,
            file=file,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            cleanup=cleanup,
            dynamic_batch=dynamic_batch,
            topk_all=topk_all,
            device=device,
            iou_thres=iou_thres,
            score_thres=conf_thres,
            image_size=imgsz,
            mask_resolution=mask_resolution,
            pooler_scale=pooler_scale,
            sampling_ratio=sampling_ratio,
            roi_align=roi_align,
            end2end=end2end,
            trt=trt,
            roi_align_type=roi_align_type,
        )

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (
            isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel)
        )  # type
        dir = Path("segment" if seg else "classify" if cls else "")
        h = "--half" if half else ""  # --half FP16 inference arg
        s = (
            "# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference"
            if cls
            else (
                "# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference"
                if seg
                else ""
            )
        )
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolo.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLO Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=14, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument(
        "--topk-per-class",
        type=int,
        default=100,
        help="TF.js NMS: topk per class to keep",
    )
    parser.add_argument(
        "--topk-all",
        type=int,
        default=100,
        help="ONNX END2END/TF.js NMS: topk for all classes to keep",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="ONNX END2END/TF.js NMS: IoU threshold",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="ONNX END2END/TF.js NMS: confidence threshold",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx"],
        help="onnx",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="ONNX: using onnx_graphsurgeon to cleanup",
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true", help="ONNX/TF/TensorRT: dynamic axes"
    )
    parser.add_argument(
        "--mask-resolution", type=int, default=56, help="ONNX: Roialign mask-resolution"
    )
    parser.add_argument(
        "--pooler-scale",
        type=float,
        default=0.25,
        help="ONNX: Roialign scale, scale = proto shape / input shape",
    )
    parser.add_argument(
        "--sampling-ratio", type=int, default=0, help="ONNX: Roialign sampling ratio"
    )
    parser.add_argument(
        "--roi-align",
        action="store_true",
        help="ONNX: Crop And Resize mask using roialign",
    )
    parser.add_argument("--end2end", action="store_true", help="ONNX: NMS")
    parser.add_argument("--trt", action="store_true", help="ONNX: TRT")

    parser.add_argument(
        "--roi-align-type",
        type=int,
        default=0,
        help="ONNX: Roialign type, 0: RoiAlign, 1: RoIAlignDynamic_TRT, 2: RoIAlign2Dynamic_TRT",
    )
    opt = parser.parse_args()

    if "onnx_end2end" in opt.include:
        opt.simplify = True
        opt.dynamic = True
        opt.inplace = True
        opt.half = False

    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
