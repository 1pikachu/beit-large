from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import os
import math
import time
import argparse
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')
    parser.add_argument('--image_path', default="/home2//pytorch-broad-models/beit-large/000000039769.jpg")
    parser.add_argument('--model_path', default="/home2//pytorch-broad-models/beit-large/models")
    parser.add_argument('--dataset', default="/home2//pytorch-broad-models/imagenet/raw")
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()
    print(args)
    return args

def test(args, val_loader, model):
    if args.nv_fuser:
       fuser_mode = "fuser2"
    else:
       fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    if args.jit:
        for image, _ in val_loader:
            image = image.to(args.device)
            if args.channels_last:
                image = image.to(memory_format=torch.channels_last)
            try:
                model = torch.jit.trace(model, image, check_trace=False, strict=False)
                print("---- JIT trace enable.")
                model = torch.jit.freeze(model)
            except (RuntimeError, TypeError) as e:
                print("---- JIT trace disable.")
                print("failed to use PyTorch jit mode due to: ", e)
            break

    total_time = 0.0
    total_sample = 0

    profile_len = min(len(val_loader), args.num_iter) // 2
    if args.profile and args.device == "xpu":
        for i, (image, _) in enumerate(val_loader):
            if i >= args.num_iter:
                break
            if args.channels_last:
                image = image.to(memory_format=torch.channels_last)

            with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
                elapsed = time.time()
                image = image.to(args.device)
                outputs = model(image)
                torch.xpu.synchronize()
                elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
            if args.profile and i == profile_len:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, (image, _) in enumerate(val_loader):
                if i >= args.num_iter:
                    break
                if args.channels_last:
                    image = image.to(memory_format=torch.channels_last)

                elapsed = time.time()
                image = image.to(args.device)
                with torch.jit.fuser(fuser_mode):
                    outputs = model(image)
                torch.cuda.synchronize()
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i, (image, _) in enumerate(val_loader):
                if i >= args.num_iter:
                    break
                if args.channels_last:
                    image = image.to(memory_format=torch.channels_last)

                elapsed = time.time()
                image = image.to(args.device)
                outputs = model(image)
                elapsed = time.time() - elapsed
                p.step()
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_sample += args.batch_size
                    total_time += elapsed
    elif not args.profile and args.device == "cuda":
        for i, (image, _) in enumerate(val_loader):
            if i >= args.num_iter:
                break
            if args.channels_last:
                image = image.to(memory_format=torch.channels_last)

            elapsed = time.time()
            image = image.to(args.device)
            with torch.jit.fuser(fuser_mode):
                outputs = model(image)
            torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed
    else:
        for i, (image, _) in enumerate(val_loader):
            if i >= args.num_iter:
                break
            if args.channels_last:
                image = image.to(memory_format=torch.channels_last)

            elapsed = time.time()
            image = image.to(args.device)
            outputs = model(image)
            if args.device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_sample += args.batch_size
                total_time += elapsed

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput)) 


def main():
    args = parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    # image = Image.open(args.image_path)
    # feature_extractor = BeitFeatureExtractor.from_pretrained(args.model_path)
    # inputs = feature_extractor(images=image, return_tensors="pt")
    model = BeitForImageClassification.from_pretrained(args.model_path)
    model = model.to(args.device)
    model.eval()

    # dataset
    valdir = os.path.join(args.dataset, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])
    print('Using image size', args.image_size)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- use NHWC format")

    with torch.no_grad():
        model.eval()
        if args.device == "xpu":
            datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
            model = torch.xpu.optimize(model=model, dtype=datatype)
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                test(args, val_loader, model)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                test(args, val_loader, model)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                test(args, val_loader, model)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                test(args, val_loader, model)
        else:
            print("---- no autocast")
            test(args, val_loader, model)

if __name__ == "__main__":
    main()
