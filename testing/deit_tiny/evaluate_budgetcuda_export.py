"""Measure the accuracy impact of shared-scale packed W4 accelerator weights."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import torch, timm
from quantization import prepare_qat_model, set_observers
from train_deit_tiny import CLASS_NAMES, build_loaders, evaluate


def main():
    here=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,default=here/"artifacts"/"w4a8"/"best_fake_quant.pt");parser.add_argument("--manifest",type=Path,default=here/"splits"/"manifest.csv");parser.add_argument("--image-root",type=Path,default=here/"data"/"cache_256");parser.add_argument("--batch-size",type=int,default=32);parser.add_argument("--weight-bits",type=int,choices=(4,8),default=8);parser.add_argument("--output",type=Path,default=here/"artifacts"/"budgetcuda_w8a8"/"accuracy.json");args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True);base=timm.create_model(checkpoint["model_name"],pretrained=False,num_classes=len(CLASS_NAMES));model=prepare_qat_model(base,4,activation_bits=8);model.load_state_dict(checkpoint["state_dict"],strict=True)
    converted=0
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module,"weight_quant") and hasattr(module,"weight"):
                limit=(1<<(args.weight_bits-1))-1;scale=(module.weight.detach().abs().max()/limit).clamp_min(1e-12);module.weight.copy_(torch.round(module.weight/scale).clamp(-limit-1,limit)*scale);module.weight_quant.fake_quant_enabled=False;converted+=1
    set_observers(model,False);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model.to(device).eval();loader_args=SimpleNamespace(manifest=args.manifest,image_root=args.image_root,batch_size=args.batch_size,workers=0,device=device);_,_,test_loader=build_loaders(loader_args,checkpoint.get("seed",42));metrics=evaluate(model,test_loader,device)
    result={"device":str(device),"weight_bits":args.weight_bits,"shared_scale_modules":converted,"samples":len(test_loader.dataset),"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"],"per_class_accuracy":metrics["per_class_accuracy"],"source_checkpoint_accuracy":checkpoint.get("metrics",{}).get("test",{}).get("accuracy")};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
