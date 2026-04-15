import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import onnx

from src.models import build_model

def get_args():
    parser = argparse.ArgumentParser(description="Unified Model Export (ONNX)")
    parser.add_argument("--config", type=str, required=True, help="Config file used for training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth). If None, exports using initialized weights.")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX Folder Path (default: ./results)")
    parser.add_argument("--height", type=int, default=256, help="Input Height")
    parser.add_argument("--width", type=int, default=256, help="Input Width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX Opset Version")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tracing (cpu recommended for export)")
    parser.add_argument("--sim", action='store_true', help="Use onnx-simplifier if available")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)
    
    # 1. Load Config
    print(f"Loading config: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 2. Build Model
    print(f"Building model: {config['model']['name']}...")
    model = build_model(config['model']).to(device)
    
    # 3. Load Weights
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        # User owns the checkpoint, so we can trust it.
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, torch.nn.Module):
            # If the checkpoint is the entire model (e.g. from AIMET), use it directly
            print("[Info] Loaded full model object.")
            # We might need to move it to device if weights_only=False loaded it to CPU by default or similar
            model = checkpoint.to(device)
        else:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
    else:
        print(f"[Info] No checkpoint provided. Exporting {config['model']['name']} with randomly initialized weights.")
    
    # 4. Switch to Deploy Mode
    if hasattr(model, 'switch_to_deploy'):
        print("Switching model to deploy mode (merging branches)...")
        model.switch_to_deploy()
    
    model.eval()

    # 5. Prepare Input (Batch=1, Ch=3, H, W)
    input_shape = (1, 3, args.height, args.width)
    dummy_input = torch.randn(input_shape, device=device)
    
    task = config.get('task', 'sr')
    dummy_guide = None
    if task == 'guide':
        dummy_guide = torch.randn(input_shape, device=device)
        export_inputs = (dummy_input, dummy_guide)
        input_names_list = ['input', 'guide']
    else:
        export_inputs = (dummy_input, )
        input_names_list = ['input']
    
    # 5-1. Calculate FLOPs & Params
    try:
        import thop
        print(f"Calculating GFLOPs and Params with input shape {input_shape}...")
        # thop.profile modifies the model hooks, so we might want to do it on a copy or reset
        # But for export it's usually fine as long as we don't restart training.
        # Note: thop prints to stdout by default, we capture it or just let it print.
        flops, params = thop.profile(model, inputs=export_inputs, verbose=False)
        
        gflops = flops / 1e9
        mparams = params / 1e6
        
        print("-" * 30)
        print(f"GFLOPs: {gflops:.4f}")
        print(f"Params: {mparams:.4f} M")
        print("-" * 30)
    except ImportError:
        print("['thop' not found] Skipping GFLOPs/Params calculation. (pip install thop)")
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
    
    # 6. Determine Output Path
    if args.checkpoint:
        ckpt_base = os.path.splitext(os.path.basename(args.checkpoint))[0]
    else:
        ckpt_base = f"{config['model']['name']}_init"
    if args.output:
        output_path = os.path.join(args.output, f"{ckpt_base}_{args.height}x{args.width}.onnx")
        os.makedirs(args.output, exist_ok=True)
    else:
        output_path = os.path.join("results", f"{ckpt_base}_{args.height}x{args.width}.onnx")
        os.makedirs("results", exist_ok=True)
    
    print(f"Exporting to {output_path} with input shape {input_shape}...")
    
    # 7. ONNX Export
    try:
        torch.onnx.export(
            model,
            export_inputs,
            output_path,
            opset_version=args.opset,
            input_names=input_names_list,
            output_names=['output'],
            # dynamic_axes removed for static shape export
        )
        print("ONNX export successful.")
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return

    # 8. Verification
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("Verification passed.")
    
    # 9. Simplification (Optional)
    if args.sim:
        sim_success = False
        # Try onnxslim first (User has this installed)
        try:
            import onnxslim
            print("Running ONNX Slim (onnxslim)...")
            # onnxslim.slim returns the optimized model
            model_simp = onnxslim.slim(onnx_model)
            onnx.save(model_simp, output_path)
            print("ONNX simplified successfully with onnxslim.")
            sim_success = True
        except ImportError:
            pass
        except Exception as e:
            print(f"Error during onnxslim: {e}")

        # Fallback to onnx-simplifier
        if not sim_success:
            try:
                from onnxsim import simplify
                print("Running ONNX Simplifier (onnxsim)...")
                model_simp, check = simplify(onnx_model)
                if check:
                    onnx.save(model_simp, output_path)
                    print("ONNX simplified successfully with onnxsim.")
                else:
                    print("ONNX simplification check failed.")
            except ImportError:
                if not sim_success:
                    print("Neither onnxslim nor onnx-simplifier is installed. Skipping simplification.")
            except Exception as e:
                print(f"Error during onnx-simplifier: {e}")

if __name__ == "__main__":
    main()
