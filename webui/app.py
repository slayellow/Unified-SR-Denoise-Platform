import streamlit as st
import glob
import os
from ui_utils import (
    read_yaml, write_yaml, dict_to_yaml_str, yaml_str_to_dict, 
    stream_command, apply_apple_design, render_dynamic_ui, DEFAULT_TEMPLATES
)

# App settings
st.set_page_config(page_title="Unified MLOps Platform", layout="centered", initial_sidebar_state="expanded")

# Apply Apple Design CSS
apply_apple_design(st)

st.title("Unified SR/Denoise Platform")
st.markdown(" **Apple-inspired MLOps Dashboard** | Consumer electronics aesthetic")

menu = st.sidebar.radio("Navigation", ["1. Configuration Manager", "2. Training Dashboard", "3. Inference & Viewer"])

if menu == "1. Configuration Manager":
    st.header("Configuration Manager")
    st.markdown("Select a category to manage your `.yaml` configurations or create new ones.")

    # 1. Category Selector
    category = st.radio("Configuration Category", ["train", "finetune", "data", "aimet"], horizontal=True)
    
    cat_dir = os.path.join("../configs", category)
    os.makedirs(cat_dir, exist_ok=True)
    
    # + Create New Config Section
    with st.expander(f"✨ Create New '{category.capitalize()}' Config"):
        st.markdown(f"Generate a new configuration based on standard {category} templates.")
        new_filename = st.text_input("New File Name (without .yaml)", value="new_config")
        if st.button("Create Config"):
            target_path = os.path.join(cat_dir, f"{new_filename}.yaml")
            if os.path.exists(target_path):
                st.error("File already exists!")
            else:
                write_yaml(DEFAULT_TEMPLATES.get(category, {}), target_path)
                st.toast(f"✅ Created {new_filename}.yaml", icon="🎉")
                st.rerun()

    # 2. File Selector
    config_files = glob.glob(f"{cat_dir}/**/*.yaml", recursive=True)
    if not config_files:
        st.info(f"No configuration files found in configs/{category}/. Create a new one above!")
    else:
        selected_file = st.selectbox("Select Target Config", config_files)
        config_data = read_yaml(selected_file)

        st.divider()
        
        # 3. View Mode Toggle
        view_mode = st.radio("Editor Mode", ["🎨 Beautiful UI", "📝 Raw YAML Text"], horizontal=True)
        
        if view_mode == "📝 Raw YAML Text":
            st.markdown(f"Editing `{os.path.basename(selected_file)}` as raw text.")
            raw_text = st.text_area("YAML Document", value=dict_to_yaml_str(config_data), height=500)
            
            if st.button("Save Raw YAML"):
                new_dict = yaml_str_to_dict(raw_text)
                if "Error" in new_dict:
                    st.error(new_dict["Error"])
                else:
                    write_yaml(new_dict, selected_file)
                    st.toast("✅ Raw YAML updated successfully!", icon="💾")
                    
        else:
            st.markdown(f"Dynamic Settings for `{os.path.basename(selected_file)}`")
            with st.form("dynamic_config_form"):
                
                # Dynamically Render Sliders, Toggles, and Inputs from dictionary
                unique_prefix = f"UI_{os.path.basename(selected_file)}"
                updated_dict = render_dynamic_ui(st, config_data, prefix=unique_prefix)
                
                submit = st.form_submit_button("Save Configuration ✨")
                if submit:
                    write_yaml(updated_dict, selected_file)
                    st.toast("✅ Settings applied and saved!", icon="💾")

elif menu == "2. Training Dashboard":
    st.header("Training Dashboard")
    st.markdown("Execute distributed training and monitor logs natively.")

    # Show only train and finetune
    train_files = glob.glob("../configs/train/**/*.yaml", recursive=True)
    train_files += glob.glob("../configs/finetune/**/*.yaml", recursive=True)
    
    selected_file = st.selectbox("Config File to Train", train_files)
    
    col1, col2 = st.columns(2)
    with col1:
        gpus = st.text_input("CUDA_VISIBLE_DEVICES", value="3,4", help="Comma separated GPU IDs")
    with col2:
        num_processes = st.number_input("Num Processes (GPUs)", value=2, min_value=1, step=1)

    if st.button("Start Training"):
        cmd = f"CUDA_VISIBLE_DEVICES={gpus} accelerate launch --num_processes={num_processes} ../tools/train.py --config {selected_file}"
        st.toast(f"🚀 Launching Training: {os.path.basename(selected_file)}...", icon="🔄")
        
        log_view = st.empty()
        log_view.code("Initializing trainer... waiting for output")
        
        return_code = stream_command(cmd, log_view)
        
        if return_code == 0:
            st.toast("✅ Training Completed Successfully!", icon="🎉")
        else:
            st.toast("❌ Error occurred during training. Check logs.", icon="🚨")

elif menu == "3. Inference & Viewer":
    st.header("Inference & Output Viewer")
    st.markdown("Run inference on validation datasets and compare visually.")
    
    # 1. Config includes both train and finetune
    config_files = glob.glob("../configs/train/**/*.yaml", recursive=True)
    config_files += glob.glob("../configs/finetune/**/*.yaml", recursive=True)
    
    # 2. Checkpoint only best.pth
    ckpt_files = glob.glob("../checkpoints/**/best.pth", recursive=True)
    
    col_cfg, col_ckpt = st.columns(2)
    with col_cfg:
        selected_cfg = st.selectbox("🎯 Target Config", config_files) if config_files else None
    with col_ckpt:
        selected_ckpt = st.selectbox("💾 Checkpoint", ckpt_files) if ckpt_files else None
    
    st.divider()
    
    st.subheader("📂 I/O Directories")
    
    input_dir = st.text_input(
        "📥 Input Image Directory (Absolute Path)", 
        value="/mnt/data_server/etc/jshong/SuperResolution/Pretrained_Dataset/val_denoise/LR",
        help="복원/추론할 이미지들이 들어있는 로컬 또는 마운트된 서버폴더의 절대경로를 입력하세요."
    )
    
    output_dir = st.text_input(
        "📤 Output Saving Directory (Absolute Path)", 
        value="../results/output/",
        help="결과 이미지를 저장할 경로를 입력하세요."
    )
    
    if st.button("Run Inference"):
        if selected_cfg and selected_ckpt:
            cmd = f"python ../tools/inference.py --config {selected_cfg} --checkpoint {selected_ckpt} --input {input_dir} --output_dir {output_dir}"
            st.toast("🔍 Running Inference Engine...", icon="⚙️")
            
            run_view = st.empty()
            code = stream_command(cmd, run_view)
            
            if code == 0:
                st.toast("✅ Inference Complete!", icon="🖼️")
                out_images = glob.glob(os.path.join(output_dir, "*.png"))
                if out_images:
                    st.image(out_images[0], caption=f"Sample Output: {os.path.basename(out_images[0])}", use_column_width=True)
            else:
                st.toast("❌ Inference Failed.", icon="🚨")
        else:
            st.warning("Please select a config and checkpoint.")
